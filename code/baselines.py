import numpy as np
import networkx as nx
import torch
    
from ortools.linear_solver import pywraplp

def top_k_firms(w, P, k):
    w_preferences = P[:, w, :]
    top_k = torch.topk(w_preferences, k, dim=-1)[1]
    return top_k

def top_k_workers(f, Q, k):
    f_preferences = Q[:, :, f]
    top_k = torch.topk(f_preferences, k, dim=-1)[1]
    return top_k


################################################
#Optimal efficiency
################################################

#For workers
def optimal_efficiency_for_workers(P, k):

    instances, num_agents = P.shape[0], P.shape[1]
    R = torch.zeros(P.shape)

    for instance in range(instances):

        G = nx.Graph()

        workers = list(range(num_agents))
        firms   = list(range(num_agents, 2*num_agents))

        G.add_nodes_from(workers, bipartite=0)
        G.add_nodes_from(firms, bipartite=1)

        for w in workers:
            for f in top_k_firms(w, P, k)[instance]:
                if P[instance, w, int(f)] > 0:
                    G.add_edge(w, num_agents + int(f))

        matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=workers)

        for w in workers:
            if w in matching:
                firm_index = matching[w] - num_agents
                R[instance, w, firm_index] = 1

    return R

#For firms
def optimal_efficiency_for_firms(Q, k):

    instances, num_agents = Q.shape[0], Q.shape[1]
    R = torch.zeros(Q.shape)

    for instance in range(instances):

        G = nx.Graph()

        firms = list(range(num_agents))
        workers   = list(range(num_agents, 2*num_agents))

        G.add_nodes_from(firms, bipartite=0)
        G.add_nodes_from(workers, bipartite=1)

        for f in firms:
            for w in top_k_workers(f, Q, k)[instance]:
                if Q[instance, w, int(f)] > 0:
                    G.add_edge(f, num_agents + int(w))

        matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=firms)

        for f in firms:
            if f in matching:
                worker_index = matching[f] - num_agents
                R[instance, worker_index, f] = 1

    return R


def optimal_efficiency_for_both(P, Q, k):

    instances, n = P.shape[0], P.shape[1]
    R = torch.zeros_like(P)

    for instance in range(instances):

        G = nx.Graph()

        workers = list(range(n))
        firms = list(range(n, 2*n))

        G.add_nodes_from(workers, bipartite=0)
        G.add_nodes_from(firms, bipartite=1)

        # precompute top-k
        topk_firms = torch.topk(P[instance], k, dim=-1).indices
        topk_workers = torch.topk(Q[instance], k, dim=0).indices

        for w in range(n):
            for f in range(n):

                weight = 0

                if f in topk_firms[w]:
                    weight += 1

                if w in topk_workers[:, f]:
                    weight += 1

                if weight > 0:
                    G.add_edge(
                        w,
                        n + f,
                        weight=weight
                    )

        matching = nx.algorithms.matching.max_weight_matching(
            G,
            maxcardinality=True
        )

        for u, v in matching:

            if u < n:
                w = u
                f = v - n
            else:
                w = v
                f = u - n

            R[instance, w, f] = 1

    return R


################################################
#Optimal Fairness
################################################

def optimal_fairness(P, k, group_A):
    """
    workers: list of worker ids
    firms: list of firm ids
    top_k: dict {worker: list of acceptable firms}
    group_A: set of workers in group A
    """
    instances, num_agents = P.shape[0], P.shape[1]
    agents = list(range(num_agents))

    R = torch.zeros(P.shape)

    solver = pywraplp.Solver.CreateSolver("GLOP")

    WA = set(group_A)
    WB = set(agents) - WA

    alpha = 1 / len(WA)
    beta = 1 / len(WB)

    for instance in range(instances):

        # Create variables only for existing edges
        x = {}
        for w in agents:
            for f in top_k_firms(w, P, k)[instance]:
                if P[instance, w, int(f)] > 0:
                    x[(w, int(f))] = solver.NumVar(0, 1, f"x_{w}_{f}")

        # Fairness auxiliary variable
        t = solver.NumVar(0, solver.infinity(), "t")

        # --- Matching constraints ---

        # Each worker matched at most once
        for w in agents:
            solver.Add(
                sum(x[(w, int(f))] for f in agents if (w, f) in x) <= 1
            )

        # Each firm matched at most once
        for f in agents:
            solver.Add(
                sum(x[(w, f)] for w in agents if (w, f) in x) <= 1
            )

        # --- Fairness expressions ---
        M_A = sum(x[(w, f)] for (w, f) in x if w in WA)
        M_B = sum(x[(w, f)] for (w, f) in x if w in WB)

        # Fairness constraints
        solver.Add(alpha * M_A - beta * M_B <= t)
        solver.Add(beta * M_B - alpha * M_A <= t)

        M_total = solver.Sum(x[(w, f)] for (w, f) in x)
        solver.Maximize(M_total)
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("No optimal solution found in step 1.")

        # Store maximum matching size
        M_star = M_total.solution_value()

        # Clear previous objective
        solver.Objective().Clear()

        # Enforce maximum matching size
        solver.Add(M_total == M_star)

        # Second objective: minimize fairness gap
        solver.Minimize(t)

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("No optimal solution found in step 2.")

        # Extract matching
        matching = [
            (w, f)
            for (w, f), var in x.items()
            if var.solution_value() > 0.5
        ]

        for (w, f) in matching:
            R[instance, w, f] = 1
    
    return R


################################################
#Deferred Acceptance
################################################
def DA(P, Q, menPreferences, womenPreferences):
    
    num_instances, num_agents = P.shape[0], P.shape[1]
    R = torch.zeros(P.shape)
    
    for inst in range(num_instances):
        
        # Start with no married men
        unmarriedMen = list(range(num_agents))
        
        # No Spouse Yet
        manSpouse, womanSpouse = [-1] * num_agents, [-1] * num_agents
        
        # Ptr to index of top choice
        nextManchoice = [0] * num_agents
        
        while unmarriedMen:
            
            he = unmarriedMen[0] 
            
            # He is out of choices, single    
            if nextManchoice[he] == num_agents: 
                manSpouse[he] = num_agents
                unmarriedMen.pop(0)
                continue
                
            she = menPreferences[inst, he, nextManchoice[he]]
            
            # He prefers being single than his top choice: Stay single
            if P[inst, he, she] < 0:
                manSpouse[he] = num_agents
                unmarriedMen.pop(0)
                continue
                            
            # Top Choice is not Married
            if womanSpouse[she] == -1:  
                # She prefers being married rather than being single, so she accepts
                if Q[inst, he, she] > 0: 
                    womanSpouse[she], manSpouse[he] = he, she
                    R[inst, he, she] = 1
                    unmarriedMen.pop(0)
            else:
                # She prefers this man over her current husband, break-up, accept proposal  
                currentHusband = womanSpouse[she]
                if Q[inst, he, she] > Q[inst, currentHusband, she]:
                    womanSpouse[she], manSpouse[he] = he, she
                    R[inst, he, she] = 1
                    R[inst, currentHusband, she] = 0                    
                    unmarriedMen[0] = currentHusband

            nextManchoice[he] = nextManchoice[he] + 1
                       
    return R

# Worker-Proposing DA
def compute_DA_batch(P, Q):
    menPreferences = torch.argsort(-P, axis = -1)
    womenPreferences = torch.argsort(-Q, axis = -2)
    return DA(P, Q, menPreferences, womenPreferences)

# Firm-Proposing DA
def compute_DA_batch_switch(P, Q):
    P, Q = Q.transpose(1, 2), P.transpose(1, 2)
    menPreferences = torch.argsort(-P, axis = -1)
    womenPreferences = torch.argsort(-Q, axis = -2)
    return DA(P, Q, menPreferences, womenPreferences).transpose(1, 2)