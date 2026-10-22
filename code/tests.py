import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from ortools.linear_solver import pywraplp

P = torch.Tensor([[[1.0, 0.5, 0.75, 0.25],
                   [0.5, -0.25, 0.25, 0.75],
                   [0.5, -0.25, 0.25, 0.75],
                   [-0.75, -0.25, 0.25, -0.5]],
                  [[0.5, -0.25, 0.25, 0.75], 
                   [0.25, -0.25, 0.75, 0.5],
                   [1.0, 0.75, 0.25, 0.5],
                   [0.25, -0.25, 0.5, -0.5]]])

PP = torch.ones(2,3,2)

Q = torch.Tensor([[[1.0, 0.5, 0.75, 0.25],
                   [0.5, -0.25, 0.25, -0.25],
                   [0.75, -0.5, 0.5, 0.5],
                   [0.25, 0.25, 1.0, -0.5]],
                  [[0.5, -0.25, 0.25, 0.75], 
                   [0.25, 0.5, 0.75, 0.5],
                   [1.0, 0.75, 1.0, 0.25],
                   [0.75, 0.25, 0.5, -0.25]]])

QQ = torch.ones(2,2,3)

R = torch.Tensor([[[0.5, 0.25, 0.0], 
                   [0.0, 0.0, 0.6], 
                   [0.0, 0.1, 0.6]],
                  [[0.25, 0.75, 0.0], 
                   [0.0, 0.7, 0.0], 
                   [0.5, 0.1, 0.4]]])

mask_p = torch.nn.functional.pad((P > 0).to(P.dtype), (0, 0, 0, 1, 0, 0), mode='constant', value=1)
mask_q = torch.nn.functional.pad((Q > 0).to(Q.dtype), (0, 1, 0, 0, 0, 0), mode='constant', value=1)


import torch
import torch.nn as nn
from data_gen import Data
import numpy as np
data = Data(10, 0.2, 0.5)


from baselines import optimal_efficiency_for_both, top_k_firms, optimal_fairness
import networkx as nx



from baselines import compute_DA_batch

wA = [0, 2]
wB = [1, 3]
k = 2

def outcome_quality(R, P, Q, g, side='w'):
    if side == 'w':
        top_agents = P[:, g, :]
        R_top_agents = R[:, g, :]
        idx = torch.topk(top_agents, k=k, dim=-1, largest=True).indices
        return R_top_agents.gather(dim=-1, index=idx).sum(dim=-1)
    else:
        top_agents = Q[:, :, g]
        R_top_agents = R[:, :, g]
        idx = torch.topk(top_agents, k=k, dim=-1, largest=True).indices
        return R_top_agents.gather(dim=-1, index=idx).sum(dim=-1)

def group_outcome(R, P, Q, G, side='w'):
    vals = torch.stack([outcome_quality(R, P, Q, int(g), side) for g in G], dim=0)
    return vals.mean(dim=0)

def fairness_violation(R, P, Q):
    diff = group_outcome(R, P, Q, wA) - group_outcome(R, P, Q, wB)
    return diff.abs().mean()

def efficiency(R, P, Q):
    n_agents = R.size(1)
    workers = np.arange(n_agents)
    firms = np.arange(n_agents)
    w = group_outcome(R, P, Q, workers, side='w')
    f = group_outcome(R, P, Q, firms, side='f')
    return 0.5 * (w + f).mean()

da = compute_DA_batch(P, Q)
of = optimal_fairness(P, k, wA)
oe = optimal_efficiency_for_both(P, Q, k)

print(da)
print(efficiency(da, P, Q))
print(fairness_violation(da, P, Q))
print()
print(of)
print(efficiency(of, P, Q))
print(fairness_violation(of, P, Q))
print()
print(oe)
print(efficiency(oe, P, Q))
print(fairness_violation(oe, P, Q))