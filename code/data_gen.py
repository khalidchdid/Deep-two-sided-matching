# Data Generation

import numpy as np

def permutation(N, n):
    P = np.zeros((N, n))
    for i in range(N):
        P[i] = np.random.permutation(n)
    return P

class Data(object):
    def __init__(self, num_agents, prob_trunc, corr):
        self.num_agents = num_agents
        self.corr = corr
        self.trunc = prob_trunc

    def sample_ranking(self, N):
        N_trunc = int(N * self.trunc)
        P = permutation(N, self.num_agents) + 1
        if N_trunc > 0:
            idx = np.random.choice(N, N_trunc, replace=False)
            trunc = np.random.randint(0, self.num_agents, size=N_trunc)
            swap = P[idx, trunc]
            P[idx, trunc] = 0
            P[idx] = P[idx] - swap[:, np.newaxis]
        return P/self.num_agents
    
    def generate_batch(self, batch_size):
        N = batch_size * self.num_agents
        P = self.sample_ranking(N).reshape(-1, self.num_agents, self.num_agents)
        Q = self.sample_ranking(N).reshape(-1, self.num_agents, self.num_agents)
        if self.corr > 0:
            P_common = self.sample_ranking(batch_size)[:, None, :]
            Q_common = self.sample_ranking(batch_size)[:, None, :]

            P_idx = np.random.binomial(1, self.corr, [batch_size, self.num_agents, 1])
            Q_idx = np.random.binomial(1, self.corr, [batch_size, self.num_agents, 1])

            P = P * (1 - P_idx) + P_common * P_idx
            Q = Q * (1 - Q_idx) + Q_common * Q_idx
        Q = Q.transpose(0, 2, 1)
        return P, Q

# Définir les groupes ici, et en désavantager un
