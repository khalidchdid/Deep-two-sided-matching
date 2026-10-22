import os
import time
import pickle
import random
import logging
import argparse
from typing import List

import numpy as np
import torch

import baselines
from data_gen import Data
from utils import init_logger


class Args: 
    num_agents: int = 10
    """ Number of agents """

    prob_trunc: float = 0.20
    """ Probability of truncation """

    corr: float = 0.25
    """ Correlation Probability: To be filled later """

    k : int = 3
    """ number of top firms or workers to consider for the output quality: To be filled later """

    wA, wB = np.ndarray, np.ndarray
    """ Arrays of groups A and B in workers """

    batch_size: int = 32
    """ Batch size """
    
    num_tst_samples: int = 20480
    """ Number of Test batches """
    
    num_tst_batches = num_tst_samples//batch_size

    seed: int = 42
    """ Random Seed """

def outcome_quality(R, P, Q, g, side='w'):
    if side == 'w':
        top_agents = P[:, g, :]
        R_top_agents = R[:, g, :]
        idx = torch.topk(top_agents, k=args.k, dim=-1, largest=True).indices
        return R_top_agents.gather(dim=-1, index=idx).sum(dim=-1)
    else:
        top_agents = Q[:, :, g]
        R_top_agents = R[:, :, g]
        idx = torch.topk(top_agents, k=args.k, dim=-1, largest=True).indices
        return R_top_agents.gather(dim=-1, index=idx).sum(dim=-1)

def group_outcome(R, P, Q, G, side='w'):
    vals = torch.stack([outcome_quality(R, P, Q, int(g), side) for g in G], dim=0)
    return vals.mean(dim=0)

def fairness_violation(R, P, Q):
    diff = group_outcome(R, P, Q, args.wA) - group_outcome(R, P, Q, args.wB)
    return diff.abs().mean()

def efficiency(R, P, Q):
    n_agents = R.size(1)
    workers = np.arange(n_agents)
    firms = np.arange(n_agents)
    w = group_outcome(R, P, Q, workers, side='w')
    f = group_outcome(R, P, Q, firms, side='f')
    return 0.5 * (w + f).mean()

def compute_violations(mech):
    np.random.seed(args.seed)
    random.seed(args.seed)
    FAI_LOSS = 0.0
    EFF_GAIN = 0.0
    
    for j in range(args.num_tst_batches):
        p, q = data.generate_batch(args.batch_size)
        P = torch.tensor(p, dtype=torch.float32)
        Q = torch.tensor(q, dtype=torch.float32)

        if mech == baselines.optimal_efficiency_for_both:
            R = mech(P, Q, args.k)          
        elif mech == baselines.optimal_efficiency_for_workers:
            R = mech(P, args.k)
        elif mech == baselines.optimal_efficiency_for_firms:
            R = mech(Q, args.k)
        elif mech == baselines.optimal_fairness:
            R = mech(P, args.k, args.wA)
        else:
            R = mech(P, Q)

        FAI_LOSS += fairness_violation(R, P, Q)
        EFF_GAIN += efficiency(R, P, Q)
    
    FAI_LOSS = FAI_LOSS / args.num_tst_batches
    EFF_GAIN = EFF_GAIN / args.num_tst_batches

    return FAI_LOSS, EFF_GAIN

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', '--num_agents', action='store',
                    dest='num_agents', required=True, type=int,
                    help='Num Agents')
    
    parser.add_argument('-p', '--prob_trunc', action='store',
                    dest='prob_trunc', required=True, type=float,
                    help='Truncation Probability')
    
    parser.add_argument('-c', '--corr', action='store',
                    dest='corr', required=True, type=float,
                    help='Correlation Probability')
    
    parser.add_argument('-k', '--topk', action='store',
                dest='topk', required=True, type=int,
                help='Top k firms')
    
    cmd_args = parser.parse_args()
        
    args = Args()
    args.num_agents = cmd_args.num_agents
    args.prob_trunc = cmd_args.prob_trunc
    args.corr = cmd_args.corr
    args.k = cmd_args.topk

    agents = np.random.permutation(args.num_agents)
    args.wA = agents[:args.num_agents // 4]
    args.wB = agents[args.num_agents // 4:]
    

    """ Loggers """
    root_dir =  os.path.join("experiments", "agents_%d"%(args.num_agents), "trunc_%.2f"%(args.prob_trunc),  "corr_%.2f"%(args.corr), "baselines") 
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    log_fname = os.path.join(root_dir, "LOG")
    logger = init_logger(log_fname)


    """ Seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = Data(args.num_agents, args.prob_trunc, args.corr)

    mech = baselines.compute_DA_batch
    f1, e1 = compute_violations(mech)
    mech = baselines.compute_DA_batch_switch
    f2, e2 = compute_violations(mech)
    DA_FAI, DA_EFF = max(f1, f2), min(e1, e2)
    logger.info("[Mechanism]: DA, [FAI-Loss]: %f, [EFF-Gain]: %f"%(DA_FAI, DA_EFF))


    mech = baselines.optimal_efficiency_for_both
    f1, e1 = compute_violations(mech)
    OPTEFF_B_FAI, OPTEFF_B_EFF = f1, e1
    logger.info("[Mechanism]: Optimal_efficiency_for_both, [FAI-Loss]: %f, [EFF-Gain]: %f"%(OPTEFF_B_FAI, OPTEFF_B_EFF))

    mech = baselines.optimal_efficiency_for_workers
    f2, e2 = compute_violations(mech)
    OPTEFF_W_FAI, OPTEFF_W_EFF = f2, e2
    logger.info("[Mechanism]: Optimal_efficiency_for_workers, [FAI-Loss]: %f, [EFF-Gain]: %f"%(OPTEFF_W_FAI, OPTEFF_W_EFF))

    mech = baselines.optimal_efficiency_for_firms
    f3, e3 = compute_violations(mech)
    OPTEFF_F_FAI, OPTEFF_F_EFF = f3, e3
    logger.info("[Mechanism]: Optimal_efficiency_for_firms, [FAI-Loss]: %f, [EFF-Gain]: %f"%(OPTEFF_F_FAI, OPTEFF_F_EFF))


    mech = baselines.optimal_fairness
    f1, e1 = compute_violations(mech)
    OPTFAI_FAI, OPTFAI_EFF = f1, e1
    logger.info("[Mechanism]: Optimal_fairness, [FAI-Loss]: %f, [EFF-Gain]: %f"%(OPTFAI_FAI, OPTFAI_EFF))


    data = dict()
    data['DA_FAI'] = DA_FAI
    data['DA_EFF'] = DA_EFF

    data['OPTEFF_B_FAI'] = OPTEFF_B_FAI
    data['OPTEFF_B_EFF'] = OPTEFF_B_EFF

    data['OPTEFF_W_FAI'] = OPTEFF_W_FAI
    data['OPTEFF_W_EFF'] = OPTEFF_W_EFF

    data['OPTEFF_F_FAI'] = OPTEFF_F_FAI
    data['OPTEFF_F_EFF'] = OPTEFF_F_EFF

    data['OPTFAI_FAI'] = OPTFAI_FAI
    data['OPTFAI_EFF'] = OPTFAI_EFF

    with open(os.path.join(root_dir, "data.p"), 'wb') as f:
        pickle.dump(data, f)