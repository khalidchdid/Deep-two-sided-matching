import os
import logging
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

import random
import torch
import torch.nn as nn
from typing import List

from data_gen import Data
from model import Mechanism

from utils import init_logger

import time

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

class Args: 
    """ Env Params ------------------------------ """

    num_agents: int = 0
    """ Number of agents: To be filled later """

    prob_trunc: float = 0.0
    """ Probability of truncation: To be filled later """

    corr: float = 0.00
    """ Correlation Probability: To be filled later """

    lambd: float = 0.00
    """ Tradeoff param: To be filled later """

    reg: float = 0.5
    """ Regularization param """

    """ Transformer Params ---------------------- """

    d_model : int = 64
    n_layer : int = 4
    n_head : int = 2
    dim_ffm : int = 256

    """ MLP Params ------------------------------ """

    net_arch: List[int] = [256, 256, 256, 256, 256]
    """ Neural Network Architecture """

    act_fn = nn.LeakyReLU

    """ Optimization Params ---------------------- """

    batch_size: int = 16
    """ Batch size """
    
    num_accums: int = 1
    """ Num accumulation """

    learning_rate:  float = 1e-3
    """ Learning Rate """

    max_epochs: int = 1000
    """ Max iterations """

    """ Data Params ------------------------------ """

    workers : np.ndarray
    """ Array of workers """

    wA, wB = np.ndarray, np.ndarray
    """ Arrays of groups A and B in workers """

    k : int = 0
    """ number of top firms or workers to consider for the output quality: To be filled later """

    """ Logging Params --------------------------- """

    print_iter: int = 100
    val_iter: int = 100
    """ Frequency of logging train stats, validation stats """

    save_iter: int = 100
    """ Frequency of saving models """

    num_val_samples: int = 2560
    """ Validation batch """

    num_tst_samples: int = 20480
    """ Number of Test batches """

    """ Miscellaneous Params --------------------- """ 

    seed: int = 0
    """ Random Seed """
    
    resume: bool = False
    """ Start new (0) or resume (1) """

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

def evaluate(model, data, batch_size, num_samples):
    model.eval()
    num_batches = num_samples // batch_size
    fairness_loss = 0.0
    efficiency_gain = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            p, q = data.generate_batch(batch_size)
            P = torch.tensor(p, dtype=torch.float32, device=device)
            Q = torch.tensor(q, dtype=torch.float32, device=device)
            R = model(P, Q)
            fairness_loss += fairness_violation(R, P, Q).item() / num_batches
            efficiency_gain += efficiency(R, P, Q).item() / num_batches
    return fairness_loss, efficiency_gain


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
    
    parser.add_argument('-l', '--lambd', action='store',
                    dest='lambd', required=True, type=float,
                    help='Lambda')
    
    parser.add_argument('-k', '--topk', action='store',
                    dest='topk', required=True, type=int,
                    help='Top k firms')
    
    parser.add_argument('-r', '--resume', action='store',
                    dest='resume', default=0, type=int,
                    help='Resume Training')

    cmd_args = parser.parse_args()
    
    args = Args()
    args.num_agents = cmd_args.num_agents
    args.prob_trunc = cmd_args.prob_trunc
    args.corr = cmd_args.corr
    args.lambd = cmd_args.lambd
    args.k = cmd_args.topk
    args.resume = (cmd_args.resume == 1)

    args.workers = np.random.permutation(args.num_agents)
    args.wA = args.workers[:args.num_agents // 4]
    args.wB = args.workers[args.num_agents // 4:]


    """ Loggers """
    root_dir =  os.path.join("experiments", "agents_%d"%(args.num_agents), "trunc_%.2f"%(args.prob_trunc),  "corr_%.2f"%(args.corr), "Transformer") 
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    
    log_fname = os.path.join(root_dir, "LOG_lambd_%.4f"%(args.lambd))
    if args.resume:
        logger = init_logger(log_fname, filemode = 'a')
    else:
        logger = init_logger(log_fname)

    model_path = os.path.join(root_dir, "CHECKPOINT_lambd_%.4f"%(args.lambd))

    """ Seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    data = Data(args.num_agents, args.prob_trunc, args.corr)
    model = Mechanism(
        args.d_model, args.n_layer, args.n_head, args.dim_ffm,
        args.net_arch, args.act_fn, args.num_agents
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.resume:
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        logger.info("*** Resuming Training ***")

    tic = time.time()

    for epoch in range(args.max_epochs):
        model.train()
        opt.zero_grad()
        total_loss = 0
        fai_loss = 0
        eff_gain = 0
        for _ in range(args.num_accums):
            p, q = data.generate_batch(args.batch_size)
            P = torch.tensor(p, dtype=torch.float32, device=device)
            Q = torch.tensor(q, dtype=torch.float32, device=device)

            R = model(P, Q)

            fai = fairness_violation(R, P, Q)
            eff = efficiency(R, P, Q)
            
            fai_loss += fai
            eff_gain += eff

            reg = torch.linalg.norm(R, dim=(-2, -1)).mean() / (args.num_agents ** 0.5)

            total_loss += (args.lambd * fai - (1 - args.lambd) * eff - args.reg * reg)
        
        fai_loss = fai_loss / args.num_accums
        eff_gain = eff_gain / args.num_accums
        total_loss = total_loss / args.num_accums
        
        total_loss.backward()
        opt.step()
        t_elapsed = time.time() - tic

        # Validation
        if epoch == 0 or epoch == args.max_epochs - 1:
            print(f'P = {P}')
            print(f'Q = {Q}')
            print(f'R = {R}')
            print(torch.sum(R[0][0]))

        if epoch % args.print_iter == 0 or epoch == args.max_epochs - 1:
            logger.info("[iter]: %d, [t]: %f, [fai_loss]: %f, [eff_gain]: %f, [R_norm]: %f"%(epoch, t_elapsed, fai_loss.item(), eff_gain.item(), reg))

        if epoch % args.save_iter == 0 or epoch == args.max_epochs - 1:
            checkpoint = dict()
            checkpoint['epoch'] = epoch
            checkpoint['model'] = model.state_dict()
            checkpoint['opt'] = opt.state_dict()
            torch.save(checkpoint, model_path)

        if epoch % args.val_iter == 0 or epoch == args.max_epochs - 1:
            num_samples = args.num_tst_samples if epoch == (args.max_epochs - 1) else args.num_val_samples
            fai_loss, eff_gain = evaluate(model, data, args.batch_size, num_samples)
            logger.info("\t[TEST]: %d, [fai_loss]: %f, [eff_gain]: %f"%(epoch, fai_loss, eff_gain))