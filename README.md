# Deep learning for two-sided matching

This project learns a one-to-one matching from the two sides' preference rankings. The network outputs a fractional matching: each entry is the probability that a given pair is matched. Training balances two goals that pull in opposite directions. Fairness asks the two groups of workers to reach top choices at similar rates. Efficiency asks both sides to be matched to a highly ranked partner as often as possible.

The full write-up is `report/deep-learning-for-two-sided-matching.pdf`. Classical mechanisms guarantee one property at a time. Deferred acceptance is stable. It does not search for a compromise when stability, strategy-proofness, fairness, and efficiency cannot all be obtained together [3, 5, 9]. The code here treats that compromise as a training objective.

The training loop follows [Deep Learning for Two-Sided Matching](https://arxiv.org/abs/2107.03427) [6]. That paper learns a randomized matching whose loss trades off ex ante stability and ordinal strategy-proofness, with an MLP or a CNN. Its implementation is in `reference/two-sided-matching`, unchanged and under its MIT license. This project keeps the data generator, the individual-rationality mask, and the doubly stochastic output, and changes two things: the loss becomes fairness against efficiency, and the network is a row-and-column transformer [11].

## Matching

There are `n` workers and `n` firms. Each agent has a strict ranking of the other side and may rank some partners as unacceptable. A fractional matching is a nonnegative `n × n` matrix. Row and column sums are at most one, so each agent is matched at most once. An entry is forced to zero when either side finds the other unacceptable. A fractional matching of this form can be turned into a lottery over deterministic matchings by the Birkhoff–von Neumann theorem.

Outcome quality is ordinal. For a cutoff `k`, a worker's outcome is the probability of being matched to one of their `k` favorite firms, and likewise for a firm. Efficiency is the average of those two rates. Workers are split into two groups. Fairness violation is the absolute gap between the groups' average outcomes [9].

## Network

A preference profile is two ranking matrices. Each matrix entry is a score that falls by a constant step as the partner drops in the ranking, and is negative once the partner is unacceptable.

`MatchingTransformer` embeds the pair of scores at each position and adds a learned row embedding and column embedding. Each layer runs attention along the row, attention along the column, and a small network that mixes the two. Padding hides unacceptable partners. The result still has one vector per worker–firm pair.

`MLPHead` flattens that representation and produces two score tables, one for each side, including an outside option. Softplus and the acceptability mask keep unacceptable pairs at zero. Normalizing each side, then taking the entrywise minimum, yields the fractional matching used by the loss. That head is the same construction as in [6].

## Objective

```text
lambda * |fairness gap| - (1 - lambda) * efficiency
```

`lambda = 1` only equalizes the two groups. `lambda = 0` only raises the top-`k` rate. A small penalty on the norm of the matching matrix is subtracted as well. Rankings are sampled with two knobs: a probability that an agent's list is truncated, and a probability that agents share one common ranking.

Three mechanisms are computed on the same draws for comparison, in `code/baselines.py`:

- worker-proposing deferred acceptance
- the assignment that maximizes the workers' top-`k` rate
- the assignment that minimizes the absolute gap between the two groups

## Run

Install the packages in `requirements.txt`, then from `code/`:

```bash
python train_model.py -n 8 -p 0.2 -c 0.0 -l 0.5 -k 1
```

`-n` is the number of agents on each side, `-p` the chance a ranking is truncated, `-c` the chance agents share a common ranking, `-l` the fairness weight, and `-k` the top-choice cutoff. Logs and checkpoints go under `experiments/`.

```bash
python compute_baselines.py -n 8 -p 0.2 -c 0.0 -k 1
```

## References

[1] F. Zhang and L. Zhong. Three-sided matching problem with mixed preferences. *Journal of Combinatorial Optimization* 42, 928–936, 2021.

[2] W. Liu, H. Bao, M. Yamada, Z. Huang, N. Zheng, and H. Qian. Many-to-many matching via sparsity controlled optimal transport. [arXiv:2503.24204](https://arxiv.org/abs/2503.24204), 2025.

[3] A. E. Roth. The economics of matching: stability and incentives. *Mathematics of Operations Research* 7(4), 617–628, 1982.

[4] T. Sugano. Minimizing instability in a strategy-proof matching mechanism using a linear programming approach. [arXiv:2502.12431](https://arxiv.org/abs/2502.12431), 2025.

[5] P. Mandal. Compatibility between stability and strategy-proofness: a single-peaked preferences investigation. [arXiv:2304.11494](https://arxiv.org/abs/2304.11494), 2025.

[6] S. S. Ravindranath, Z. Feng, S. Li, J. Ma, S. D. Kominers, and D. C. Parkes. Deep learning for two-sided matching. [arXiv:2107.03427](https://arxiv.org/abs/2107.03427), 2023.

[7] A. E. Roth and M. A. O. Sotomayor. *Two-Sided Matching: A Study in Game-Theoretic Modeling and Analysis*. Cambridge University Press, 1990. [doi:10.1017/CCOL052139015X](https://doi.org/10.1017/CCOL052139015X).

[8] H. Aziz, P. Biró, and M. Yokoo. Matching market design with constraints. In *Proceedings of the 36th AAAI Conference on Artificial Intelligence*, 12308–12316, 2022. [doi:10.1609/aaai.v36i11.21495](https://doi.org/10.1609/aaai.v36i11.21495).

[9] S. Cho, K. Kimura, K. Liu, K. Liu, Z. Liu, Z. Sun, K. Yahiro, and M. Yokoo. Fairness and efficiency trade-off in two-sided matching. [arXiv:2402.01084](https://arxiv.org/abs/2402.01084), 2024.

[10] S. Kimata and Y. Sakurai. Automated many-to-one matching mechanism design using deep learning. In *Multi-disciplinary Trends in Artificial Intelligence*, LNCS 15431, Springer, 2025. [doi:10.1007/978-981-96-0692-4_21](https://doi.org/10.1007/978-981-96-0692-4_21).

[11] H. Xu, L. Xiang, H. Ye, D. Yao, P. Chu, and B. Li. Permutation equivariance of transformers and its applications. [arXiv:2304.07735](https://arxiv.org/abs/2304.07735), 2024.
