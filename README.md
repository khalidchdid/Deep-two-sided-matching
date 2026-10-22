# Deep learning for two-sided matching

A one-to-one matching between two sides, learned from preference profiles. The network outputs a fractional matching. Training trades off two goals that cannot both be maximized: fairness between two groups on one side, and efficiency, measured as the chance of being matched to a top-choice partner.

The written report is in `report/deep-learning-for-two-sided-matching.pdf`. The implementation in `code/` follows the training setup of [Deep Learning for Two-Sided Matching](https://arxiv.org/abs/2107.03427) (Ravindranath, Feng, Li, Ma, Kominers, and Parkes). That paper's code is included, unchanged, under `reference/two-sided-matching` with its MIT license. This project changes the objective from stability and strategy-proofness to fairness and efficiency, and replaces the MLP and CNN with a row-and-column transformer followed by the same doubly-stochastic matching head.

## Model

Each side has `n` agents. A profile is two `n × n` ranking matrices. A transformer reads each row and each column, so an agent can attend to its own ranking and to the rankings in which it appears. An MLP then maps the result to a matching matrix: nonnegative entries, row and column sums at most one, and a zero wherever either side ranked the other as unacceptable.

Workers are split into two groups. For a top-`k` cutoff, fairness is the gap between the groups' average outcome, and efficiency is the average top-`k` rate on both sides. The loss is

```text
lambda * fairness gap - (1 - lambda) * efficiency
```

`lambda = 1` cares only about equalizing the groups. `lambda = 0` cares only about top-choice matches. Deferred acceptance, an efficiency-optimal assignment, and a fairness-optimal assignment are computed in `code/baselines.py` as comparisons.

## Run

From `code/`, with the packages in `requirements.txt`:

```bash
python train_model.py -n 8 -p 0.2 -c 0.0 -l 0.5 -k 1
```

`-n` is the number of agents on each side, `-p` the chance a ranking is truncated, `-c` the chance agents share a common ranking, `-l` the fairness weight, and `-k` the top-choice cutoff. Logs and checkpoints are written under `experiments/`.

```bash
python compute_baselines.py -n 8 -p 0.2 -c 0.0 -k 1
```

## What is not included

The papers and book read for this project are not in the repository. The report cites them. Training checkpoints are not included either.
