# Experiment Notes

## 2026-04-16

Switched Rust data pipeline to DuckDB + Aix. Enables processing more data (evals, fens, checks, captures). Added <check> token to pretrain data - model initially struggled (0.5% accuracy). Solution: doubled vocabulary size and differentiated white/black moves, improving <check> token prediction to 30% accuracy.

## 2026-04-07

We have two major ways to process the data:
- Concatenate all games into a single row and split by block_size
- One row per game, truncated to context window

The second approach is supperior in practice, but much slower in training.
Another problem is with torch.compile overhead for every single game length.
For this we use bucket padding to group games by length and pad to the same length.

## 2026-03-27

Tested Muon optimizer - 50% slower in training iterations compared to baseline.
Possible fix: increase batch size or adjust hyperparameters.

## 2026-03-17

Tested architectural variants:

- SwiGLU
- ReLU^2
- QK-norm

No significant improvement was observed over the baseline in the current training and evaluation setup.
