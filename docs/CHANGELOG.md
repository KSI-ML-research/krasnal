# Changelog

## 2026-05-07

- Optional NCCL DDP for pretrain and SFT: use `torchrun` with `WORLD_SIZE>1`; single-process runs unchanged. `train.batch_size` is per GPU when distributed.

## 2026-05-06

*Squashed changes from 2026-04-28 - 2026-05-06*

- Removed unused `<capture>` token
- Added `<whats_on_XX>` `<empty>/<w:pawn>/<b:pawn>` auxiliary Q&A
- Renamed auxiliary piece-type Q&A question token `<what_piece>` to `<piece_type_moved>`.
- Added configurable probability to generating `<is_check>` tokens
- Simplified configs and wandb view
- Mask Q&A tokens in the inference
- Mask loss for Questions and Conditioning tokens in the training

## 2026-04-25

Added production inference benchmark (CPU, Ryzen 5 2600, 12M model, 100 games × 40 moves):

| Version | Avg time/move | Min    | Max     | Total time | Throughput |
|---------|---------------|--------|---------|------------|------------|
| Before  | 9.90ms        | 5.44ms | 154.36ms| 39.60s     | -          |
| After   | 7.06ms        | 5.97ms | 57.02ms | 28.23s     | 141.7/s    |

Fixed Stockfish-backed eval on terminal positions. When Stockfish returns `bestmove (none)` after a model move ends the game, eval now treats it as a valid terminal analysis instead of crashing pretrain/SFT runs.

## 2026-04-23

Added auxiliary piece-probing QA tokens in pretraining:

- Question token: `<piece_type_moved>` (introduced as `<what_piece>`)
- Answer tokens: `<pawn>`, `<knight>`, `<bishop>`, `<rook>`, `<queen>`, `<king>`

Piece QA is inserted after sampled moves and uses deterministic inverse-frequency sampling with `p_king=0.5` baseline. This targets a more balanced piece-answer distribution while keeping preprocessing reproducible.

Question-token targets are loss-masked (`-100`), so the model is trained to predict piece answers and continuation, not stochastic question placement.

Added piece probe evaluation metrics: `piece_acc`, `piece_macro_f1`, and 6x6 confusion-matrix counters.

## 2026-04-22

Replaced single `<check>` annotation with auxiliary QA tokens in pretraining:

- Question token: `<is_check>`
- Answer tokens: `<yes_check>`, `<no_check>`

All check plies are asked (`p_yes=1.0`).
Non-check plies are sampled with `p_no=N_yes/N_no` to target global ~50/50 answer balance.

Question-token targets are loss-masked (`-100`) so model is not trained to predict random question placement; it is trained to predict answers and chess continuation.

Token mix: UCI=73.13%, check QA=21.36%, outcome=3.31%. Details: `<is_check>=50.4M`, `<yes_check>=25.1M`, `<no_check>=25.3M`, result=5.2M, elo=10.4M, total=472M.

Speed: 4.20ms/token (RTX 5070 Ti).

Added unknown-ELO augmentation in preprocessing via `config/preprocess.yaml`, so the model knows how to act when the elo is unknown.

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
