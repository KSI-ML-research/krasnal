# Evaluation

This script evaluates how well the model plays chess. It measures legal move adherence and quality (accuracy) against Stockfish.

## Key Metrics

- **Top-1 Legal**: How often the most likely predicted move is actually legal.
- **Illegal Move Mass**: How much probability is "wasted" on non-legal moves.
- **ACPL (Average Centipawn Loss)**: We force the model to play its best **legal** choice, then compare its quality to Stockfish. Lower is better.

All metrics are broken down by game phase (opening, middle, endgame) and shown as a trend by move number.

## Architecture

Inference logic lives in `src/inference.py`:
- `load_model()` — loads checkpoint + tokenizer
- `InferenceSession` — stateful, move-by-move inference (feed tokens, get probability distributions)

Evaluation logic (`src/evaluate.py`) uses `InferenceSession` to get per-move probabilities and computes metrics on top.

## How to Run

```bash
uv run python src/evaluate.py
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-games` | 100 | Number of games to evaluate |
| `--seed` | 42 | Random seed |
| `--model-path` | `models/chess_model.pt` | Path to model checkpoint |
| `--stockfish-path` | `stockfish` | Path to Stockfish binary |
| `--stockfish-time` | 0.05 | Seconds per Stockfish evaluation |
| `--skip-acpl` | off | Skip Stockfish entirely |

## Output

Results are saved to `results/` with a timestamp:
- `eval_YYYYMMDD_HHMMSS.csv` — per-move metrics
- `eval_YYYYMMDD_HHMMSS.png` — trend plot (legal rate + ACPL by move number)
