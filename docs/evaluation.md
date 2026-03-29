# Evaluation Pipeline

## Overview

The evaluation pipeline converts game token sequences into metric scores. It has 4 stages:

```
Dataset → Parse → Replay → Infer → Aggregate → Metrics
```

---

## Stage 1: Parse

Input: Raw token IDs from dataset
Output: Structured game data (outcome, ELO, moves)

```python
# Input: [0, 3, 8, 9, 5001, 5023, ..., 1]
#        ^  ^  ^  ^  ^^^^^ ^^^^^^
#        |  |  |  |  |     └── black_elo_token
#        |  |  |  └──┼───── white_elo_token
#        |  |  └────┼───── outcome (white_won/black_won/draw)
#        |  └───────┼───── game_start
#        └──────────┴───── game_end
```

Extracts:
- Outcome (who won/draw)
- White/Black ELO buckets (6 levels each)
- List of move tokens

---

## Stage 2: Replay

Input: Structured game data
Output: List of `EvalContext` (one per position)

```python
for each move in the game:
    board = chess.Board()
    board.apply(move)

    context = EvalContext(
        sequence=[game_start, outcome, elo, ...],  # tokens so far
        legal_ids=[token_ids_of_legal_moves],    # what model can play
        piece_type=1,                            # pawn=1, knight=2, ...
        actual_token=move_token,                 # ground truth
        in_check=True/False,                      # king under attack?
        phase="opening"/"middlegame"/"endgame",
        gives_check=True/False,
        fen=board.fen(),                         # position string
    )
```

The board is replayed move-by-move to capture position metadata.

---

## Stage 3: Infer

Input: List of `EvalContext` (sequences without probs)
Output: Same list, but with `probs` attached

```python
# Batch inference - run model on all positions at once
all_sequences = [ctx.sequence for ctx in contexts]
probs = batch_inference(all_sequences)

for ctx, prob in zip(contexts, probs):
    ctx.probs = prob  # probability distribution over vocab
```

Now each context has the model's predictions.

---

## Stage 4: Aggregate

Input: `EvalContext` with probs
Output: Final metric scores

```python
results = {}
for ctx in contexts:
    for metric in metrics:
        value = metric.compute(ctx)
        results[metric_name].append(value)

# Average each metric
final = {name: sum(values)/len(values) for name, values in results.items()}
```

---

## Example Metrics

| Metric | What it measures |
|--------|-------------------|
| `top1_legal` | Does model pick a legal move as top prediction? |
| `acc` | Is model's top-1 the actual move played? |
| `illegal_mass` | Total probability on illegal moves |
| `mrr` | Mean Reciprocal Rank of actual move |
| `acpl` | Average centipawn loss (needs Stockfish) |
| `blunder_rate` | % of moves that are blunders |

---

## CoT Evaluation

Chain-of-thought evaluation follows the same 4 stages, but with `<think>` tokens:

```python
# Input: [game_start, outcome, move1, move2, <think>, thinking..., <think>, move_n, game_end]
#                     ──────────────prompt─────────── ──────generated──────
```

Special handling:
1. **Parse**: Extract prompt tokens, think tokens, and post-think move
2. **Generate**: Use model to generate continuation (sampling)
3. **Validate**: Check CoT format is valid (matching `<think>` tags)

---

## Files

| File | Responsibility |
|------|----------------|
| `evaluator.py` | Orchestrator - runs the 4 stages |
| `metrics/` | Individual metric implementations |
| `context.py` | `EvalContext` dataclass |

---

## Usage

```python
evaluator = ChessEvaluator(metrics=["top1_legal", "acc"])
results = evaluator.evaluate(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    num_games=1000,
    device=torch.device("cuda"),
)
# {'top1_legal': 0.85, 'acc': 0.72, ...}
```
