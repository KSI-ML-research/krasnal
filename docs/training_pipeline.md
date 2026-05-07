# Training Pipeline

This document describes the complete data pipeline: from downloading raw chess games to training a model.

## Overview

```
Download (Aix DB) → Filter → Preprocess → Pretrain
       ↓                    ↓         ↓
  HuggingFace          move_vocab   artifacts/
   Cache
```

---

## 1. Download Games

Download high-quality Lichess games with Stockfish evaluations from HuggingFace.

```bash
just download-games target_games=5000000
```

**Configuration** (in `config/download.yaml`):
- `target_games` - stop after reaching this many filtered games
- `min_elo` - minimum player ELO (default: 1500)
- `min_time` - minimum time control in seconds (default: 300 = 5+0)
- `compression` - dataset compression (default: high)

**Output:** Parquet files in `data/1_filtered/`

**Note:** Files are cached by HuggingFace at `~/.cache/huggingface/hub/` (not in the project directory). See [HuggingFace Cache](#huggingface-cache) for cleanup commands.

---

## 2. Preprocess

Tokenize filtered games into training format.

```bash
just preprocess target_games=5000000
```

**Configuration** (in `config/preprocess.yaml`):
- `target_games` - number of games to tokenize
- `preprocess_workers` - parallel workers (0 = auto)
- `side_prefixed_moves` - include mover side in move tokens (`w:e2e4`, `b:e7e5`)
- `piece_aware_moves` - include mover piece type in move tokens (`w:pawn:e2e4`)

**Output:**
- `data/2_tokenized/move_vocab.json` - generated vocabulary with manifest and token IDs
- `data/2_tokenized/pretrain.parquet` - tokenized training games
- `data/2_tokenized/eval.parquet` - tokenized eval games

Preprocessing always rebuilds `move_vocab.json` from all `data/1_filtered/` games before the train/eval split. Move IDs are assigned from sorted token strings, so the mapping is deterministic for a fixed filtered corpus and vocabulary config. The corpus must include valid `piece_moved` lists for every move.

---

## 3. Pretrain

Train the model on preprocessed data.

```bash
just pretrain model=large train=cuda
```

**Configuration** (in `config/pretrain.yaml`):
- `model` - model size (small, medium, large)
- `train` - training backend (cuda, mps, cpu)

**Multi-GPU** (`torchrun`): use one process per GPU, e.g. `torchrun --standalone --nproc_per_node=2 $(which uv) run scripts/training/pretrain.py ...`. Config `train.batch_size` is **per GPU**; effective batch size is `batch_size × world_size`. W&B, checkpoint writes, and eval games run on rank 0 only.

**Output:** Model checkpoints in `artifacts/pretrain/`

Pretraining reads `data/2_tokenized/move_vocab.json` directly. It fails before training if the file is missing or if `piece_aware_moves` / `side_prefixed_moves` do not match the manifest.

Training applies the supervised CE loss mask in one place (`src/krasnal/supervised_target_mask.py`): Q&A prompts, square-query prompts, outcome/Elo conditioning tokens, and related prompt-like positions are written as `ignore_index` on the shifted target so they are never trained as next-token labels.

---

## Full Pipeline

Run everything in sequence:

```bash
just pipeline
```

Or step by step:

```bash
just download-games
just preprocess
just pretrain
```

---

## HuggingFace Cache

Downloads are cached by HuggingFace at `~/.cache/huggingface/hub/`.

**Location:**
```bash
~/.cache/huggingface/
```

**Check size:**
```bash
du -sh ~/.cache/huggingface/
hf cache ls
```

**Clean Aix dataset cache:**
```bash
hf cache rm dataset/thomasd1/aix-lichess-database -y
```

**Clean all HuggingFace cache:**
```bash
rm -rf ~/.cache/huggingface/
```
