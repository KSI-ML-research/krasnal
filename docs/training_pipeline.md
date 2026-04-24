# Training Pipeline

This document describes the complete data pipeline: from downloading raw chess games to training a model.

## Overview

```
Download (Aix DB) → Filter → Preprocess → Pretrain
       ↓                              ↓
  HuggingFace                 artifacts/
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

**Output:** Binary training files in `data/2_pretrain/`

---

## 3. Pretrain

Train the model on preprocessed data.

```bash
just pretrain model=large train=cuda
```

**Configuration** (in `config/pretrain.yaml`):
- `model` - model size (small, medium, large)
- `train` - training backend (cuda, mps, cpu)

**Output:** Model checkpoints in `artifacts/pretrain/`

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
