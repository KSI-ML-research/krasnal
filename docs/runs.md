# Training Runs

This document describes the `outputs/runs/` structure for organizing training runs.

## Directory Structure

```
outputs/
├── runs/
│   └── <stage>_<params>M_<YYYYMMDD>_<HHMMSS>_<run_hash>_<commit_hash>/
│       ├── model.pt
│       ├── config.json
│       └── tokenizer.json
└── ...
```

## Folder Naming Convention

| Part | Example | Description |
|------|---------|-------------|
| `stage` | `pretrain`, `sft`, `grpo_legal` | Training stage |
| `params` | `12M` | Millions of parameters |
| `YYYYMMDD` | `20260321` | Start date |
| `HHMMSS` | `163845` | Start time |
| `run_hash` | `a1b2c3d4` | Hash identifying this experiment |
| `commit_hash` | `g5h6i7j8` | Git commit (short) |

## Run Hash

The `run_hash` is computed from:
- Model configuration (layers, heads, embedding size, etc.)
- Training configuration (batch size, learning rate, etc.)
- Random seed

**Important:** The dataset is NOT included in the hash. This means:
- Same hash = same model architecture + same training setup
- But different datasets (e.g., pretrain vs SFT) will produce different final models
- If you change the dataset and re-run, you'll get the same hash but potentially different results

## Config JSON

Each run folder contains a `config.json` with full experiment metadata:

```json
{
  "stage": "pretrain",
  "run_hash": "a1b2c3d4",
  "params_M": 12,
  "timestamp": "2026-03-21T16:38:45",
  "commit_hash": "g5h6i7j8",
  "git_status": "clean",
  "parent": null,
  "model_config": {
    "n_layer": 12,
    "n_head": 8,
    "n_embd": 512,
    "block_size": 2048,
    "dropout": 0.1,
    "bias": false
  },
  "train_config": {
    "batch_size": 32,
    "learning_rate": 0.0005,
    "max_iters": 100000
  },
  "seed": 42,
  "final_loss": 2.34,
  "duration_seconds": 3600
}
```

For SFT/GRPO stages, additional fields are added:
```json
{
  "parent": "pretrain_12M_20260321_1638_a1b2c3d4_g5h6i7j8",
  "parent_hash": "a1b2c3d4",
  "in_model_path": "outputs/runs/pretrain_.../model.pt"
}
```

## Finding Runs

Use the `run_hash` to reference runs when continuing training:

```bash
# SFT from pretrain run
just sft --parent a1b2c3d4

# GRPO from SFT run
just rlvr_legal --parent a1b2c3d4
```

Or use `--latest` to automatically find the most recent run:

```bash
# SFT from latest pretrain run
just sft --latest

# GRPO from latest SFT run
just rlvr_legal --latest

# Run the full finetune pipeline (SFT then RLVR) using latest runs
just finetune
```

If a run hash already exists, you'll see a warning:
```
WARNING: Run with hash a1b2c3d4 already exists.
This means identical model_config + train_config + seed was used.
Proceeding anyway...
```

## Reproducibility

100% reproducibility is achieved by:
1. `commit_hash` → exact code version
2. `config.json` → all hyperparameters
3. `tokenizer.json` → exact tokenizer state
4. Same `run_hash` + same `seed` = same random initialization

## Sharing Runs

To share a run with your team:
1. Zip the run folder: `zip -r pretrain_12M_... run_folder/`
2. Send the zip
3. Recipient can unzip and evaluate or continue training

## Example Workflow

```bash
# 1. Train pretrain
just pretrain
# Output: Run folder created, hash printed (e.g., a1b2c3d4)

# 2. SFT from pretrain
just sft --parent a1b2c3d4
# Output: New run folder with new hash (e.g., b2c3d4e5), parent linked

# 3. GRPO from SFT
just rlvr_legal --parent b2c3d4e5
# Output: New run folder, chain complete

# 4. Evaluate any run
just eval a1b2c3d4
# Output: Eval results saved to outputs/results/
```

## Or use the finetune pipeline

```bash
# Run the full pipeline using latest runs automatically
just finetune

# Or with extra arguments
just finetune --epochs 2 --batch-size 64
```
