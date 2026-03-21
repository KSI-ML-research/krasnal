# Training Runs

Quick reference for how run artifacts are stored in `outputs/runs`.

## Layout

```
outputs/
  runs/
    <stage>_<params>M_<run_hash>_<commit_hash>/
      model.pt
      config.json
      tokenizer.json
```

## Name Parts

- `stage`: training stage (`pretrain` today; `sft`/`rlvr` planned)
- `params`: model size in millions (example: `12M`)
- `run_hash`: hash of model config + training config + seed + dataset_mtime
- `commit_hash`: short git commit id

## Run Hash
- `run_hash = sha256(stage + model_config + train_config + seed + model_repr + dataset_mtime)[:8]`
- Same `run_hash` means identical training stage, model, setup, seed, and dataset version.
- Dataset version is determined by the file's modification time (mtime) of the dataset.
- Re-processing the dataset updates its mtime, producing a new hash and new run folder.


## What Is Saved

- `model.pt`: model weights
- `tokenizer.json`: tokenizer artifact for that run
- `config.json`: metadata (stage, hash, seed, configs, commit, metrics)

## Workflow

```bash
# data prep
just download-games
just preprocess

# train and evaluate
just pretrain
just evaluate <model-path-or-run-hash>
just evaluate --latest  # optional way to evaluate the most recent run

# planned: finetuning
just sft --parent <run-hash>
just sft --latest  # optional way to finetune from the most recent run
just rlvr --parent <run-hash>
just rlvr --latest  # optional way to rlvr from the most recent run

# all-in-one
just pipeline
```
