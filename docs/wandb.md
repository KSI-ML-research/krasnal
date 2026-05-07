# W&B Integration

## What's logged

- **Config**: model architecture, training hyperparameters, `model_repr`
- **Metrics**: `train_loss` per step
- **Artifact**: saved locally and uploaded to wandb

## Artifact structure

```
artifacts/pretrain/YYYYMMDD_HHMMSS/
  model.pt
  move_vocab.json
  config.json
  wandb_run_link.txt
```

Each pretrain run creates a new timestamped folder. `wandb_run_link.txt` contains the wandb run URL.

## Evaluating

```bash
# use latest local artifact
just evaluate --latest

# use local file by path
just evaluate artifacts/pretrain/YYYYMMDD_HHMMSS/model.pt
```
