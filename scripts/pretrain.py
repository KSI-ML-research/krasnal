import argparse
import json
import math
from datetime import datetime

import torch
import wandb
from torch.utils.data import DataLoader
from utils import set_seed

from config import (
    ARTIFACTS_DIR,
    MOVES_FILE,
    PAD_ID,
    PRETRAIN_DATASET_PATH,
    ChessGPTConfig,
    TrainConfig,
)
from dataset import ChessDataset, collate_fn
from model import GPT, GPTConfig
from tokenizer import Tokenizer
from trainer import (
    cosine_warmup_lr,
    run_supervised_training,
    save_model_state,
    setup_runtime,
    unwrap_model,
)

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain on the raw dataset.")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="krasnal")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    if not PRETRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Pretraining dataset not found at {PRETRAIN_DATASET_PATH}. "
            "Run scripts/preprocess.py first to generate it."
        )

    train_dataset = ChessDataset(PRETRAIN_DATASET_PATH)
    dataset_mtime = int(PRETRAIN_DATASET_PATH.stat().st_mtime)
    tokenizer = Tokenizer(MOVES_FILE)
    vocab_size = tokenizer.get_vocab_size()

    mconf = ChessGPTConfig()
    model_config = GPTConfig(
        block_size=mconf.block_size,
        vocab_size=vocab_size,
        n_layer=mconf.n_layer,
        n_head=mconf.n_head,
        n_embd=mconf.n_embd,
        dropout=mconf.dropout,
        bias=mconf.bias,
    )
    model = GPT(model_config)

    tconf = TrainConfig()
    if args.epochs is not None:
        tconf.epochs = args.epochs
    if tconf.epochs <= 0:
        raise ValueError("TrainConfig.epochs must be > 0")

    device, device_type, dtype, ctx, scaler = setup_runtime()

    params_M = round(model.get_num_params() / 1_000_000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_DIR / "pretrain" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    wandb_config = {
        "stage": "pretrain",
        "params_M": params_M,
        "vocab_size": vocab_size,
        "block_size": mconf.block_size,
        "n_layer": mconf.n_layer,
        "n_head": mconf.n_head,
        "n_embd": mconf.n_embd,
        "epochs": tconf.epochs,
        "batch_size": tconf.batch_size,
        "learning_rate": tconf.learning_rate,
        "seed": args.seed,
        "dataset_mtime": dataset_mtime,
        "dataset_size": len(train_dataset),
        "model_repr": repr(model),
    }

    wandb.init(
        project=args.wandb_project,
        config=wandb_config,
    )
    run_id = wandb.run.id  # type: ignore[union-attr]
    entity = wandb.run.entity  # type: ignore[union-attr]
    project = wandb.run.project  # type: ignore[union-attr]
    wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

    print(
        f"{'=' * 60}\n"
        f"  Pretrain  |  {params_M:.2f}M params  |  {len(train_dataset):,} games\n"
        f"  layers={mconf.n_layer}, heads={mconf.n_head}, embd={mconf.n_embd}, "
        f"context={mconf.block_size}, vocab={vocab_size}\n"
        f"  Device: {device}  |  dtype: {dtype}  |  compile: {tconf.compile}\n"
        f"  Artifact dir: {artifact_dir.name}\n"
        f"{'=' * 60}"
    )

    optimizer = model.configure_optimizers(
        weight_decay=tconf.weight_decay,
        learning_rate=tconf.learning_rate,
        betas=(tconf.beta1, tconf.beta2),
        device_type=device_type,
    )

    model.to(device)

    if tconf.compile and device_type == "cuda":
        model = torch.compile(
            model,
            mode=tconf.compile_mode,
            dynamic=tconf.compile_dynamic,
            fullgraph=tconf.compile_fullgraph,
        )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=tconf.batch_size,
        num_workers=tconf.num_workers,
        collate_fn=collate_fn,
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Training dataset is empty. Cannot run training.")

    total_iters = max(1, math.ceil(tconf.epochs * steps_per_epoch))
    tconf.max_iters = total_iters

    def log_fn(_iter_num, last_loss_value, epoch_float):
        wandb.log({"train_loss": last_loss_value, "epoch": epoch_float})

    run_supervised_training(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        max_iters=total_iters,
        steps_per_epoch=steps_per_epoch,
        device=device,
        ctx=ctx,
        scaler=scaler,
        grad_clip=tconf.grad_clip,
        pad_id=PAD_ID,
        lr_fn=lambda i: cosine_warmup_lr(i, tconf),
        desc="train",
        log_interval=10,
        log_fn=log_fn,
    )

    print("Training finished.")
    model_path = artifact_dir / "model.pt"
    save_model_state(unwrap_model(model), model_path, tokenizer=tokenizer)
    print(f"Model saved to {model_path}")

    with open(artifact_dir / "config.json", "w") as f:
        json.dump(wandb_config, f, indent=2)

    with open(artifact_dir / "wandb_run_link.txt", "w") as f:
        f.write(f"{wandb_run_url}\n")

    artifact = wandb.Artifact("pretrain", type="model")
    artifact.add_dir(str(artifact_dir))
    wandb.log_artifact(artifact)
    print("Model logged to wandb as artifact: pretrain")

    wandb.finish()


if __name__ == "__main__":
    main()
