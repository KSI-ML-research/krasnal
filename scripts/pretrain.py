import argparse
import time

import torch
from torch.utils.data import DataLoader
from utils import set_seed

from config import (
    MOVES_FILE,
    PAD_ID,
    PRETRAIN_DATASET_PATH,
    ChessGPTConfig,
    TrainConfig,
)
from dataset import ChessDataset, collate_fn
from model import GPT, GPTConfig
from run_manager import compute_params_M, create_run_folder, save_run_config, update_run_config
from tokenizer import Tokenizer, save_tokenizer_for_artifact
from trainer import cosine_warmup_lr, run_supervised_training, setup_runtime, unwrap_model

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain on the raw dataset.")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

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
    device, device_type, dtype, ctx, scaler = setup_runtime()

    params_M = compute_params_M(model)

    run_folder, run_hash, commit_hash = create_run_folder(
        stage="pretrain",
        params_M=params_M,
        model_config=mconf,
        train_config=tconf,
        seed=args.seed,
        model_repr=repr(model),
        dataset_mtime=dataset_mtime,
    )
    compile_str = "on" if tconf.compile else "off"
    print(
        f"{'=' * 60}\n"
        f"  Pretrain  |  {params_M:.2f}M params  |  {len(train_dataset):,} games\n"
        f"  layers={mconf.n_layer}, heads={mconf.n_head}, embd={mconf.n_embd}, "
        f"context={mconf.block_size}, vocab={vocab_size}\n"
        f"  Device: {device}  |  dtype: {dtype}  |  compile: {compile_str}\n"
        f"  run: {run_hash}  |  commit: {commit_hash[:7]}\n"
        f"{'=' * 60}"
    )

    save_run_config(
        folder=run_folder,
        stage="pretrain",
        run_hash=run_hash,
        params_M=params_M,
        model_config=mconf,
        train_config=tconf,
        seed=args.seed,
        commit_hash=commit_hash,
        extra={
            "dataset_path": str(PRETRAIN_DATASET_PATH),
            "dataset_size": len(train_dataset),
        },
    )

    model.to(device)

    if tconf.compile and device_type == "cuda":
        model = torch.compile(
            model,
            mode=tconf.compile_mode,
            dynamic=tconf.compile_dynamic,
            fullgraph=tconf.compile_fullgraph,
        )

    optimizer = model.configure_optimizers(
        weight_decay=tconf.weight_decay,
        learning_rate=tconf.learning_rate,
        betas=(tconf.beta1, tconf.beta2),
        device_type=device_type,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=tconf.batch_size,
        num_workers=tconf.num_workers,
        collate_fn=collate_fn,
    )

    print("Starting training...")

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Empty dataset: train_loader has no batches")
    tconf.max_iters = int(steps_per_epoch * tconf.epochs)
    stage_start = time.perf_counter()
    final_loss = run_supervised_training(
        model,
        optimizer,
        train_loader,
        max_iters=tconf.max_iters,
        steps_per_epoch=steps_per_epoch,
        device=device,
        ctx=ctx,
        scaler=scaler,
        grad_clip=tconf.grad_clip,
        pad_id=PAD_ID,
        lr_fn=lambda i: cosine_warmup_lr(i, tconf),
        desc="train",
    )
    duration_seconds = time.perf_counter() - stage_start
    print("Training finished.")

    model_path = run_folder / "model.pt"
    # If the model is compiled, we want to save the original uncompiled weights
    raw_model = unwrap_model(model)
    torch.save(raw_model.state_dict(), model_path)
    save_tokenizer_for_artifact(tokenizer, model_path)
    print(f"Model saved to {model_path}")

    update_run_config(
        run_folder,
        {"final_loss": float(final_loss), "duration_seconds": duration_seconds},
    )


if __name__ == "__main__":
    main()
