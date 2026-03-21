import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import MOVES_FILE, SFT_DATA_PATH, ChessGPTConfig, TrainConfig
from src.dataset import ChessDataset, collate_fn
from src.model import GPT, GPTConfig
from src.run_manager import (
    compute_params_M,
    create_run_folder,
    find_latest_run,
    find_run_by_hash,
    save_run_config,
    update_run_config,
)
from src.tokenizer import PAD_ID, Tokenizer, save_tokenizer_for_artifact
from src.trainer import cosine_warmup_lr, run_supervised_training, setup_runtime


def parse_args():
    parser = argparse.ArgumentParser(
        description=("SFT warmup with <think> blocks from prebuilt dataset")
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parent",
        type=str,
        default=None,
        help="Parent run hash or folder name (optional if --latest is used).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest pretrain run as parent.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(SFT_DATA_PATH),
        help="Path to the prebuilt SFT mix dataset (required).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.latest:
        parent_run = find_latest_run("pretrain")
        if not parent_run:
            raise SystemExit("No pretrain run found. Run 'just pretrain' first.")
        print(f"Using latest pretrain run: {parent_run.name}")
    elif args.parent:
        parent_run = find_run_by_hash(args.parent, stage="pretrain")
        if not parent_run:
            parent_run = Path(args.parent)
        print(f"Parent run: {parent_run.name}")
    else:
        raise SystemExit("--parent or --latest is required for SFT.")

    parent_config_file = parent_run / "config.json"
    if not parent_config_file.exists():
        raise SystemExit(f"Parent run config not found: {parent_config_file}")

    import json

    with parent_config_file.open() as f:
        parent_config = json.load(f)
    parent_hash = parent_config.get("run_hash")

    print("Loading dataset...")
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise SystemExit(
            f"SFT mix dataset not found at {dataset_path}. Run scripts/preprocess.py first."
        )
    dataset = ChessDataset(dataset_path)
    print(f"Using SFT dataset: {dataset_path}")
    print(f"Dataset size: {len(dataset)} games")

    mconf = ChessGPTConfig()
    tokenizer = Tokenizer(MOVES_FILE)
    vocab_size = tokenizer.get_vocab_size()
    model_config = GPTConfig(
        block_size=mconf.block_size,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=mconf.n_layer,
        n_head=mconf.n_head,
        n_embd=mconf.n_embd,
        dropout=mconf.dropout,
        bias=mconf.bias,
    )
    model = GPT(model_config)

    parent_model_path = parent_run / "model.pt"
    if parent_model_path.exists():
        state = torch.load(parent_model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded model from {parent_model_path}")
    else:
        print(f"WARNING: Parent model not found: {parent_model_path}. Training from scratch.")

    tconf = TrainConfig()
    tconf.batch_size = args.batch_size
    tconf.num_workers = 0

    params_M = compute_params_M(model)

    run_folder, run_hash, commit_hash = create_run_folder(
        stage="sft",
        params_M=params_M,
        model_config=mconf,
        train_config=tconf,
        seed=args.seed,
        parent=parent_run.name,
    )
    print(f"Run folder: {run_folder.name}")
    print(f"Run hash: {run_hash}")
    print(f"Git commit: {commit_hash}")
    print(
        f"layers={mconf.n_layer}, heads={mconf.n_head}, embd={mconf.n_embd}, "
        f"context={mconf.block_size}, vocab={vocab_size}"
    )
    compile_str = (
        f"mode={tconf.compile_mode}, dynamic={tconf.compile_dynamic}, "
        f"fullgraph={tconf.compile_fullgraph}"
    )
    print(f"torch.compile: {compile_str if tconf.compile else 'False'}")

    save_run_config(
        folder=run_folder,
        stage="sft",
        run_hash=run_hash,
        params_M=params_M,
        model_config=mconf,
        train_config=tconf,
        seed=args.seed,
        commit_hash=commit_hash,
        parent=parent_run.name,
        parent_hash=parent_hash,
        extra={
            "dataset_path": str(dataset_path),
            "dataset_size": len(dataset),
            "in_model_path": str(parent_model_path),
        },
    )

    device, device_type, dtype, ctx, scaler = setup_runtime()
    print(f"Using device: {device}")
    print(f"Mixed precision: {dtype}")

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

    steps_per_epoch = max(len(dataset) // max(tconf.batch_size, 1), 1)
    tconf.max_iters = steps_per_epoch * max(args.epochs, 1)

    train_loader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=tconf.batch_size,
        num_workers=tconf.num_workers,
        collate_fn=collate_fn,
    )

    print("Starting SFT CoT warmup...")

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
        desc="sft-cot",
    )
    duration_seconds = time.perf_counter() - stage_start
    print("SFT CoT warmup finished.")

    model_path = run_folder / "model.pt"
    torch.save(model.state_dict(), model_path)
    save_tokenizer_for_artifact(tokenizer, model_path)
    print(f"Model saved to {model_path}")

    update_run_config(
        run_folder,
        {"final_loss": float(final_loss), "duration_seconds": duration_seconds},
    )


if __name__ == "__main__":
    main()
