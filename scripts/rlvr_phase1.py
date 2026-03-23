import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from utils import set_seed

import wandb
from config import ARTIFACTS_DIR, MOVES_FILE, PRETRAIN_DATASET_PATH, ChessGPTConfig
from model import GPT, GPTConfig
from rl import (
    RLPhase1Config,
    RLPhase1DataSource,
    resolve_pretrained_checkpoint,
    run_phase1_training,
)
from tokenizer import Tokenizer

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Run RLVR phase 1 training.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to pretrained model.pt or run directory",
    )
    parser.add_argument(
        "--latest-pretrain",
        action="store_true",
        help="Resolve the latest local pretrain artifact as the starting checkpoint",
    )
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--indefinitely", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="krasnal")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sft-batch-size", type=int, default=32)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--sft-mix-ratio", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-minutes", type=float, default=30.0)
    return parser.parse_args()


def _build_model(tokenizer: Tokenizer) -> GPT:
    mconf = ChessGPTConfig()
    model_config = GPTConfig(
        block_size=mconf.block_size,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=mconf.n_layer,
        n_head=mconf.n_head,
        n_embd=mconf.n_embd,
        dropout=mconf.dropout,
        bias=mconf.bias,
    )
    return GPT(model_config)


def _load_state_dict(model_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def main():
    args = parse_args()

    if (args.max_iters is None) == (not args.indefinitely):
        raise ValueError("Use exactly one of --max-iters or --indefinitely")
    if args.max_iters is not None and args.max_iters <= 0:
        raise ValueError("--max-iters must be > 0")
    if not PRETRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Pretraining dataset not found at {PRETRAIN_DATASET_PATH}")

    set_seed(args.seed)

    checkpoint_path = resolve_pretrained_checkpoint(args.model, args.latest_pretrain)
    tokenizer = Tokenizer(MOVES_FILE)
    policy_model = _build_model(tokenizer)
    reference_model = _build_model(tokenizer)
    state_dict = _load_state_dict(checkpoint_path)
    policy_model.load_state_dict(state_dict)
    reference_model.load_state_dict(state_dict)

    config = RLPhase1Config(
        batch_size=args.batch_size,
        sft_batch_size=args.sft_batch_size,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        sft_mix_ratio=args.sft_mix_ratio,
        log_every=args.log_every,
        save_minutes=args.save_minutes,
    )
    config.validate()

    data_source = RLPhase1DataSource(
        PRETRAIN_DATASET_PATH,
        max_prompt_tokens=config.max_prompt_tokens,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_DIR / "rlvr_phase1" / timestamp

    run_config = {
        "stage": "rlvr_phase1",
        "seed": args.seed,
        "checkpoint_path": str(checkpoint_path),
        "max_iters": args.max_iters,
        "indefinitely": args.indefinitely,
        "batch_size": config.batch_size,
        "sft_batch_size": config.sft_batch_size,
        "group_size": config.group_size,
        "think_min_tokens": config.think_min_tokens,
        "think_max_tokens": config.think_max_tokens,
        "learning_rate": config.learning_rate,
        "kl_coef": config.kl_coef,
        "sft_mix_ratio": config.sft_mix_ratio,
        "sft_coef": config.sft_coef,
        "save_minutes": config.save_minutes,
        "log_every": config.log_every,
        "dataset_path": str(PRETRAIN_DATASET_PATH),
        "dataset_size": len(data_source),
    }

    wandb.init(project=args.wandb_project, config=run_config)
    run_id = wandb.run.id  # type: ignore[union-attr]
    entity = wandb.run.entity  # type: ignore[union-attr]
    project = wandb.run.project  # type: ignore[union-attr]
    wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "wandb_run_link.txt", "w") as f:
        f.write(f"{wandb_run_url}\n")

    try:
        result = run_phase1_training(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            data_source=data_source,
            config=config,
            artifact_dir=artifact_dir,
            run_config=run_config,
            wandb_module=wandb,
            max_iters=args.max_iters,
            indefinitely=args.indefinitely,
            checkpoint_source=str(checkpoint_path),
        )
        with open(artifact_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
