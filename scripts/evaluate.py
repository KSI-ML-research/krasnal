import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from utils import set_seed

from config import ARTIFACTS_DIR, EVAL_DATASET_PATH, MOVES_FILE, ChessGPTConfig
from dataset import ChessDataset, collate_fn
from model import GPT, GPTConfig
from tokenizer import PAD_ID, Tokenizer


def evaluate(model_path: Path, dataset_path: Path, batch_size: int, num_workers: int) -> float:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Eval dataset not found at {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = ChessDataset(dataset_path)
    if len(dataset) == 0:
        raise ValueError("Eval dataset is empty")

    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=(device == "cuda"),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, loss = model(x, y, ignore_index=PAD_ID)

            valid_tokens = (y != PAD_ID).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("Eval dataset has no valid tokens")

    return total_loss / total_tokens


def resolve_latest_model_path() -> Path:
    """Find the latest artifact folder by modification time and return its model.pt path."""
    pretrain_dir = ARTIFACTS_DIR / "pretrain"
    if not pretrain_dir.exists():
        raise FileNotFoundError(f"No artifacts found in {pretrain_dir}")

    subdirs = sorted(
        [d for d in pretrain_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime
    )
    if not subdirs:
        raise FileNotFoundError(f"No artifact folders found in {pretrain_dir}")

    latest = subdirs[-1]
    model_path = latest / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found in {latest}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on eval.parquet")
    parser.add_argument(
        "model",
        nargs="?",
        type=str,
        help="Path to model.pt or run folder",
    )
    parser.add_argument("--latest", action="store_true", help="Evaluate the latest local artifact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-path", type=Path, default=EVAL_DATASET_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.latest and args.model is not None:
        parser.error("Use either a model argument or --latest, not both.")
    if not args.latest and args.model is None:
        parser.error("the following arguments are required: model (or use --latest)")

    set_seed(args.seed)

    if args.model:
        model_path = Path(args.model)
        if model_path.is_dir():
            model_path = model_path / "model.pt"
    else:
        model_path = resolve_latest_model_path()

    eval_loss = evaluate(model_path, args.dataset_path, args.batch_size, args.num_workers)
    print(f"eval_loss={eval_loss:.6f}")


if __name__ == "__main__":
    main()
