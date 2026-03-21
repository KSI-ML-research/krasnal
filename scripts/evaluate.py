import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from utils import set_seed  # noqa: E402

from config import EVAL_DATASET_PATH, MOVES_FILE, PAD_ID, ChessGPTConfig  # noqa: E402
from dataset import ChessDataset, collate_fn  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from run_manager import find_latest_run, find_run_by_hash  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402


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


def resolve_model_path(path_or_hash: Path | str) -> Path:
    """Resolve a model path from various input formats.

    Args:
        path_or_hash: Can be:
            - Path to a model.pt file (returned as-is)
            - Path to a run folder (returns folder/model.pt)
            - A run hash string (looks up by hash, returns folder/model.pt)

    Returns:
        Path to the model.pt file.

    Raises:
        FileNotFoundError: If a run folder or hash is given but no matching run exists.
    """
    candidate = path_or_hash if isinstance(path_or_hash, Path) else Path(path_or_hash)
    if candidate.exists():
        if candidate.is_dir():
            model_path = candidate / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"model.pt not found in run folder: {candidate}")
            return model_path
        return candidate

    run_folder = find_run_by_hash(str(path_or_hash), stage="pretrain")
    if run_folder is None:
        raise FileNotFoundError(f"No run found for hash: {path_or_hash}")
    return run_folder / "model.pt"


def resolve_latest_model_path() -> Path:
    """Resolve the checkpoint path for the latest pretrain run.

    Returns:
        Path to model.pt inside the latest pretrain run folder.

    Raises:
        FileNotFoundError: If no pretrain run folder exists.
    """
    run_folder = find_latest_run(stage="pretrain")
    if run_folder is None:
        raise FileNotFoundError("No pretrain run found in outputs/runs")
    return run_folder / "model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on eval.parquet")
    parser.add_argument(
        "model",
        nargs="?",
        type=str,
        help="Path to model.pt, run folder, or run hash",
    )
    parser.add_argument("--latest", action="store_true", help="Evaluate the latest pretrain run")
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

    model_path = resolve_latest_model_path() if args.latest else resolve_model_path(args.model)
    eval_loss = evaluate(model_path, args.dataset_path, args.batch_size, args.num_workers)
    print(f"eval_loss={eval_loss:.6f}")


if __name__ == "__main__":
    main()
