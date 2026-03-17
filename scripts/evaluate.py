import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EVAL_DATASET_PATH, MODEL_PATH, MOVES_FILE, PAD_ID, ChessGPTConfig  # noqa: E402
from dataset import ChessDataset, collate_fn  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on eval.parquet")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=EVAL_DATASET_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    eval_loss = evaluate(args.model_path, args.dataset_path, args.batch_size, args.num_workers)
    print(f"eval_loss={eval_loss:.6f}")


if __name__ == "__main__":
    main()
