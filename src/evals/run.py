from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import torch

from src.config import EVAL_DATASET_PATH
from src.dataset import ChessDataset
from src.inference import load_model
from src.run_manager import find_run_by_hash
from src.tokenizer import Tokenizer

from .evaluator import ChessEvaluator
from .loss import evaluate_unseen_loss
from .reporting import print_results, save_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chess model on legal move metrics")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run hash to evaluate (required).",
    )
    parser.add_argument(
        "--eval-dataset-path",
        type=str,
        default=str(EVAL_DATASET_PATH),
        help="Path to unseen evaluation dataset parquet.",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="stockfish",
        help="Path to Stockfish binary (default: 'stockfish' from PATH).",
    )
    parser.add_argument(
        "--stockfish-time",
        type=float,
        default=0.05,
        help="Time limit (seconds) for Stockfish evaluations.",
    )
    parser.add_argument(
        "--loss-batch-size",
        type=int,
        default=256,
        help="Batch size for unseen-data loss evaluation.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="top1_legal,illegal_mass,acpl",
        help="Comma-separated list of metrics to compute (default: top1_legal,illegal_mass,acpl).",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Evaluate moves after a reasoning <think> block.",
    )
    parser.add_argument(
        "--cot-max-tokens",
        type=int,
        default=128,
        help="Max total tokens to generate while evaluating <think> blocks.",
    )
    parser.add_argument(
        "--allow-legacy-tokenizer-check",
        action="store_true",
        help=(
            "Allow evaluation without tokenizer metadata sidecars. "
            "Use only for old artifacts; this may produce invalid low-position evals."
        ),
    )
    return parser.parse_args()


def _dataset_meta_path(dataset_path: Path) -> Path:
    return Path(f"{dataset_path}.meta.json")


def validate_eval_tokenizer_compatibility(
    eval_dataset_path: Path,
    tokenizer: Tokenizer,
    *,
    allow_legacy: bool,
) -> None:
    meta_path = _dataset_meta_path(eval_dataset_path)
    if not meta_path.exists():
        if allow_legacy:
            return
        raise SystemExit(
            "Missing eval dataset metadata at "
            f"{meta_path}. Re-run preprocessing with the current code or pass "
            "--allow-legacy-tokenizer-check."
        )

    with meta_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    expected_hash = payload.get("tokenizer_hash")
    expected_vocab = payload.get("vocab_size")
    if not expected_hash or expected_vocab is None:
        if allow_legacy:
            return
        raise SystemExit(
            f"Invalid eval dataset metadata at {meta_path}. Missing tokenizer_hash/vocab_size."
        )

    actual_hash = tokenizer.mapping_hash()
    actual_vocab = tokenizer.get_vocab_size()
    if expected_hash != actual_hash or int(expected_vocab) != int(actual_vocab):
        raise SystemExit(
            "Tokenizer mismatch between eval dataset and loaded model tokenizer. "
            f"dataset_hash={expected_hash}, model_hash={actual_hash}, "
            f"dataset_vocab={expected_vocab}, model_vocab={actual_vocab}. "
            "Rebuild eval dataset/checkpoint with consistent tokenizer metadata."
        )


def setup_logging(timestamp: str, results_dir: Path) -> logging.Logger:
    """Configure logging to both console and file."""
    log_path = results_dir / f"eval_{timestamp}.log"

    logger = logging.getLogger("evals")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    return logger


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    args = parse_args()

    run_folder = find_run_by_hash(args.run)
    if not run_folder:
        raise SystemExit(f"Could not find run with hash {args.run}")
    print(f"Run: {run_folder.name}")

    model_path = run_folder / "model.pt"
    if not model_path.exists():
        raise SystemExit(f"Model not found in run: {model_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(timestamp, results_dir)

    logger.info("Loading model...")
    model, tokenizer = load_model(str(model_path), device)

    total_params, trainable_params = count_parameters(model)

    eval_dataset_path = Path(args.eval_dataset_path)
    if not eval_dataset_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {eval_dataset_path}. Run preprocessing to generate it."
        )

    validate_eval_tokenizer_compatibility(
        eval_dataset_path,
        tokenizer,
        allow_legacy=args.allow_legacy_tokenizer_check,
    )

    logger.info(f"Device: {device}")
    logger.info(f"Run: {run_folder.name}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
    logger.info("")
    logger.info(model)
    logger.info("")

    logger.info("Loading unseen evaluation dataset...")
    eval_dataset = ChessDataset(eval_dataset_path)
    logger.info(f"Eval dataset: {len(eval_dataset)} games")

    eval_loss, eval_ppl, eval_sequences, eval_tokens = evaluate_unseen_loss(
        model,
        eval_dataset,
        device,
        batch_size=args.loss_batch_size,
    )
    logger.info("")
    logger.info("=== Unseen Loss ===")
    logger.info(f"Sequences evaluated:      {eval_sequences:,}")
    logger.info(f"Predicted tokens:        {eval_tokens:,}")
    logger.info(f"Mean cross-entropy loss: {eval_loss:.6f}")
    logger.info(f"Perplexity:              {eval_ppl:.3f}")

    metrics_list = [m.strip() for m in args.metrics.split(",")]
    logger.info("")
    logger.info(f"Evaluating metrics on {args.num_games} sampled games: {metrics_list}")

    evaluator = ChessEvaluator(
        metrics=metrics_list,
        stockfish_path=args.stockfish_path,
        stockfish_time=args.stockfish_time,
        cot=args.cot,
        cot_max_tokens=args.cot_max_tokens,
    )

    stats = evaluator.evaluate(
        model,
        tokenizer,
        eval_dataset,
        args.num_games,
        device,
    )

    logger.info("")
    print_results(stats, logger)

    plot_path = results_dir / f"eval_{timestamp}.png"
    save_plot(stats, str(plot_path))
    logger.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
