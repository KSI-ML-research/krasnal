#!/usr/bin/env python3
"""Evaluate a trained model on Lichess puzzles.

Reads a JSONL file of filtered puzzles and evaluates the model's ability to
find the correct first move solution.

Usage:
    python scripts/evals/eval_puzzles.py --artifact-dir artifacts/pretrain/...
    python scripts/evals/eval_puzzles.py  # uses latest model
"""

import argparse
import json
from pathlib import Path

import torch

from krasnal.eval.puzzles import evaluate_model_on_puzzle_file
from krasnal.uci_engine.provider import ModelProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Krasnal model on Lichess puzzles.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help=(
            "Artifact directory containing model.pt and config.json. This argument is now required."
        ),
    )
    parser.add_argument(
        "--puzzles",
        type=Path,
        default=Path("data/puzzles_filtered.jsonl"),
        help="Path to filtered puzzles JSONL file. Defaults to data/puzzles_filtered.jsonl",
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=None,
        help="Number of puzzles to evaluate. Defaults to all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda/cpu). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--log-mrr",
        action="store_true",
        help="Include puzzle MRR in the output metrics.",
    )
    parser.add_argument(
        "--log-bucket-metrics",
        action="store_true",
        help="Include per-rating-bucket puzzle metrics in the output.",
    )
    parser.add_argument(
        "--log-diagnostics",
        action="store_true",
        help="Include diagnostic counts such as totals and skipped puzzles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load model
    artifact_dir = args.artifact_dir
    print(f"Loading model from {artifact_dir}...")
    provider = ModelProvider.from_artifact_dir(artifact_dir)

    if not args.puzzles.exists():
        raise FileNotFoundError(
            f"Puzzles file not found: {args.puzzles}\n"
            "Run: just download-puzzles && just prepare-puzzles"
        )

    device = torch.device(args.device) if args.device else torch.device(provider.device)
    result = evaluate_model_on_puzzle_file(
        model=provider.model,
        device=device,
        puzzle_path=args.puzzles,
        sample_size=args.num_puzzles,
    )
    metrics = result.to_metrics(
        log_mrr=args.log_mrr,
        log_bucket_metrics=args.log_bucket_metrics,
        log_diagnostics=args.log_diagnostics,
    )

    payload = {
        "artifact_dir": str(artifact_dir),
        "puzzles": str(args.puzzles),
        "num_puzzles": args.num_puzzles,
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
