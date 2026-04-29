#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch

from krasnal.eval.puzzles import evaluate_model_on_puzzle_file
from krasnal.uci_engine.provider import ModelProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Krasnal artifact on puzzle datasets.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Artifact directory containing model.pt and config.json.",
    )
    parser.add_argument(
        "--puzzles",
        type=Path,
        default=Path("data/puzzles_filtered.jsonl"),
        help="Path to puzzle JSONL file. Defaults to data/puzzles_filtered.jsonl.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional random sample size from the puzzle file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for puzzle sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional runtime device override (e.g. cpu, cuda).",
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

    if not args.puzzles.exists():
        raise FileNotFoundError(f"Puzzle file not found: {args.puzzles}")

    device = torch.device(args.device) if args.device else None
    provider = ModelProvider.from_artifact_dir(args.artifact_dir, device=device)
    metrics = evaluate_model_on_puzzle_file(
        model=provider.model,
        device=torch.device(provider.device),
        puzzle_path=args.puzzles,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    metrics = metrics.to_metrics(
        log_mrr=args.log_mrr,
        log_bucket_metrics=args.log_bucket_metrics,
        log_diagnostics=args.log_diagnostics,
    )

    payload = {
        "artifact_dir": str(args.artifact_dir),
        "puzzles": str(args.puzzles),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
