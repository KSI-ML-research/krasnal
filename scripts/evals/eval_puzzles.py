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

from krasnal.eval import ChessEvaluator
from krasnal.eval.metrics import DEFAULT_METRICS
from krasnal.uci_engine.provider import ModelProvider
from krasnal.utils import find_latest_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Krasnal model on Lichess puzzles.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help=(
            "Artifact directory containing model.pt and config.json. "
            "Defaults to latest runnable artifact."
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
    return parser.parse_args()


def load_puzzles(puzzle_path: Path, num_puzzles: int | None = None) -> list[dict]:
    """Load puzzles from JSONL file.

    Each line should be:
    {
        "fen": "...",
        "solution": "e2e4",
        "rating": 2000,
        "game_url": "..."
    }
    """
    puzzles = []
    with open(puzzle_path) as f:
        for i, line in enumerate(f):
            if num_puzzles and i >= num_puzzles:
                break
            puzzle = json.loads(line)
            puzzles.append(puzzle)

    if not puzzles:
        raise ValueError(f"No puzzles found in {puzzle_path}")

    print(f"Loaded {len(puzzles)} puzzles from {puzzle_path}")
    return puzzles


def main() -> None:
    args = parse_args()

    # Load model
    artifact_dir = args.artifact_dir or find_latest_model_artifact()
    print(f"Loading model from {artifact_dir}...")
    provider = ModelProvider.from_artifact_dir(artifact_dir)

    # Load puzzles
    if not args.puzzles.exists():
        raise FileNotFoundError(
            f"Puzzles file not found: {args.puzzles}\n"
            "Run: just download-puzzles && just prepare-puzzles"
        )
    puzzles = load_puzzles(args.puzzles, args.num_puzzles)

    # Create evaluator with default metrics
    print(f"Using {len(DEFAULT_METRICS)} metrics: {', '.join(DEFAULT_METRICS[:5])}...")
    _evaluator = ChessEvaluator(metrics=DEFAULT_METRICS)

    # Evaluate on puzzles
    # TODO: Implement puzzle-specific evaluation logic
    # For now, we'll use the standard evaluator on puzzle FENs
    print(f"Evaluating model on {len(puzzles)} puzzles...")

    _device = torch.device(args.device) if args.device else torch.device(provider.device)

    # Results placeholder
    results = {}
    try:
        for metric_name in DEFAULT_METRICS:
            results[metric_name] = 0.0  # Placeholder
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

    payload = {
        "artifact_dir": str(artifact_dir),
        "puzzles": str(args.puzzles),
        "num_puzzles": len(puzzles),
        "num_metrics": len(DEFAULT_METRICS),
        "metrics": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
