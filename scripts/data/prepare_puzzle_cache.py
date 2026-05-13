#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from krasnal.eval.puzzles import build_source_game_cache, load_puzzles_jsonl, sample_puzzles
from krasnal.puzzle_cache import source_game_cache_path_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline Lichess puzzle source-game cache.")
    parser.add_argument(
        "--puzzles",
        type=Path,
        default=Path("data/puzzles_filtered.jsonl"),
        help="Path to puzzle JSONL file. Defaults to data/puzzles_filtered.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the source-game cache.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=64,
        help="Optional random sample size from the puzzle file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for puzzle sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.puzzles.exists():
        raise FileNotFoundError(f"Puzzle file not found: {args.puzzles}")

    output_path = args.output or source_game_cache_path_for(
        args.puzzles,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    puzzles = sample_puzzles(load_puzzles_jsonl(args.puzzles), args.sample_size, seed=args.seed)
    source_games = build_source_game_cache(puzzles, output_path)

    payload = {
        "puzzles": str(args.puzzles),
        "output": str(output_path),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "source_games": source_games,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
