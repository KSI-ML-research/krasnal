#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from krasnal.config import EVAL_DATASET_PATH
from krasnal.dataset import ChessDataset
from krasnal.eval import ChessEvaluator, get_stockfish_client
from krasnal.uci_engine.provider import ModelProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Krasnal artifact against Stockfish-backed dataset metrics."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Artifact directory containing model.pt and config.json.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=EVAL_DATASET_PATH,
        help="Parquet dataset path. Defaults to the repo eval dataset.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to sample from the dataset.",
    )
    parser.add_argument(
        "--stockfish-binary",
        default="stockfish",
        help="Path to the Stockfish binary.",
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=10,
        help="Fixed Stockfish depth for evaluation.",
    )
    parser.add_argument(
        "--stockfish-nodes",
        type=int,
        default=None,
        help="Fixed Stockfish node budget. Mutually exclusive with --stockfish-depth.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of positions to use for Stockfish-backed metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact_dir = args.artifact_dir
    provider = ModelProvider.from_artifact_dir(artifact_dir)
    dataset = ChessDataset(
        args.dataset, include_elo=provider.artifact_config.get("include_elo", True)
    )
    stockfish = get_stockfish_client(
        depth=args.stockfish_depth if args.stockfish_nodes is None else None,
        nodes=args.stockfish_nodes,
        binary=args.stockfish_binary,
    )
    evaluator = ChessEvaluator(
        metrics=["acpl", "blunder_rate", "stockfish_top1"],
        stockfish=stockfish,
        acpl_sample_size=args.sample_size,
    )

    try:
        results = evaluator.evaluate(
            model=provider.model,
            dataset=dataset,
            num_games=args.num_games,
            device=torch.device(provider.device),
        )
    finally:
        stockfish.close()

    payload = {
        "artifact_dir": str(artifact_dir),
        "dataset": str(args.dataset),
        "num_games": args.num_games,
        "stockfish_binary": args.stockfish_binary,
        "stockfish_depth": args.stockfish_depth if args.stockfish_nodes is None else None,
        "stockfish_nodes": args.stockfish_nodes,
        "sample_size": args.sample_size,
        "metrics": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
