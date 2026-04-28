#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bulletchess
import polars as pl
import torch

from krasnal.config import RAW_UCI_DIR
from krasnal.inference import StatelessBatchInferenceSession
from krasnal.tokens import (
    GAME_START_ID,
    IS_CHECK_ID,
    NO_CHECK_ID,
    PAD_ID,
    YES_CHECK_ID,
    get_elo_bucket,
    legal_token_ids,
    move_token_id_for_ply,
    result_to_token_id,
    set_side_prefixed_moves,
)
from krasnal.uci_engine.provider import ModelProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure raw next-token bias toward <yes_check> after checking moves."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Artifact directory containing model.pt and config.json.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=RAW_UCI_DIR,
        help="Directory containing raw filtered parquet shards.",
    )
    parser.add_argument(
        "--sample-games",
        type=int,
        default=200,
        help="Approximate number of raw games to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for batched inference.",
    )
    parser.add_argument(
        "--side-prefixed-moves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the move-tokenization mode used during preprocessing.",
    )
    return parser.parse_args()


def _load_sample(dataset_dir: Path, sample_games: int, seed: int) -> pl.DataFrame:
    lf = pl.scan_parquet(str(dataset_dir / "*.parquet")).select(
        "uci_moves",
        "is_check",
        "result",
        "white_rating",
        "black_rating",
    )
    total_games = lf.select(pl.len()).collect().item()
    if total_games == 0:
        raise ValueError(f"No raw games found in {dataset_dir}")

    if sample_games >= total_games:
        return lf.collect()

    bucket_cutoff = max(1, round(sample_games / total_games * 1000))
    sampled = (
        lf.with_columns((pl.col("uci_moves").hash(seed=seed) % 1000).alias("sample_bucket"))
        .filter(pl.col("sample_bucket") < bucket_cutoff)
        .head(sample_games)
        .collect()
    )
    if sampled.is_empty():
        raise ValueError("Sampling produced no games; try a larger sample_games value")
    return sampled


def _build_context_tokens(
    *,
    uci_moves: str,
    result: str,
    white_rating: int,
    black_rating: int,
    ply: int,
) -> list[int]:
    tokens = [
        GAME_START_ID,
        result_to_token_id(result),
        get_elo_bucket(int(white_rating)),
        get_elo_bucket(int(black_rating)),
    ]
    moves = uci_moves.split()
    for move_ply in range(ply + 1):
        move_id = move_token_id_for_ply(moves[move_ply], move_ply)
        tokens.append(move_id if move_id is not None else PAD_ID)
    return tokens


def _legal_metrics(probs: torch.Tensor, legal_ids: list[int]) -> tuple[float, float]:
    top1 = int(torch.argmax(probs).item())
    top1_legal = 1.0 if top1 in legal_ids else 0.0
    vocab_size = probs.shape[0]
    illegal_mask = torch.ones(vocab_size, dtype=torch.bool, device=probs.device)
    illegal_mask[legal_ids] = False
    illegal_mass = float(probs[illegal_mask].sum().item())
    return top1_legal, illegal_mass


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def main() -> None:
    args = parse_args()

    set_side_prefixed_moves(args.side_prefixed_moves)
    provider = ModelProvider.from_artifact_dir(args.artifact_dir)
    sample = _load_sample(args.dataset_dir, args.sample_games, args.seed)

    sequences: list[list[int]] = []
    legal_sets: list[list[int]] = []
    for row in sample.iter_rows(named=True):
        uci_moves = str(row["uci_moves"])
        is_check = list(row["is_check"])
        moves = uci_moves.split()
        board = bulletchess.Board()
        for ply, gives_check in enumerate(is_check):
            move = bulletchess.Move.from_uci(moves[ply])
            board.apply(move)
            if gives_check:
                sequences.append(
                    _build_context_tokens(
                        uci_moves=uci_moves,
                        result=str(row["result"]),
                        white_rating=int(row["white_rating"]),
                        black_rating=int(row["black_rating"]),
                        ply=ply,
                    )
                )
                legal_sets.append(legal_token_ids(board))

    if not sequences:
        raise ValueError("No checking positions found in the sampled games")

    session = StatelessBatchInferenceSession(provider.model, torch.device(provider.device))
    probs = session.get_raw_probs_batch(sequences, batch_size=args.batch_size)

    yes_probs = probs[:, YES_CHECK_ID].tolist()
    is_probs = probs[:, IS_CHECK_ID].tolist()
    no_probs = probs[:, NO_CHECK_ID].tolist()
    top1_ids = probs.argmax(dim=-1).tolist()
    top1_legal_values: list[float] = []
    illegal_masses: list[float] = []
    for prob, legal_ids in zip(probs, legal_sets, strict=True):
        top1_legal_value, illegal_mass = _legal_metrics(prob, legal_ids)
        top1_legal_values.append(top1_legal_value)
        illegal_masses.append(illegal_mass)

    summary = {
        "artifact_dir": str(args.artifact_dir),
        "dataset_dir": str(args.dataset_dir),
        "sample_games": len(sample),
        "check_positions": len(sequences),
        "p_yes_check_mean": _mean(yes_probs),
        "p_yes_check_median": _median(yes_probs),
        "p_is_check_mean": _mean(is_probs),
        "p_no_check_mean": _mean(no_probs),
        "p_top_1_legal": _mean(top1_legal_values),
        "illegal_mass": _mean(illegal_masses),
        "top1_yes_check_count": top1_ids.count(YES_CHECK_ID),
        "top1_is_check_count": top1_ids.count(IS_CHECK_ID),
        "top1_no_check_count": top1_ids.count(NO_CHECK_ID),
        "top1_other_count": sum(
            1 for token_id in top1_ids if token_id not in {YES_CHECK_ID, IS_CHECK_ID, NO_CHECK_ID}
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
