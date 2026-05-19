"""Aggregate (square, ply) counts from tokenized training games for what_is_on baseline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from krasnal.config import MOVE_VOCAB_PATH, PRETRAIN_DATASET_PATH
from krasnal.dataset import ChessDataset
from krasnal.eval.parsers import parse_row_to_game_tokens
from krasnal.eval.replayer import replay_games
from krasnal.eval.what_is_on_baseline import accumulate_from_eval_contexts
from krasnal.tokens import load_move_vocab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-parquet",
        type=Path,
        default=PRETRAIN_DATASET_PATH,
        help="Training parquet (default: krasnal pretrain path)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/2_tokenized/what_is_on_baseline_counts.json"),
    )
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="If >0, shuffle and cap games (for smoke tests)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-include-elo",
        action="store_true",
        help="Disable Elo filtering when loading rows (match preprocess include_elo)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_move_vocab(
        MOVE_VOCAB_PATH,
        piece_aware_moves=True,
        side_prefixed_moves=True,
    )

    if not args.train_parquet.exists():
        raise FileNotFoundError(args.train_parquet)

    ds = ChessDataset(args.train_parquet, include_elo=not args.no_include_elo)
    indices = list(range(len(ds)))
    if args.max_games > 0:
        rng = random.Random(args.seed)
        rng.shuffle(indices)
        indices = indices[: args.max_games]

    all_ctx = []
    block_size = int(args.block_size)
    for idx in indices:
        row = ds[idx]
        game_tokens = parse_row_to_game_tokens(row)
        if game_tokens is None:
            continue

        all_ctx.extend(replay_games([game_tokens], block_size))

    stats = accumulate_from_eval_contexts(all_ctx)
    stats.dump(args.output)
    print(f"Wrote {args.output} from {len(indices)} games ({len(all_ctx)} positions)")


if __name__ == "__main__":
    main()
