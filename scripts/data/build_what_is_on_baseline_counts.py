"""Aggregate (square, ply) counts from tokenized training games for what_is_on baseline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from krasnal.config import MOVE_VOCAB_PATH, PRETRAIN_DATASET_PATH
from krasnal.dataset import PretrainDataset
from krasnal.eval.parsers import parse_row_to_game_tokens, split_packed_window_token_ids
from krasnal.eval.replayer import replay_games
from krasnal.eval.what_is_on_baseline import accumulate_from_eval_contexts
from krasnal.tokens import load_move_vocab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-parquet",
        type=Path,
        default=PRETRAIN_DATASET_PATH,
        help="Packed training parquet (default: krasnal pretrain path)",
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

    ds = PretrainDataset(args.train_parquet)
    window_indices = list(range(len(ds)))
    if args.max_games > 0:
        rng = random.Random(args.seed)
        rng.shuffle(window_indices)
        window_indices = window_indices[: max(1, args.max_games // 4)]

    all_ctx = []
    block_size = int(args.block_size)
    games_seen = 0
    for idx in window_indices:
        tokens, active, opponent, _segment, _position = ds[idx]
        token_list = tokens.tolist()
        for start, end in _game_spans(token_list):
            games_seen += 1
            if args.max_games > 0 and games_seen > args.max_games:
                break
            game_row = (
                tokens[start:end],
                active[start:end],
                opponent[start:end],
            )
            game_tokens = parse_row_to_game_tokens(game_row)
            if game_tokens is None:
                continue
            all_ctx.extend(replay_games([game_tokens], block_size))
        if args.max_games > 0 and games_seen > args.max_games:
            break

    stats = accumulate_from_eval_contexts(all_ctx)
    stats.dump(args.output)
    print(f"Wrote {args.output} from {games_seen} games ({len(all_ctx)} positions)")


def _game_spans(token_list: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for game_tokens in split_packed_window_token_ids(token_list):
        n = len(game_tokens)
        spans.append((offset, offset + n))
        offset += n
    return spans


if __name__ == "__main__":
    main()
