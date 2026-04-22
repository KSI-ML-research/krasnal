#!/usr/bin/env python3
"""Count `<elo_unknown>` usage in tokenized pretrain dataset."""

import argparse
from pathlib import Path

import polars as pl

from krasnal.config import PRETRAIN_DATASET_PATH
from krasnal.tokens import ELO_UNKNOWN_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count unknown ELO tokens in a parquet dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PRETRAIN_DATASET_PATH,
        help="Path to tokenized parquet with token_ids column",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    lf = pl.scan_parquet(dataset_path)

    white_unknown_expr = pl.col("token_ids").list.get(2).eq(ELO_UNKNOWN_ID).fill_null(False)
    black_unknown_expr = pl.col("token_ids").list.get(3).eq(ELO_UNKNOWN_ID).fill_null(False)

    stats = (
        lf.select(
            pl.len().alias("total_rows"),
            pl.col("token_ids")
            .list.eval((pl.element() == ELO_UNKNOWN_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("unknown_token_count"),
            white_unknown_expr.cast(pl.UInt32).sum().alias("white_unknown_rows"),
            black_unknown_expr.cast(pl.UInt32).sum().alias("black_unknown_rows"),
            (white_unknown_expr | black_unknown_expr)
            .cast(pl.UInt32)
            .sum()
            .alias("rows_with_any_unknown"),
            (white_unknown_expr & black_unknown_expr)
            .cast(pl.UInt32)
            .sum()
            .alias("rows_with_both_unknown"),
        )
        .collect()
        .row(0)
    )

    total_rows = int(stats[0] or 0)
    unknown_token_count = int(stats[1] or 0)
    white_unknown_rows = int(stats[2] or 0)
    black_unknown_rows = int(stats[3] or 0)
    rows_with_any_unknown = int(stats[4] or 0)
    rows_with_both_unknown = int(stats[5] or 0)

    def pct(count: int) -> float:
        return (count / total_rows * 100.0) if total_rows else 0.0

    print(f"dataset: {dataset_path}")
    print(f"elo_unknown_token_id: {ELO_UNKNOWN_ID}")
    print(f"rows: {total_rows}")
    print(f"unknown tokens (global): {unknown_token_count}")
    print(f"rows with white elo unknown: {white_unknown_rows} ({pct(white_unknown_rows):.4f}%)")
    print(f"rows with black elo unknown: {black_unknown_rows} ({pct(black_unknown_rows):.4f}%)")
    print(f"rows with any unknown elo: {rows_with_any_unknown} ({pct(rows_with_any_unknown):.4f}%)")
    both_pct = pct(rows_with_both_unknown)
    print(f"rows with both unknown elos: {rows_with_both_unknown} ({both_pct:.4f}%)")


if __name__ == "__main__":
    main()
