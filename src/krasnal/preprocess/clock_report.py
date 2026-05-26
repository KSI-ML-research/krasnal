"""Clock coverage diagnostics for preprocessed eval datasets."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from krasnal.config import CLOCK_IGNORE_ID, EVAL_DATASET_PATH
from krasnal.tokens import SPECIAL_TOKENS

_THRESHOLD = 30

_BUCKET_EDGES = [10, 30, 60, 120, 300]
_BUCKET_LABELS = ("<10s", "10-30s", "30-60s", "60-120s", "120-300s", ">300s")

_SPECIAL_IDS = frozenset(SPECIAL_TOKENS.values())


def run_clock_report(path: Path = EVAL_DATASET_PATH, threshold: int = _THRESHOLD) -> None:
    df = pl.read_parquet(path)
    total_games = len(df)

    # Explode into per-token rows, filter to move tokens only
    exploded = (
        df.select(
            pl.int_range(pl.len()).alias("game_idx"),
            pl.col("token_ids"),
            pl.col("active_clock_ids"),
            pl.col("opponent_clock_ids"),
        )
        .explode("token_ids", "active_clock_ids", "opponent_clock_ids")
        .filter(~pl.col("token_ids").is_in(list(_SPECIAL_IDS)))
        .rename({"active_clock_ids": "active", "opponent_clock_ids": "opponent"})
    )

    total_plies = len(exploded)
    if total_plies == 0:
        logger.info("Clock Report — no move plies found")
        return

    both_known = exploded.filter(
        (pl.col("active") != CLOCK_IGNORE_ID) & (pl.col("opponent") != CLOCK_IGNORE_ID)
    )
    plies_both = len(both_known)
    games_with_clock = both_known["game_idx"].n_unique()

    active_known = exploded.filter(pl.col("active") != CLOCK_IGNORE_ID)
    plies_low_active = active_known.filter(pl.col("active") < threshold).height
    plies_low_both = both_known.filter(
        (pl.col("active") < threshold) & (pl.col("opponent") < threshold)
    ).height

    # Bucket distribution
    bucket_counts = (
        active_known.select(
            pl.when(pl.col("active") < 10)
            .then(pl.lit("<10s"))
            .when(pl.col("active") < 30)
            .then(pl.lit("10-30s"))
            .when(pl.col("active") < 60)
            .then(pl.lit("30-60s"))
            .when(pl.col("active") < 120)
            .then(pl.lit("60-120s"))
            .when(pl.col("active") < 300)
            .then(pl.lit("120-300s"))
            .otherwise(pl.lit(">300s"))
            .alias("bucket")
        )
        .group_by("bucket")
        .len()
    )
    buckets = dict(
        zip(bucket_counts["bucket"].to_list(), bucket_counts["len"].to_list(), strict=True)
    )

    logger.info(f"Clock Report — {path}")
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total move plies: {total_plies}")

    pct_both = 100.0 * plies_both / total_plies if total_plies else 0.0
    pct_games = 100.0 * games_with_clock / total_games if total_games else 0.0
    logger.info("Clock Coverage:")
    logger.info(f"  positions with both clocks known:  {pct_both:.2f}% ({plies_both})")
    logger.info(f"  games with any clock data:         {pct_games:.2f}% ({games_with_clock})")

    pct_low_clocked = 100.0 * plies_low_active / plies_both if plies_both else 0.0
    pct_low_all = 100.0 * plies_low_active / total_plies if total_plies else 0.0
    pct_low_both = 100.0 * plies_low_both / plies_both if plies_both else 0.0
    logger.info(f"Low Time (<{threshold}s side to move):")
    logger.info(
        f"  among clocked positions:            {pct_low_clocked:.2f}% ({plies_low_active})"
    )
    logger.info(f"  among all positions:                {pct_low_all:.2f}% ({plies_low_active})")
    logger.info(
        f"  both players <{threshold}s:                  {pct_low_both:.2f}% ({plies_low_both})"
    )

    plies_with_active = sum(buckets.values())
    logger.info("Clock Distribution (active clock):")
    for label in _BUCKET_LABELS:
        pct = 100.0 * buckets.get(label, 0) / plies_with_active if plies_with_active else 0.0
        logger.info(f"  {label:>8s}: {pct:.2f}%")
