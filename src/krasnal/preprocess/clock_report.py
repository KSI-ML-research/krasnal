"""Clock coverage diagnostics for preprocessed eval datasets."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from krasnal.config import CLOCK_IGNORE_ID, EVAL_DATASET_PATH
from krasnal.tokens import get_move_clock_pairs, get_moves_only

_THRESHOLD = 30

_BUCKET_LABELS = ("<10s", "10-30s", "30-60s", "60-120s", "120-300s", ">300s")


def _bucket(seconds: int) -> str:
    if seconds < 10:
        return "<10s"
    if seconds < 30:
        return "10-30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    if seconds < 300:
        return "120-300s"
    return ">300s"


def run_clock_report(path: Path = EVAL_DATASET_PATH, threshold: int = _THRESHOLD) -> None:
    df = (
        pl.scan_parquet(path)
        .select(pl.col("token_ids", "active_clock_ids", "opponent_clock_ids"))
        .collect()
    )

    total_games = len(df)
    total_plies = 0
    plies_both = 0
    games_with_clock = 0
    plies_low_active = 0
    plies_low_both = 0
    buckets = {label: 0 for label in _BUCKET_LABELS}

    for row in df.iter_rows(named=True):
        token_ids = [int(x) for x in row["token_ids"]]
        act = [int(x) for x in row["active_clock_ids"]]
        opp = [int(x) for x in row["opponent_clock_ids"]]
        moves = get_moves_only(token_ids)
        pairs = get_move_clock_pairs(token_ids, act, opp)

        if pairs is None or len(pairs) != len(moves):
            continue

        game_has_clock = False
        for a, o in pairs:
            total_plies += 1
            a_ok = a != CLOCK_IGNORE_ID
            o_ok = o != CLOCK_IGNORE_ID

            if a_ok and o_ok:
                plies_both += 1
                game_has_clock = True

            if a_ok:
                if a < threshold:
                    plies_low_active += 1
                    if o_ok and o < threshold:
                        plies_low_both += 1
                buckets[_bucket(a)] += 1

        if game_has_clock:
            games_with_clock += 1

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
        pct = 100.0 * buckets[label] / plies_with_active if plies_with_active else 0.0
        logger.info(f"  {label:>8s}: {pct:.2f}%")
