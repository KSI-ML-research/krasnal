from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def print_results(df: pl.DataFrame, logger: logging.Logger) -> None:
    if df.is_empty():
        logger.info("No positions evaluated.")
        return

    has_cot = "cot_acpl" in df.columns

    total_n = len(df)
    logger.info("")
    logger.info("=== Evaluation Results ===")
    logger.info("")
    logger.info(f"Positions evaluated:      {total_n:,}")

    def _print_summary(label: str, top1_col: str, illegal_col: str, acpl_col: str):
        legal_rate = float(df[top1_col].drop_nulls().mean() * 100)
        avg_illegal_mass = float(df[illegal_col].drop_nulls().mean() * 100)
        logger.info("")
        logger.info(f"{label}")
        logger.info(f"Top-1 Legal Move Rate:    {legal_rate:.1f}%")
        logger.info(f"Mean Illegal Prob Mass:   {avg_illegal_mass:.1f}%")
        acpl_df = df.filter(pl.col(acpl_col).is_not_null())
        if not acpl_df.is_empty():
            avg_acpl = acpl_df[acpl_col].mean()
            logger.info(f"Average Centipawn Loss:   {avg_acpl:.1f}")

    _print_summary("Baseline (No Thinking)", "top1_legal", "illegal_mass", "acpl")

    if has_cot:
        acpl_df = df.filter(pl.col("cot_acpl").is_not_null())
        if not acpl_df.is_empty():
            avg_acpl = acpl_df["cot_acpl"].mean()
            logger.info("")
            logger.info("With <think> Block")
            logger.info(f"Average Centipawn Loss:   {avg_acpl:.1f}")

    logger.info("")
    logger.info("Metrics by Phase:")
    logger.info(f"{'Phase':<18} | {'Legal %':>8} | {'Ill. Mass %':>12} | {'ACPL':>8}")
    logger.info("-" * 57)
    for label, (start, end) in [
        ("Opening (1-10)", (1, 10)),
        ("Middle (11-30)", (11, 30)),
        ("Endgame (31+)", (31, 999)),
    ]:
        pdf = df.filter(pl.col("move_num").is_between(start, end))
        if pdf.is_empty():
            continue
        acpl = pdf["acpl"].drop_nulls().mean()
        legal_r = float(pdf["top1_legal"].mean() * 100)
        ill_m = float(pdf["illegal_mass"].mean() * 100)
        acpl_s = f"{acpl:.1f}" if acpl else "N/A"
        logger.info(f"{label:<18} | {legal_r:>7.1f}% | {ill_m:>11.1f}% | {acpl_s:>8}")

    logger.info("")
    logger.info("Trend by Move Number:")
    trends = _bin_by_move_number(df)
    for row in trends.iter_rows(named=True):
        start = row["bin_start"]
        l_rate = float(row["top1_legal"] * 100)
        acpl = row["acpl"]
        acpl_str = f"{acpl:>5.0f}" if acpl is not None else "  N/A"
        bar = "#" * int(l_rate / 5)
        logger.info(
            f"Moves {start:2}-{start + 9:<2} | Legal: {l_rate:>5.1f}% | ACPL: {acpl_str} | {bar}"
        )


def _bin_by_move_number(df: pl.DataFrame) -> pl.DataFrame:
    """Bin moves into groups of 10 and compute mean metrics."""
    return (
        df.with_columns((((pl.col("move_num") - 1) // 10) * 10 + 1).alias("bin_start"))
        .group_by("bin_start")
        .mean()
        .sort("bin_start")
    )


def save_plot(df: pl.DataFrame, path: str) -> None:
    """Save a trend plot of Legal Rate and ACPL by move number."""
    trends = _bin_by_move_number(df)
    labels = [f"{b}–{b + 9}" for b in trends["bin_start"]]

    fig, ax_legal = plt.subplots(figsize=(10, 4))
    ax_legal.plot(labels, trends["top1_legal"] * 100, marker="o", color="steelblue")
    ax_legal.set(ylabel="Top-1 Legal Rate (%)", title="Evaluation Trends", ylim=(0, 105))
    ax_legal.grid(axis="y", alpha=0.3)

    if df["acpl"].is_not_null().any():
        ax_acpl = ax_legal.twinx()
        ax_acpl.plot(labels, trends["acpl"], marker="s", color="tomato", alpha=0.7)
        ax_acpl.set(ylabel="ACPL")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
