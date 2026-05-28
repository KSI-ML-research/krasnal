"""Token mix and sequence length statistics for preprocessed datasets."""

from __future__ import annotations

import polars as pl
from loguru import logger
from omegaconf import DictConfig

import wandb
from krasnal.tokens import (
    BLACK_WON_ID,
    COLORED_PIECE_TOKENS,
    DRAW_ID,
    ELO_TOKENS,
    EMPTY_ID,
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    NO_CHECK_ID,
    SPECIAL_TOKENS,
    TC_TOKENS,
    UNKNOWN_RESULT_ID,
    WHATS_ON_SQUARE_TOKEN_IDS,
    WHITE_WON_ID,
    YES_CHECK_ID,
)

# Declarative counter definitions for token mix analysis.
# Each entry is (counter_name, single_token_id | frozenset_of_ids).
_TOKEN_MIX_COUNTERS: list[tuple[str, int | frozenset[int]]] = [
    ("is_check_count", IS_CHECK_ID),
    ("yes_check_count", YES_CHECK_ID),
    ("no_check_count", NO_CHECK_ID),
    ("what_is_on_count", frozenset(WHATS_ON_SQUARE_TOKEN_IDS)),
    ("empty_count", EMPTY_ID),
    ("piece_answer_count", frozenset(COLORED_PIECE_TOKENS.values())),
    ("result_count", frozenset({WHITE_WON_ID, BLACK_WON_ID, DRAW_ID, UNKNOWN_RESULT_ID})),
    ("elo_count", frozenset(ELO_TOKENS.values())),
    ("tc_count", frozenset(TC_TOKENS.values())),
    ("special_count", frozenset(SPECIAL_TOKENS.values())),
    ("game_start_count", GAME_START_ID),
    ("game_end_count", GAME_END_ID),
    *((f"elo_{name}_count", bid) for name, bid in ELO_TOKENS.items()),
    *((f"tc_{name}_count", bid) for name, bid in TC_TOKENS.items()),
]


def _token_mix_raw_sums(tokenized_lf: pl.LazyFrame) -> dict[str, int]:
    total = tokenized_lf.select(pl.col("token_ids").list.len().sum()).collect().item() or 0
    counts_df = (
        tokenized_lf.select(pl.col("token_ids").explode().alias("tid"))
        .group_by("tid")
        .len()
        .collect()
    )
    id_to_count: dict[int, int] = dict(
        zip(counts_df["tid"].to_list(), counts_df["len"].to_list(), strict=True)
    )

    result: dict[str, int] = {"total_tokens": int(total)}
    for name, ids in _TOKEN_MIX_COUNTERS:
        if isinstance(ids, int):
            result[name] = id_to_count.get(ids, 0)
        else:
            result[name] = sum(id_to_count.get(i, 0) for i in ids)
    return result


def merge_token_mix_raw(
    acc: dict[str, int] | None,
    part: dict[str, int],
) -> dict[str, int]:
    if acc is None:
        return part
    return {k: acc.get(k, 0) + v for k, v in part.items()}


def token_mix_from_raw_sums(raw: dict[str, int]) -> dict[str, float]:
    total_tokens = raw["total_tokens"]
    check_qa_count = raw["is_check_count"] + raw["yes_check_count"] + raw["no_check_count"]
    outcome_prefix_count = raw["result_count"] + raw["elo_count"] + raw["tc_count"]
    uci_move_count = max(0, total_tokens - raw["special_count"])
    whats_on_answer_count = raw["empty_count"] + raw["piece_answer_count"]

    def pct(count: int) -> float:
        return (count / total_tokens * 100.0) if total_tokens > 0 else 0.0

    result: dict[str, float] = {
        "total_tokens": total_tokens,
        "uci_move_count": uci_move_count,
        "check_qa_count": check_qa_count,
        "outcome_prefix_count": outcome_prefix_count,
        "is_check_count": raw["is_check_count"],
        "yes_check_count": raw["yes_check_count"],
        "no_check_count": raw["no_check_count"],
        "result_count": raw["result_count"],
        "elo_count": raw["elo_count"],
        "tc_count": raw["tc_count"],
        "uci_move_pct": pct(uci_move_count),
        "check_qa_pct": pct(check_qa_count),
        "outcome_prefix_pct": pct(outcome_prefix_count),
        "what_is_on_count": raw["what_is_on_count"],
        "what_is_on_pct": pct(raw["what_is_on_count"]),
        "empty_count": raw["empty_count"],
        "empty_pct": pct(raw["empty_count"]),
        "piece_answer_count": raw["piece_answer_count"],
        "piece_answer_pct": pct(raw["piece_answer_count"]),
        "whats_on_answer_count": whats_on_answer_count,
        "whats_on_answer_pct": pct(whats_on_answer_count),
        "game_start_count": raw["game_start_count"],
        "game_end_count": raw["game_end_count"],
        "game_start_pct": pct(raw["game_start_count"]),
        "game_end_pct": pct(raw["game_end_count"]),
    }

    for bucket_name in ELO_TOKENS:
        result[f"elo_{bucket_name}_count"] = float(raw[f"elo_{bucket_name}_count"])
    for bucket_name in TC_TOKENS:
        result[f"tc_{bucket_name}_count"] = float(raw[f"tc_{bucket_name}_count"])

    return result


def compute_token_mix_stats(tokenized_lf: pl.LazyFrame) -> dict[str, float]:
    return token_mix_from_raw_sums(_token_mix_raw_sums(tokenized_lf))


def seq_len_stats(seq_len_lf: pl.LazyFrame, block_size: int) -> dict[str, float]:
    stats = seq_len_lf.select(
        pl.col("len").count().alias("total"),
        pl.col("len").min().alias("min"),
        pl.col("len").max().alias("max"),
        pl.col("len").mean().alias("mean"),
        pl.col("len").median().alias("median"),
        pl.col("len").std().alias("std"),
        pl.col("len").quantile(0.95).alias("p95"),
        pl.col("len").quantile(0.99).alias("p99"),
        pl.col("len").quantile(0.999).alias("p999"),
        (pl.col("len") > block_size).sum().alias("over_block_size"),
    ).collect()
    return {
        "total": stats.item(0, "total"),
        "min": stats.item(0, "min"),
        "max": stats.item(0, "max"),
        "mean": stats.item(0, "mean"),
        "median": stats.item(0, "median"),
        "std": stats.item(0, "std"),
        "p95": stats.item(0, "p95"),
        "p99": stats.item(0, "p99"),
        "p999": stats.item(0, "p999"),
        "over_block_size": stats.item(0, "over_block_size"),
    }


def log_preprocess_to_wandb(
    *,
    cfg: DictConfig,
    token_mix: dict[str, float],
    seq_stats: dict[str, float],
    total_games: int,
    train_window_rows: int,
    eval_rows: int,
    over_block_size_count: int,
    eval_elo_bins: dict[str, int] | None = None,
) -> None:
    """Log dataset statistics to a W&B run tagged 'preprocess'."""
    project = str(cfg.get("wandb_project", "uwr-ksai/krasnal"))
    wandb.init(project=project, job_type="preprocess", tags=["preprocess"])

    wandb.summary["dataset/total_games"] = total_games
    wandb.summary["dataset/train_window_rows"] = train_window_rows
    wandb.summary["dataset/eval_rows"] = eval_rows
    wandb.summary["dataset/removed_over_context_games"] = over_block_size_count
    wandb.summary["dataset/removed_over_context_game_pct"] = (
        over_block_size_count / total_games * 100.0 if total_games > 0 else 0.0
    )

    wandb.summary["dataset/seq_len_min"] = seq_stats["min"]
    wandb.summary["dataset/seq_len_max"] = seq_stats["max"]
    wandb.summary["dataset/seq_len_mean"] = seq_stats["mean"]
    wandb.summary["dataset/seq_len_p95"] = seq_stats["p95"]
    wandb.summary["dataset/seq_len_p99"] = seq_stats["p99"]

    wandb.summary["dataset/input_tokens"] = token_mix["total_tokens"]
    wandb.summary["dataset/input_token_pct/move"] = token_mix["uci_move_pct"]
    wandb.summary["dataset/input_token_pct/check_qa"] = token_mix["check_qa_pct"]
    wandb.summary["dataset/input_token_pct/what_is_on_prompt"] = token_mix["what_is_on_pct"]
    wandb.summary["dataset/input_token_pct/what_is_on_answer"] = token_mix["whats_on_answer_pct"]
    wandb.summary["dataset/input_token_pct/conditioning_prefix"] = token_mix["outcome_prefix_pct"]
    wandb.summary["dataset/input_token_pct/game_start"] = token_mix["game_start_pct"]
    wandb.summary["dataset/input_token_pct/game_end"] = token_mix["game_end_pct"]

    total_elo = sum(token_mix.get(f"elo_{b}_count", 0) for b in ELO_TOKENS)
    for bucket_name in ELO_TOKENS:
        count = token_mix.get(f"elo_{bucket_name}_count", 0)
        pct = (count / total_elo * 100.0) if total_elo > 0 else 0.0
        wandb.summary[f"dataset/elo/{bucket_name}"] = pct

    total_tc = sum(token_mix.get(f"tc_{b}_count", 0) for b in TC_TOKENS)
    for bucket_name in TC_TOKENS:
        count = token_mix.get(f"tc_{bucket_name}_count", 0)
        pct = (count / total_tc * 100.0) if total_tc > 0 else 0.0
        wandb.summary[f"dataset/tc/{bucket_name}"] = pct

    if eval_elo_bins:
        for bucket_name, count in eval_elo_bins.items():
            wandb.summary[f"dataset/eval_elo/{bucket_name}"] = count

    wandb.finish()
    logger.info("Preprocessing statistics logged to W&B project '{}'", project)
