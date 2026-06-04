"""Token mix and sequence length statistics for preprocessed datasets."""

from __future__ import annotations

import math

import wandb
from loguru import logger
from omegaconf import DictConfig

from krasnal.tokens import (
    BLACK_WON_ID,
    COLORED_PIECE_TOKENS,
    DRAW_ID,
    ELO_TOKENS,
    EMPTY_ID,
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    MAX_SIDE_MATERIAL,
    NO_CHECK_ID,
    OPP_MATERIAL_START_ID,
    OPP_MATERIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    TC_TOKENS,
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
    ("result_count", frozenset({WHITE_WON_ID, BLACK_WON_ID, DRAW_ID})),
    ("elo_count", frozenset(ELO_TOKENS.values())),
    ("tc_count", frozenset(TC_TOKENS.values())),
    ("material_count", frozenset(OPP_MATERIAL_TOKEN_IDS)),
    ("special_count", frozenset(SPECIAL_TOKENS.values())),
    ("game_start_count", GAME_START_ID),
    ("game_end_count", GAME_END_ID),
    *((f"elo_{name}_count", bid) for name, bid in ELO_TOKENS.items()),
    *((f"tc_{name}_count", bid) for name, bid in TC_TOKENS.items()),
    *(
        (f"opp_mat_{points}_count", OPP_MATERIAL_START_ID + points)
        for points in range(MAX_SIDE_MATERIAL + 1)
    ),
]


def token_mix_raw_from_counts(id_counts: dict[int, int]) -> dict[str, int]:
    result = {"total_tokens": sum(id_counts.values())}
    for name, ids in _TOKEN_MIX_COUNTERS:
        if isinstance(ids, int):
            result[name] = id_counts.get(ids, 0)
        else:
            result[name] = sum(id_counts.get(i, 0) for i in ids)
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
        "material_count": raw["material_count"],
        "uci_move_pct": pct(uci_move_count),
        "check_qa_pct": pct(check_qa_count),
        "outcome_prefix_pct": pct(outcome_prefix_count),
        "material_pct": pct(raw["material_count"]),
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
    for points in range(MAX_SIDE_MATERIAL + 1):
        result[f"opp_mat_{points}_count"] = float(raw[f"opp_mat_{points}_count"])

    return result


def merge_seq_len_raw(
    acc: dict[int, int] | None,
    part: dict[int, int],
) -> dict[int, int]:
    if acc is None:
        return part
    for length, count in part.items():
        acc[length] = acc.get(length, 0) + count
    return acc


def seq_len_stats_from_counts(length_counts: dict[int, int], block_size: int) -> dict[str, float]:
    total = sum(length_counts.values())
    if total == 0:
        return {
            "total": 0,
            "min": None,
            "max": None,
            "mean": 0.0,
            "median": None,
            "std": 0.0,
            "p95": None,
            "p99": None,
            "p999": None,
            "over_block_size": 0,
        }

    ordered = sorted(length_counts.items())
    total_sum = sum(length * count for length, count in ordered)
    mean = total_sum / total
    variance = sum(((length - mean) ** 2) * count for length, count in ordered) / total

    def quantile(q: float) -> int:
        target = max(1, math.ceil(total * q))
        seen = 0
        for length, count in ordered:
            seen += count
            if seen >= target:
                return length
        return ordered[-1][0]

    return {
        "total": total,
        "min": ordered[0][0],
        "max": ordered[-1][0],
        "mean": mean,
        "median": quantile(0.5),
        "std": math.sqrt(variance),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "p999": quantile(0.999),
        "over_block_size": sum(
            count for length, count in length_counts.items() if length > block_size
        ),
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
    wandb.summary["dataset/input_token_pct/opponent_material"] = token_mix["material_pct"]
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

    total_material = token_mix.get("material_count", 0)
    for points in range(MAX_SIDE_MATERIAL + 1):
        count = token_mix.get(f"opp_mat_{points}_count", 0)
        pct = (count / total_material * 100.0) if total_material > 0 else 0.0
        wandb.summary[f"dataset/opponent_material/{points}"] = count
        wandb.summary[f"dataset/opponent_material_pct/{points}"] = pct

    if eval_elo_bins:
        for bucket_name, count in eval_elo_bins.items():
            wandb.summary[f"dataset/eval_elo/{bucket_name}"] = count

    wandb.finish()
    logger.info("Preprocessing statistics logged to W&B project '{}'", project)
