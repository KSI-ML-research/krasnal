import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import blake2b
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from krasnal.config import (
    EVAL_DATASET_PATH,
    PRETRAIN_DATASET_PATH,
    RAW_UCI_DIR,
)
from krasnal.tokens import (
    BLACK_WON_ID,
    DRAW_ID,
    ELO_1000_1499_ID,
    ELO_1500_1999_ID,
    ELO_2000_2499_ID,
    ELO_2500_2999_ID,
    ELO_ABOVE_3000_ID,
    ELO_BELOW_1000_ID,
    ELO_UNKNOWN_ID,
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    NO_CHECK_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNKNOWN_RESULT_ID,
    WHITE_WON_ID,
    YES_CHECK_ID,
    get_elo_bucket,
    move_token_id_for_ply,
    result_to_token_id,
    set_side_prefixed_moves,
)


def _sample_bool(seed: int, game_key: str, ply: int, probability: float) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    digest = blake2b(f"{seed}|{game_key}|{ply}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big") / 2**64
    return value < probability


def _build_game_tokens(
    uci_moves: str,
    is_check: list[bool],
    result: str,
    white_rating: int,
    black_rating: int,
    elo_bucket: int,
    include_check_qa: bool,
    normal_prob: float,
    white_unknown_prob: float,
    black_unknown_prob: float,
    both_unknown_prob: float,
    seed: int,
    p_no: float,
) -> list[int]:
    if not uci_moves:
        return []

    moves_list = uci_moves.split()

    result_tokens = []
    for ply, move in enumerate(moves_list):
        move_id = move_token_id_for_ply(move, ply)
        if move_id is None:
            move_id = PAD_ID
        result_tokens.append(move_id)

        if include_check_qa:
            gives_check = ply < len(is_check) and bool(is_check[ply])
            if gives_check:
                result_tokens.extend([IS_CHECK_ID, YES_CHECK_ID])
            elif _sample_bool(seed=seed, game_key=uci_moves, ply=ply, probability=p_no):
                result_tokens.extend([IS_CHECK_ID, NO_CHECK_ID])

    white_elo = get_elo_bucket(white_rating)
    black_elo = get_elo_bucket(black_rating)

    normal_threshold = round(normal_prob * 1000)
    white_unknown_threshold = normal_threshold + round(white_unknown_prob * 1000)
    black_unknown_threshold = white_unknown_threshold + round(black_unknown_prob * 1000)
    both_unknown_threshold = black_unknown_threshold + round(both_unknown_prob * 1000)

    if not 0 <= elo_bucket < 1000:
        raise ValueError(f"elo_bucket must be in [0, 1000), got {elo_bucket}")
    if both_unknown_threshold != 1000:
        raise ValueError(
            "Unknown ELO probabilities must sum to 1.0 in 0.001 increments; "
            f"got total={both_unknown_threshold / 1000:.3f}"
        )

    if elo_bucket < normal_threshold:
        pass
    elif elo_bucket < white_unknown_threshold:
        white_elo = ELO_UNKNOWN_ID
    elif elo_bucket < black_unknown_threshold:
        black_elo = ELO_UNKNOWN_ID
    elif elo_bucket < both_unknown_threshold:
        white_elo = ELO_UNKNOWN_ID
        black_elo = ELO_UNKNOWN_ID

    prefix_tokens = [
        GAME_START_ID,
        result_to_token_id(result),
        white_elo,
        black_elo,
    ]
    return prefix_tokens + result_tokens + [GAME_END_ID]


def process_file_streaming(
    parquet_path: Path,
    seed: int,
    output_path: Path,
    include_check_qa: bool,
    unknown_elo: dict[str, float],
) -> int:
    lf = pl.scan_parquet(parquet_path)

    if include_check_qa:
        count_stats = (
            lf.select(
                pl.col("is_check")
                .list.eval(pl.element().cast(pl.UInt32), parallel=True)
                .list.sum()
                .sum()
                .alias("check_count"),
                pl.col("is_check").list.len().sum().alias("ply_count"),
            )
            .collect()
            .row(0)
        )
        check_count = int(count_stats[0] or 0)
        ply_count = int(count_stats[1] or 0)
        no_check_count = max(0, ply_count - check_count)
        p_no = (check_count / no_check_count) if no_check_count > 0 else 0.0
        p_no = min(max(p_no, 0.0), 1.0)
    else:
        p_no = 1.0

    normal_prob = float(unknown_elo["normal_prob"])
    white_unknown_prob = float(unknown_elo["white_unknown_prob"])
    black_unknown_prob = float(unknown_elo["black_unknown_prob"])
    both_unknown_prob = float(unknown_elo["both_unknown_prob"])

    def build_tokens_batch(batch: pl.DataFrame) -> pl.DataFrame:
        token_ids_list = []
        for i in range(len(batch)):
            token_ids = _build_game_tokens(
                uci_moves=batch["uci_moves"][i],
                is_check=batch["is_check"][i],
                result=batch["result"][i],
                white_rating=batch["white_rating"][i],
                black_rating=batch["black_rating"][i],
                elo_bucket=batch["elo_bucket"][i],
                include_check_qa=include_check_qa,
                normal_prob=normal_prob,
                white_unknown_prob=white_unknown_prob,
                black_unknown_prob=black_unknown_prob,
                both_unknown_prob=both_unknown_prob,
                seed=seed,
                p_no=p_no,
            )
            token_ids_list.append(token_ids)

        return batch.select("split_bucket").with_columns(
            pl.Series("token_ids", token_ids_list, dtype=pl.List(pl.UInt16))
        )

    lf = lf.with_columns(
        [
            (pl.col("uci_moves").hash(seed=seed) % 1000).alias("split_bucket"),
            (pl.col("uci_moves").hash(seed=seed + 1) % 1000).alias("elo_bucket"),
        ],
    )

    row_count = lf.select(pl.len()).collect().item()
    lf.map_batches(
        build_tokens_batch,
        schema={"split_bucket": pl.UInt64, "token_ids": pl.List(pl.UInt16)},
    ).sink_parquet(output_path)
    return row_count


def _process_one_shard(
    parquet_path: Path,
    seed: int,
    output_path: Path,
    side_prefixed_moves: bool,
    include_check_qa: bool,
    unknown_elo: dict[str, float],
) -> tuple[str, int, str]:
    set_side_prefixed_moves(side_prefixed_moves)
    count = process_file_streaming(
        parquet_path, seed, output_path, include_check_qa=include_check_qa, unknown_elo=unknown_elo
    )
    return parquet_path.name, count, output_path.name


def compute_stats(tokenized_lf: pl.LazyFrame, block_size: int) -> dict[str, float]:
    seq_len_lf = tokenized_lf.select(pl.col("token_ids").list.len().alias("len"))

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


def compute_token_mix_stats(tokenized_lf: pl.LazyFrame) -> dict[str, float]:
    result_ids = [WHITE_WON_ID, BLACK_WON_ID, DRAW_ID, UNKNOWN_RESULT_ID]
    elo_ids = [
        ELO_BELOW_1000_ID,
        ELO_1000_1499_ID,
        ELO_1500_1999_ID,
        ELO_2000_2499_ID,
        ELO_2500_2999_ID,
        ELO_ABOVE_3000_ID,
        ELO_UNKNOWN_ID,
    ]
    special_ids = list(SPECIAL_TOKENS.values())

    stats = (
        tokenized_lf.select(
            pl.col("token_ids").list.len().sum().alias("total_tokens"),
            pl.col("token_ids")
            .list.eval((pl.element() == IS_CHECK_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("is_check_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == YES_CHECK_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("yes_check_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == NO_CHECK_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("no_check_count"),
            pl.col("token_ids")
            .list.eval(pl.element().is_in(result_ids).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("result_count"),
            pl.col("token_ids")
            .list.eval(pl.element().is_in(elo_ids).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("elo_count"),
            pl.col("token_ids")
            .list.eval(pl.element().is_in(special_ids).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("special_count"),
        )
        .collect()
        .row(0)
    )

    total_tokens = int(stats[0] or 0)
    is_check_count = int(stats[1] or 0)
    yes_check_count = int(stats[2] or 0)
    no_check_count = int(stats[3] or 0)
    result_count = int(stats[4] or 0)
    elo_count = int(stats[5] or 0)
    special_count = int(stats[6] or 0)

    check_qa_count = is_check_count + yes_check_count + no_check_count
    outcome_prefix_count = result_count + elo_count
    uci_move_count = max(0, total_tokens - special_count)

    def pct(count: int) -> float:
        return (count / total_tokens * 100.0) if total_tokens > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "uci_move_count": uci_move_count,
        "check_qa_count": check_qa_count,
        "outcome_prefix_count": outcome_prefix_count,
        "is_check_count": is_check_count,
        "yes_check_count": yes_check_count,
        "no_check_count": no_check_count,
        "result_count": result_count,
        "elo_count": elo_count,
        "uci_move_pct": pct(uci_move_count),
        "check_qa_pct": pct(check_qa_count),
        "outcome_prefix_pct": pct(outcome_prefix_count),
    }


def one_row_one_game(lazy_df: pl.LazyFrame, block_size: int) -> pl.LazyFrame:
    window_size = block_size + 1
    return lazy_df.select(pl.col("token_ids").list.slice(0, window_size).alias("token_ids"))


@hydra.main(version_base=None, config_path="../../config", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    set_side_prefixed_moves(bool(cfg.get("side_prefixed_moves", True)))
    block_size = int(cfg.block_size)
    seed = int(cfg.seed)
    include_check_qa = bool(cfg.get("include_check_qa", True))
    unknown_elo = {
        "normal_prob": float(cfg.unknown_elo.normal_prob),
        "white_unknown_prob": float(cfg.unknown_elo.white_unknown_prob),
        "black_unknown_prob": float(cfg.unknown_elo.black_unknown_prob),
        "both_unknown_prob": float(cfg.unknown_elo.both_unknown_prob),
    }
    if abs(sum(unknown_elo.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"unknown_elo probabilities must sum to 1.0, got {sum(unknown_elo.values())}"
        )

    parquet_files = sorted(RAW_UCI_DIR.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No Aix-filtered games found in {RAW_UCI_DIR}")
        return

    PRETRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = PRETRAIN_DATASET_PATH.parent / "temp_preprocess"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total_games = 0
    side_prefixed_moves = bool(cfg.get("side_prefixed_moves", True))
    workers_cfg = int(cfg.get("preprocess_workers", 0) or 0)
    max_workers = min(
        workers_cfg if workers_cfg > 0 else min(len(parquet_files), os.cpu_count() or 1), 8
    )
    logger.info("Processing {} shards with {} workers", len(parquet_files), max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, parquet_path in enumerate(parquet_files):
            output_path = temp_dir / f"part_{idx:04d}.parquet"
            future = executor.submit(
                _process_one_shard,
                parquet_path,
                seed,
                output_path,
                side_prefixed_moves,
                include_check_qa,
                unknown_elo,
            )
            futures[future] = parquet_path.name

        for future in as_completed(futures):
            parquet_name = futures[future]
            try:
                done_name, count, output_name = future.result()
                total_games += count
                logger.info("Processed {}: {} games -> {}", done_name, count, output_name)
            except Exception as e:
                logger.error("Failed to process {}: {}", parquet_name, e)

    all_parts = list(temp_dir.glob("part_*.parquet"))
    if not all_parts:
        logger.error("No data generated")
        return

    combined_lf = pl.concat(pl.scan_parquet(p) for p in all_parts)

    stats = compute_stats(combined_lf, block_size)
    if stats["total"] == 0:
        logger.error("No games found in raw dataset.")
        return

    logger.info(
        "Sequence length stats: total={}, min={}, max={}, mean={:.1f}, median={}, "
        "std={:.1f}, p95={}, p99={}, p999={}",
        stats["total"],
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["median"],
        stats["std"],
        stats["p95"],
        stats["p99"],
        stats["p999"],
    )

    max_tokens = block_size
    over_block_size_count = stats.get("over_block_size", 0)
    total_count = stats["total"]
    pct = over_block_size_count / total_count * 100
    logger.info(
        "Games with >{} tokens: {} ({:.2f}%) - filtering out >{}",
        max_tokens,
        over_block_size_count,
        pct,
        max_tokens,
    )

    filtered_lf = combined_lf.filter(pl.col("token_ids").list.len() <= max_tokens)

    token_mix = compute_token_mix_stats(filtered_lf.select("token_ids"))
    logger.info(
        "Token mix after >block_size filtering: UCI moves={} ({:.2f}%), "
        "check QA={} ({:.2f}%), outcome prefix={} ({:.2f}%)",
        token_mix["uci_move_count"],
        token_mix["uci_move_pct"],
        token_mix["check_qa_count"],
        token_mix["check_qa_pct"],
        token_mix["outcome_prefix_count"],
        token_mix["outcome_prefix_pct"],
    )
    logger.info(
        "Token mix details: <is_check>={}, <yes_check>={}, <no_check>={}, "
        "result_tokens={}, elo_tokens={}, total_tokens={}",
        token_mix["is_check_count"],
        token_mix["yes_check_count"],
        token_mix["no_check_count"],
        token_mix["result_count"],
        token_mix["elo_count"],
        token_mix["total_tokens"],
    )

    train_lf = one_row_one_game(
        filtered_lf.filter(pl.col("split_bucket") != 0).select("token_ids"),
        block_size=block_size,
    )
    eval_lf = one_row_one_game(
        filtered_lf.filter(pl.col("split_bucket") == 0).select("token_ids"),
        block_size=block_size,
    )

    train_lf.sink_parquet(PRETRAIN_DATASET_PATH)
    eval_lf.sink_parquet(EVAL_DATASET_PATH)

    shutil.rmtree(temp_dir)

    train_rows = pl.scan_parquet(PRETRAIN_DATASET_PATH).select(pl.len()).collect().item()
    eval_rows = pl.scan_parquet(EVAL_DATASET_PATH).select(pl.len()).collect().item()
    if train_rows == 0:
        logger.error("Train dataset is empty. Increase input data or reduce block_size.")
        return

    logger.info(
        "Successfully processed {} games -> {} (one-row-one-game, train rows: {}, eval rows: {})",
        stats["total"],
        PRETRAIN_DATASET_PATH.parent,
        train_rows,
        eval_rows,
    )


if __name__ == "__main__":
    main()
