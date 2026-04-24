import shutil
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
    BLACK_PREFIX,
    CHECK_ID,
    GAME_END_ID,
    GAME_START_ID,
    MOVE_TO_ID,
    PAD_ID,
    WHITE_PREFIX,
    get_elo_bucket,
    result_to_token_id,
)


def get_moves_dict() -> dict[str, int]:
    return {
        k: v
        for k, v in MOVE_TO_ID.items()
        if k.startswith(WHITE_PREFIX) or k.startswith(BLACK_PREFIX)
    }


def _build_game_tokens(
    uci_moves: str,
    is_check: list[bool],
    result: str,
    white_rating: int,
    black_rating: int,
    moves_dict: dict[str, int],
) -> list[int]:
    if not uci_moves:
        return []

    moves_list = uci_moves.split()

    result_tokens = []
    for ply, move in enumerate(moves_list):
        prefix = WHITE_PREFIX if ply % 2 == 0 else BLACK_PREFIX
        prefixed_move = prefix + move
        move_id = moves_dict.get(prefixed_move, PAD_ID)
        result_tokens.append(move_id)

        if ply > 0 and ply - 1 < len(is_check) and is_check[ply - 1]:
            result_tokens.append(CHECK_ID)

    white_elo = get_elo_bucket(white_rating)
    black_elo = get_elo_bucket(black_rating)

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
) -> int:
    moves_dict = get_moves_dict()

    def build_tokens_batch(batch: pl.DataFrame) -> pl.DataFrame:
        token_ids_list = []
        for i in range(len(batch)):
            token_ids = _build_game_tokens(
                uci_moves=batch["uci_moves"][i],
                is_check=batch["is_check"][i],
                result=batch["result"][i],
                white_rating=batch["white_rating"][i],
                black_rating=batch["black_rating"][i],
                moves_dict=moves_dict,
            )
            token_ids_list.append(token_ids)

        return batch.select("split_bucket").with_columns(
            pl.Series("token_ids", token_ids_list, dtype=pl.List(pl.UInt16))
        )

    lf = pl.scan_parquet(parquet_path)

    lf = lf.with_columns(
        [(pl.col("uci_moves").hash(seed=seed) % 1000).alias("split_bucket")],
    )

    row_count = lf.select(pl.len()).collect().item()
    lf.map_batches(
        build_tokens_batch, schema={"split_bucket": pl.UInt64, "token_ids": pl.List(pl.UInt16)}
    ).sink_parquet(output_path)
    return row_count


def compute_stats(tokenized_lf: pl.LazyFrame) -> dict[str, float]:
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
        (pl.col("len") > 256).sum().alias("over_256"),
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
        "over_256": stats.item(0, "over_256"),
    }


def one_row_one_game(lazy_df: pl.LazyFrame, block_size: int) -> pl.LazyFrame:
    window_size = block_size + 1
    return lazy_df.select(pl.col("token_ids").list.slice(0, window_size).alias("token_ids"))


@hydra.main(version_base=None, config_path="../../config", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    block_size = int(cfg.model.block_size)
    seed = int(cfg.seed)

    parquet_files = sorted(RAW_UCI_DIR.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No Aix-filtered games found in {RAW_UCI_DIR}")
        return

    PRETRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = PRETRAIN_DATASET_PATH.parent / "temp_preprocess"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total_games = 0

    for idx, pf in enumerate(parquet_files):
        logger.info(f"Processing {pf.name}...")
        output_path = temp_dir / f"part_{idx:04d}.parquet"
        try:
            count = process_file_streaming(pf, seed, output_path)
            total_games += count
            logger.info(f"  {count} games -> {output_path.name}")
        except Exception as e:
            logger.error(f"Failed to process {pf.name}: {e}")
            continue

    all_parts = list(temp_dir.glob("part_*.parquet"))
    if not all_parts:
        logger.error("No data generated")
        return

    combined_lf = pl.concat(pl.scan_parquet(p) for p in all_parts)

    stats = compute_stats(combined_lf)
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

    max_tokens = 256
    over_256_count = stats.get("over_256", 0)
    total_count = stats["total"]
    logger.info(
        "Games with >256 tokens: {} ({:.2f}%) - filtering out >256",
        over_256_count,
        over_256_count / total_count * 100,
    )

    filtered_lf = combined_lf.filter(pl.col("token_ids").list.len() <= max_tokens)

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
