import argparse
import logging

import polars as pl

from config import (
    EVAL_DATASET_PATH,
    MOVES_FILE,
    PRETRAIN_DATASET_PATH,
    RAW_DATA_DIR,
    GPTConfig,
    TrainConfig,
)
from tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def tokenize_df(lazy_df: pl.LazyFrame, tokenizer: Tokenizer) -> pl.LazyFrame:
    return lazy_df.select(
        pl.concat_list(
            [
                (
                    pl.when(pl.col("result") == 1)
                    .then(pl.lit([tokenizer.win_white_id], dtype=pl.List(pl.UInt16)))
                    .when(pl.col("result") == -1)
                    .then(pl.lit([tokenizer.win_black_id], dtype=pl.List(pl.UInt16)))
                    .otherwise(pl.lit([tokenizer.draw_id], dtype=pl.List(pl.UInt16)))
                ),
                pl.col("moves")
                .str.split(" ")
                .list.eval(pl.element().replace_strict(tokenizer.move_to_id))
                .cast(pl.List(pl.UInt16)),
                pl.lit([tokenizer.eos_id], dtype=pl.List(pl.UInt16)),
            ]
        ).alias("token_ids")
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw games into parquet files.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not MOVES_FILE.exists():
        logger.error(f"Moves file not found at {MOVES_FILE}")
        return

    tokenizer = Tokenizer(MOVES_FILE)
    try:
        df = (
            pl.scan_parquet(f"{RAW_DATA_DIR}/*.parquet")
            .pipe(tokenize_df, tokenizer)
            .collect()
            .sample(fraction=1.0, shuffle=True, seed=args.seed)
        )
    except Exception as e:
        logger.error(f"Failed to process parquet files in {RAW_DATA_DIR}: {e}")
        return

    max_len = GPTConfig.block_size
    oversized_count = df.filter(pl.col("token_ids").list.len() > max_len).height

    if oversized_count > 0:
        logger.warning(
            f"Found {oversized_count} games longer than {max_len} tokens! "
            "They might be truncated during training."
        )

    bucket_sizes = TrainConfig().padding_bucket_sizes
    token_counts = df.select(pl.col("token_ids").list.len().alias("len"))
    total_games = df.height

    cumulative = 0
    for bucket in bucket_sizes:
        count = token_counts.filter(pl.col("len") <= bucket).height - cumulative
        cumulative += count
        pct = count / total_games * 100
        logger.info(f"  <= {bucket:4d} tokens: {count:6d} games ({pct:5.1f}%)")
    remaining = total_games - cumulative
    if remaining > 0:
        pct = remaining / total_games * 100
        logger.info(f"  > {bucket_sizes[-1]:4d} tokens: {remaining:6d} games ({pct:5.1f}%)")

    if df.height < 2:
        logger.error("Need at least 2 games to build train/eval split.")
        return

    eval_size = max(1, int(df.height * 0.01))
    eval_size = min(eval_size, df.height - 1)

    eval_df = df.head(eval_size)
    train_df = df.slice(eval_size)

    PRETRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(PRETRAIN_DATASET_PATH)
    eval_df.write_parquet(EVAL_DATASET_PATH)
    logger.info(
        "Successfully processed %s games -> %s (train: %s, eval: %s)",
        df.height,
        PRETRAIN_DATASET_PATH.parent,
        train_df.height,
        eval_df.height,
    )


if __name__ == "__main__":
    main()
