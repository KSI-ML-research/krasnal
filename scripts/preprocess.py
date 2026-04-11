import logging

import polars as pl

from config import (
    EVAL_DATASET_PATH,
    MOVES_FILE,
    PRETRAIN_DATASET_PATH,
    RAW_DATA_DIR,
    ChessGPTConfig,
)
from tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _result_token_expr(tokenizer: Tokenizer) -> pl.Expr:
    return (
        pl.when(pl.col("result") == 1)
        .then(pl.lit([tokenizer.win_white_id], dtype=pl.List(pl.UInt16)))
        .when(pl.col("result") == -1)
        .then(pl.lit([tokenizer.win_black_id], dtype=pl.List(pl.UInt16)))
        .otherwise(pl.lit([tokenizer.draw_id], dtype=pl.List(pl.UInt16)))
    )


def _elo_token_expr(column: str, tokenizer: Tokenizer) -> pl.Expr:
    ELO_BREAKS = [999, 1499, 1999, 2499, 2999] 
    # pl.cut creates (] intervals, so breakpoints-1 to make them [)
    labels = [
        str(tokenizer.elo_below_1000_id),
        str(tokenizer.elo_1000_1499_id),
        str(tokenizer.elo_1500_1999_id),
        str(tokenizer.elo_2000_2499_id),
        str(tokenizer.elo_2500_2999_id),
        str(tokenizer.elo_above_2999_id),
    ]
    return pl.concat_list(
        pl.col(column).cut(ELO_BREAKS, labels=labels).cast(pl.String).cast(pl.UInt16)
    )


def _moves_token_expr(tokenizer: Tokenizer) -> pl.Expr:
    return (
        pl.col("moves")
        .str.split(" ")
        .list.eval(pl.element().replace_strict(tokenizer.move_to_id))
        .cast(pl.List(pl.UInt16))
    )


def tokenize_df(lazy_df: pl.LazyFrame, tokenizer: Tokenizer) -> pl.LazyFrame:
    return lazy_df.select(
        pl.concat_list(
            [
                _result_token_expr(tokenizer),
                _elo_token_expr("white_elo", tokenizer),
                _elo_token_expr("black_elo", tokenizer),
                _moves_token_expr(tokenizer),
                pl.lit([tokenizer.eos_id], dtype=pl.List(pl.UInt16)),
            ]
        ).alias("token_ids")
    )


def main():
    if not MOVES_FILE.exists():
        logger.error(f"Moves file not found at {MOVES_FILE}")
        return

    tokenizer = Tokenizer(MOVES_FILE)

    df = (
        pl.scan_parquet(f"{RAW_DATA_DIR}/*.parquet")
        .pipe(tokenize_df, tokenizer)
        .collect()
        .sample(fraction=1.0, shuffle=True, seed=42)
    )

    max_len = ChessGPTConfig.block_size
    oversized_count = df.filter(pl.col("token_ids").list.len() > max_len).height

    if oversized_count > 0:
        logger.warning(
            f"Found {oversized_count} games longer than {max_len} tokens! "
            "They might be truncated during training."
        )

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
