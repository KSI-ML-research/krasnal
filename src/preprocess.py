import logging

import polars as pl

from config import DATASET_PATH, MOVES_FILE, RAW_DATA_DIR, ChessGPTConfig
from tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def tokenize_df(lazy_df: pl.LazyFrame, tokenizer: Tokenizer) -> pl.LazyFrame:
    return lazy_df.select(
        pl.concat_list(
            [
                pl.lit([tokenizer.sos_id], dtype=pl.List(pl.UInt16)),
                pl.col("moves")
                .str.split(" ")
                .list.eval(pl.element().replace_strict(tokenizer.move_to_id))
                .cast(pl.List(pl.UInt16)),
                pl.lit([tokenizer.eos_id], dtype=pl.List(pl.UInt16)),
            ]
        ).alias("token_ids")
    )


def main():
    if not MOVES_FILE.exists():
        logger.error(f"Moves file not found at {MOVES_FILE}")
        return

    tokenizer = Tokenizer(MOVES_FILE)
    try:
        df = (
            pl.scan_parquet(f"{RAW_DATA_DIR}/*.parquet")
            .pipe(tokenize_df, tokenizer)
            .collect()
            .sample(fraction=1.0, shuffle=True, seed=42)
        )
    except Exception as e:
        logger.error(f"Failed to process parquet files in {RAW_DATA_DIR}: {e}")
        return

    max_len = ChessGPTConfig.block_size
    oversized_count = df.filter(pl.col("token_ids").list.len() > max_len).height

    if oversized_count > 0:
        logger.warning(
            f"Found {oversized_count} games longer than {max_len} tokens! "
            "They might be truncated during training."
        )

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(DATASET_PATH)
    logger.info(f"Successfully processed {df.height} games -> {DATASET_PATH}")


if __name__ == "__main__":
    main()
