import polars as pl
import logging
from pathlib import Path
from tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/parquet")
MOVES_FILE = Path("data/all_uci_moves.txt")
OUTPUT_FILE = Path("data/parquet/tokenized_games.parquet")


def tokenize_df(lazy_df: pl.LazyFrame, tokenizer: Tokenizer) -> pl.LazyFrame:
    sos_id = tokenizer.sos_id
    eos_id = tokenizer.eos_id

    return lazy_df.select(
        pl.concat_list(
            [
                pl.lit([sos_id], dtype=pl.List(pl.UInt16)),
                pl.col("moves")
                .str.split(" ")
                .list.eval(pl.element().replace_strict(tokenizer.move_to_id))
                .cast(pl.List(pl.UInt16)),
                pl.lit([eos_id], dtype=pl.List(pl.UInt16)),
            ]
        ).alias("token_ids")
    )


def main():
    if not MOVES_FILE.exists():
        logger.error(f"Moves file not found at {MOVES_FILE}")
        return

    tokenizer = Tokenizer(MOVES_FILE)

    df = pl.scan_parquet(f"{DATA_DIR}/*.parquet").pipe(tokenize_df, tokenizer).collect()

    oversized_count = df.filter(pl.col("token_ids").list.len() > 510).height
    if oversized_count > 0:
        logger.error(f"Found {oversized_count} games longer than 510 tokens!")

    df.write_parquet(OUTPUT_FILE)
    logger.info(f"Successfully processed {df.height} games.")


if __name__ == "__main__":
    main()
