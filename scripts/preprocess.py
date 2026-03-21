import argparse
import json
import logging
import random
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from src.config import (
    DATA_DIR,
    DATASET_PATH,
    EVAL_DATASET_PATH,
    MOVES_FILE,
    RAW_DATA_DIR,
    RLVR_DATASET_PATH,
    SFT_DATA_PATH,
    ChessGPTConfig,
)
from src.sft_cot_generator import StockfishCoTConfig, StockfishCoTGenerator
from src.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PRETRAIN_RATIO = 0.90
SFT_POOL_RATIO = 0.025
EVAL_RATIO = 0.01
RLVR_POOL_RATIO = 1.0 - PRETRAIN_RATIO - SFT_POOL_RATIO

assert 0.0 <= EVAL_RATIO <= 1.0, "EVAL_RATIO must be in [0, 1]."
assert abs(PRETRAIN_RATIO + SFT_POOL_RATIO + RLVR_POOL_RATIO - 1.0) < 1e-12, (
    "PRETRAIN_RATIO + SFT_POOL_RATIO + RLVR_POOL_RATIO must sum to 1.0"
)

SFT_COT_RATIO = 0.80
SFT_PLAIN_RATIO = 0.20

MIN_PREFIX = 4
THINK_MIN = 8
THINK_MAX = 16
BACKTRACK_PROB = 0.15
TAIL_LEN = 0
MAX_SEQ_LEN = 256

STOCKFISH_PATH = "stockfish"
STOCKFISH_TIME = 0.02
STOCKFISH_DEPTH = 0
STOCKFISH_NODES = 0
STOCKFISH_MULTIPV = 2
STOCKFISH_THREADS = 1
STOCKFISH_HASH_MB = 128
STOCKFISH_CACHE_SIZE = 50000
PV_TEMPERATURE = 2.0
PV_EXPLORE_PROB = 0.35
BRANCH_PROB = 0.12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw games into pretrain/sft-data/sft/rlvr/eval splits."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["pretrain", "sft-data", "sft", "rlvr", "eval"],
        default=["pretrain", "sft-data", "sft", "rlvr", "eval"],
        help="Which steps to save. Default: all. Note: 'sft' requires 'sft-data'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stockfish-path", type=str, default=STOCKFISH_PATH)
    parser.add_argument("--stockfish-time", type=float, default=STOCKFISH_TIME)
    parser.add_argument("--stockfish-depth", type=int, default=STOCKFISH_DEPTH)
    parser.add_argument("--stockfish-nodes", type=int, default=STOCKFISH_NODES)
    parser.add_argument("--multipv", type=int, default=STOCKFISH_MULTIPV)
    parser.add_argument("--stockfish-threads", type=int, default=STOCKFISH_THREADS)
    parser.add_argument("--stockfish-hash-mb", type=int, default=STOCKFISH_HASH_MB)
    parser.add_argument("--stockfish-cache-size", type=int, default=STOCKFISH_CACHE_SIZE)
    parser.add_argument("--pv-temperature", type=float, default=PV_TEMPERATURE)
    parser.add_argument("--pv-explore-prob", type=float, default=PV_EXPLORE_PROB)
    parser.add_argument("--branch-prob", type=float, default=BRANCH_PROB)
    parser.add_argument("--min-prefix", type=int, default=MIN_PREFIX)
    parser.add_argument("--think-min", type=int, default=THINK_MIN)
    parser.add_argument("--think-max", type=int, default=THINK_MAX)
    parser.add_argument("--tail-len", type=int, default=TAIL_LEN)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--backtrack-prob", type=float, default=BACKTRACK_PROB)
    return parser.parse_args()


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


def _sft_mix_path() -> Path:
    return DATA_DIR / "processed" / f"sft_mix_{SFT_COT_RATIO:.2f}.parquet"


def _dataset_meta_path(dataset_path: Path) -> Path:
    return Path(f"{dataset_path}.meta.json")


def _write_dataset_metadata(dataset_path: Path, tokenizer: Tokenizer, seed: int) -> Path:
    meta_path = _dataset_meta_path(dataset_path)
    payload = {
        "metadata_format": 1,
        "dataset_path": str(dataset_path),
        "tokenizer_hash": tokenizer.mapping_hash(),
        "vocab_size": tokenizer.get_vocab_size(),
        "seed": int(seed),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
    return meta_path


def main():
    args = parse_args()
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

    if df.height == 0:
        logger.error("No games found after preprocessing.")
        return

    max_len = ChessGPTConfig.block_size
    oversized_count = df.filter(pl.col("token_ids").list.len() > max_len).height
    if oversized_count > 0:
        logger.warning(
            f"Found {oversized_count} games longer than {max_len} tokens! "
            "They might be truncated during training."
        )

    total = df.height
    eval_count = int(total * EVAL_RATIO)
    eval_count = min(eval_count, total)
    train_pool_count = total - eval_count

    pretrain_count = int(train_pool_count * PRETRAIN_RATIO)
    sft_pool_count = int(train_pool_count * SFT_POOL_RATIO)
    pretrain_count = min(pretrain_count, train_pool_count)
    sft_pool_count = min(sft_pool_count, train_pool_count - pretrain_count)
    rlvr_count = train_pool_count - pretrain_count - sft_pool_count

    assert pretrain_count + sft_pool_count + rlvr_count == train_pool_count, (
        "Train pool split must sum to train_pool_count"
    )
    assert pretrain_count + sft_pool_count + rlvr_count + eval_count == total, (
        "All splits must sum to total"
    )

    pretrain_df = df.slice(0, pretrain_count)
    sft_pool_df = df.slice(pretrain_count, sft_pool_count)
    rlvr_df = df.slice(pretrain_count + sft_pool_count, rlvr_count)
    eval_df = df.slice(train_pool_count, eval_count)

    if "pretrain" in args.steps:
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        pretrain_df.write_parquet(DATASET_PATH)
        _write_dataset_metadata(DATASET_PATH, tokenizer, args.seed)
        logger.info(f"Saved {pretrain_df.height} pretrain games -> {DATASET_PATH}")

    if "sft-data" in args.steps or "sft" in args.steps:
        SFT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        sft_pool_df.write_parquet(SFT_DATA_PATH)
        _write_dataset_metadata(SFT_DATA_PATH, tokenizer, args.seed)
        logger.info(f"Saved {sft_pool_df.height} sft-data games -> {SFT_DATA_PATH}")

    if "rlvr" in args.steps:
        RLVR_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        rlvr_df.write_parquet(RLVR_DATASET_PATH)
        _write_dataset_metadata(RLVR_DATASET_PATH, tokenizer, args.seed)
        logger.info(f"Saved {rlvr_df.height} GRPO games -> {RLVR_DATASET_PATH}")

    if "eval" in args.steps:
        EVAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        eval_df.write_parquet(EVAL_DATASET_PATH)
        _write_dataset_metadata(EVAL_DATASET_PATH, tokenizer, args.seed)
        logger.info(f"Saved {eval_df.height} eval games -> {EVAL_DATASET_PATH}")

    if "sft" not in args.steps:
        return

    plain_series = sft_pool_df.get_column("token_ids")
    plain_count = plain_series.len()
    if plain_count == 0:
        logger.error("No SFT plain pool available to build SFT mix.")
        return

    pretrain_series = pretrain_df.get_column("token_ids")
    rlvr_series = rlvr_df.get_column("token_ids")
    pretrain_count = pretrain_series.len()
    rlvr_count = rlvr_series.len()
    cot_source_total = pretrain_count + rlvr_count

    def sample_cot_source_sequence() -> list[int]:
        if cot_source_total > 0:
            idx = rng.randrange(cot_source_total)
            seq = (
                pretrain_series[idx] if idx < pretrain_count else rlvr_series[idx - pretrain_count]
            )
            return [int(t) for t in seq]

        fallback_idx = rng.randrange(plain_count)
        return [int(t) for t in plain_series[fallback_idx]]

    rng = random.Random(args.seed)
    cot_multiplier = SFT_COT_RATIO / SFT_PLAIN_RATIO
    cot_target = int(plain_count * cot_multiplier)

    cot_cfg = StockfishCoTConfig(
        min_prefix=args.min_prefix,
        think_min=args.think_min,
        think_max=args.think_max,
        tail_len=args.tail_len,
        max_seq_len=args.max_seq_len,
        backtrack_prob=args.backtrack_prob,
        stockfish_path=args.stockfish_path,
        stockfish_time=args.stockfish_time,
        stockfish_depth=args.stockfish_depth,
        stockfish_nodes=args.stockfish_nodes,
        multipv=args.multipv,
        threads=args.stockfish_threads,
        hash_mb=args.stockfish_hash_mb,
        cache_size=args.stockfish_cache_size,
        pv_temperature=args.pv_temperature,
        pv_explore_prob=args.pv_explore_prob,
        branch_prob=args.branch_prob,
    )

    cot_sequences: list[list[int]] = []
    attempts = 0
    with StockfishCoTGenerator(tokenizer, cot_cfg) as cot_generator:
        pbar = tqdm(total=cot_target, desc="Building SFT CoT samples")
        while len(cot_sequences) < cot_target:
            seq = sample_cot_source_sequence()
            tokens = cot_generator.build_sample(seq, rng)
            attempts += 1
            if tokens is None:
                continue
            cot_sequences.append(tokens)
            pbar.update(1)

        pbar.close()

    logger.info(f"Built {len(cot_sequences)} CoT samples (attempts: {attempts}).")

    all_sequences: list[list[int]] = []
    all_sequences.extend([[int(t) for t in seq[: args.max_seq_len]] for seq in plain_series])
    all_sequences.extend(cot_sequences)
    rng.shuffle(all_sequences)

    sft_mix_path = _sft_mix_path()
    sft_mix_path.parent.mkdir(parents=True, exist_ok=True)
    sft_df = pl.DataFrame({"token_ids": all_sequences}).with_columns(
        pl.col("token_ids").cast(pl.List(pl.UInt16))
    )
    sft_df.write_parquet(sft_mix_path)
    _write_dataset_metadata(sft_mix_path, tokenizer, args.seed)
    logger.info(
        "Saved %d SFT mix samples (cot=%d plain=%d) -> %s",
        len(all_sequences),
        len(cot_sequences),
        plain_count,
        sft_mix_path,
    )


if __name__ == "__main__":
    main()
