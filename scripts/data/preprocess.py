"""Hydra CLI entrypoint for the preprocessing pipeline.

All logic lives in ``krasnal.preprocess``; this script is orchestration only.
"""

import json
import multiprocessing
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from krasnal.config import (
    EVAL_DATASET_PATH,
    MOVE_VOCAB_PATH,
    PRETRAIN_DATASET_PATH,
    RAW_UCI_DIR,
)
from krasnal.preprocess import (
    PackedWindowBuilder,
    PreprocessConfig,
    build_move_vocab_from_corpus,
    log_preprocess_to_wandb,
    merge_token_mix_raw,
    one_row_one_game,
    process_one_shard,
    run_clock_report,
    seq_len_stats,
    token_mix_from_raw_sums,
)
from krasnal.preprocess.stats import _token_mix_raw_sums
from krasnal.tokens import ELO_TOKENS, TC_TOKENS, load_move_vocab


def _chunk_paths(paths: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size < 1:
        raise ValueError(f"preprocess_concat_batch_size must be >= 1, got {chunk_size}")
    return [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]


@hydra.main(version_base=None, config_path="../../config", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    piece_aware_moves = bool(cfg.get("piece_aware_moves", False))
    side_prefixed_moves = bool(cfg.get("side_prefixed_moves", True))
    block_size = int(cfg.block_size)
    seed = int(cfg.seed)

    qa = cfg.get("qa", {})

    check_cfg = qa.get("check", {})
    include_check_qa = bool(check_cfg.get("enabled", True))
    check_qa_prob = float(check_cfg.get("prob", 0.5))
    if not 0.0 <= check_qa_prob <= 1.0:
        raise ValueError(f"qa.check.prob must be in [0, 1], got {check_qa_prob}")

    what_is_on_cfg = qa.get("what_is_on", {})
    include_what_is_on_qa = bool(what_is_on_cfg.get("enabled", False))
    what_is_on_prob = float(what_is_on_cfg.get("prob", 0.0))
    if not 0.0 <= what_is_on_prob <= 1.0:
        raise ValueError(f"qa.what_is_on.prob must be in [0, 1], got {what_is_on_prob}")

    time_control_cfg = cfg.get("time_control", {})
    time_control_enabled = bool(time_control_cfg.get("enabled", True))

    pp_cfg = PreprocessConfig(
        seed=seed,
        block_size=block_size,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
        include_check_qa=include_check_qa,
        check_qa_prob=check_qa_prob,
        include_what_is_on_qa=include_what_is_on_qa,
        what_is_on_prob=what_is_on_prob,
        time_control_enabled=time_control_enabled,
        move_vocab_path=MOVE_VOCAB_PATH,
    )

    parquet_files = sorted(RAW_UCI_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Aix-filtered games found in {RAW_UCI_DIR}")

    PRETRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_move_vocab_from_corpus(
        parquet_files,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
        output_path=MOVE_VOCAB_PATH,
    )
    load_move_vocab(
        MOVE_VOCAB_PATH,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )

    temp_dir = PRETRAIN_DATASET_PATH.parent / "temp_preprocess"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total_games = 0
    max_workers = int(cfg.preprocess_workers)
    logger.info("Processing {} shards with {} workers", len(parquet_files), max_workers)

    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {}
        for idx, parquet_path in enumerate(parquet_files):
            output_path = temp_dir / f"part_{idx:04d}.parquet"
            future = executor.submit(
                process_one_shard,
                parquet_path,
                output_path,
                pp_cfg,
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
                raise

    all_parts = sorted(temp_dir.glob("part_*.parquet"))
    if not all_parts:
        raise RuntimeError("No data generated")

    concat_batch_size = max(1, int(cfg.get("preprocess_concat_batch_size", 10)))
    eval_batches_dir = temp_dir / "eval_batches"
    packed_batches_dir = temp_dir / "packed_batches"
    shutil.rmtree(eval_batches_dir, ignore_errors=True)
    shutil.rmtree(packed_batches_dir, ignore_errors=True)
    eval_batches_dir.mkdir(parents=True)

    pack_flush_windows = max(1, int(cfg.get("pack_flush_windows", 8_000)))
    packed_builder = PackedWindowBuilder(block_size, flush_every=pack_flush_windows)

    max_tokens = block_size
    len_chunks: list[pl.DataFrame] = []
    mix_raw: dict[str, int] | None = None

    batch_jobs = list(enumerate(_chunk_paths(all_parts, concat_batch_size)))
    random.Random(seed).shuffle(batch_jobs)

    for _batch_idx, batch_paths in batch_jobs:
        for part_path in batch_paths:
            part_lf = pl.scan_parquet(part_path)
            len_chunks.append(
                part_lf.select(pl.col("token_ids").list.len().alias("len"))
                .filter(pl.col("len") <= max_tokens)
                .collect()
            )
            filtered_lf = part_lf.filter(pl.col("token_ids").list.len() <= max_tokens)
            mix_raw = merge_token_mix_raw(
                mix_raw,
                _token_mix_raw_sums(filtered_lf.select("token_ids")),
            )
            train_lf = one_row_one_game(
                filtered_lf.filter(pl.col("split_bucket") != 0).select(
                    "token_ids",
                    "active_clock_ids",
                    "opponent_clock_ids",
                ),
                block_size=block_size,
            )
            eval_lf = one_row_one_game(
                filtered_lf.filter(pl.col("split_bucket") == 0).select(
                    "token_ids",
                    "active_clock_ids",
                    "opponent_clock_ids",
                ),
                block_size=block_size,
            )
            eval_lf.sink_parquet(eval_batches_dir / f"{part_path.stem}.parquet")
            train_df = train_lf.collect()
            for row in train_df.iter_rows(named=True):
                packed_builder.feed_game(PackedWindowBuilder._parse_game_row(row))
            packed_builder.maybe_flush(packed_batches_dir)

        packed_builder.drain()
        packed_builder.maybe_flush(packed_batches_dir)

    seq_lens = pl.concat(len_chunks, how="vertical")
    stats = seq_len_stats(seq_lens.lazy(), block_size)
    if stats["total"] == 0:
        raise RuntimeError("No games found in raw dataset.")

    logger.info(
        "Sequence length stats: min={}, max={}, mean={:.1f}, p95={}, p99={}, p999={}",
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["p95"],
        stats["p99"],
        stats["p999"],
    )

    over_block_size_count = stats.get("over_block_size", 0)
    total_count = stats["total"]
    pct_long = over_block_size_count / total_count * 100
    logger.info(
        "Filtering games with >{} tokens done: removed {} games ({:.2f}%)",
        max_tokens,
        over_block_size_count,
        pct_long,
    )

    if mix_raw is None:
        raise RuntimeError("Token mix aggregation failed")
    token_mix = token_mix_from_raw_sums(mix_raw)
    logger.info("Token distribution:")
    logger.info("  total: 100.00% ({})", token_mix["total_tokens"])
    logger.info(
        "  moves: {:.2f}% ({})",
        token_mix["uci_move_pct"],
        token_mix["uci_move_count"],
    )
    logger.info(
        "  qa_is_check: {:.2f}% ({})",
        token_mix["check_qa_pct"],
        token_mix["check_qa_count"],
    )
    logger.info(
        "  qa_whats_on_prompt: {:.2f}% ({})",
        token_mix["what_is_on_pct"],
        token_mix["what_is_on_count"],
    )
    logger.info(
        "  qa_whats_on_answer_empty: {:.2f}% ({})",
        token_mix["empty_pct"],
        token_mix["empty_count"],
    )
    logger.info(
        "  qa_whats_on_answer_piece: {:.2f}% ({})",
        token_mix["piece_answer_pct"],
        token_mix["piece_answer_count"],
    )
    logger.info(
        "  conditioning_prefix: {:.2f}% ({})",
        token_mix["outcome_prefix_pct"],
        token_mix["outcome_prefix_count"],
    )
    logger.info(
        "  game_start: {:.2f}% ({})",
        token_mix["game_start_pct"],
        token_mix["game_start_count"],
    )
    logger.info(
        "  game_end: {:.2f}% ({})",
        token_mix["game_end_pct"],
        token_mix["game_end_count"],
    )

    logger.info("ELO Bucket Distribution:")
    total_elo = sum(token_mix[f"elo_{b}_count"] for b in ELO_TOKENS)
    if total_elo > 0:
        for bucket_name in ELO_TOKENS:
            count = token_mix[f"elo_{bucket_name}_count"]
            pct = (count / total_elo) * 100.0
            logger.info("  {}: {:.2f}%", bucket_name, pct)

    logger.info("Time Control Bucket Distribution:")
    total_tc = sum(token_mix[f"tc_{b}_count"] for b in TC_TOKENS)
    if total_tc > 0:
        for bucket_name in TC_TOKENS:
            count = token_mix[f"tc_{bucket_name}_count"]
            pct = (count / total_tc) * 100.0
            logger.info("  {}: {:.2f}%", bucket_name, pct)

    token_mix_path = PRETRAIN_DATASET_PATH.parent / "token_mix_stats.json"
    with token_mix_path.open("w") as f:
        json.dump(token_mix, f, indent=2, sort_keys=True)
        f.write("\n")

    eval_parts = sorted(eval_batches_dir.glob("*.parquet"))
    pl.concat(pl.scan_parquet(p) for p in eval_parts).sink_parquet(EVAL_DATASET_PATH)
    PRETRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    packed_builder.finish(PRETRAIN_DATASET_PATH, part_dir=packed_batches_dir)

    shutil.rmtree(temp_dir)

    train_window_rows = pl.scan_parquet(PRETRAIN_DATASET_PATH).select(pl.len()).collect().item()
    eval_rows = pl.scan_parquet(EVAL_DATASET_PATH).select(pl.len()).collect().item()
    if train_window_rows == 0:
        raise RuntimeError("Train dataset is empty. Increase input data or reduce block_size.")

    logger.info(
        "Processed {} games -> {} (train windows: {}, eval games: {})",
        stats["total"],
        PRETRAIN_DATASET_PATH.parent,
        train_window_rows,
        eval_rows,
    )

    run_clock_report(EVAL_DATASET_PATH)

    log_preprocess_to_wandb(
        cfg=cfg,
        token_mix=token_mix,
        seq_stats=stats,
        total_games=stats["total"],
        train_window_rows=train_window_rows,
        eval_rows=eval_rows,
        over_block_size_count=over_block_size_count,
    )


if __name__ == "__main__":
    main()
