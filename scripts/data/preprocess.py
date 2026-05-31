"""Hydra CLI entrypoint for the preprocessing pipeline.

All logic lives in ``krasnal.preprocess``; this script is orchestration only.
"""

import faulthandler
import json
import multiprocessing
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
    WHAT_IS_ON_BASELINE_COUNTS_PATH,
)
from krasnal.preprocess import (
    PreprocessConfig,
    build_move_vocab_from_corpus,
    build_what_is_on_baseline_counts,
    log_preprocess_to_wandb,
    merge_seq_len_raw,
    merge_token_mix_raw,
    process_one_shard,
    run_clock_report,
    seq_len_stats_from_counts,
    token_mix_from_raw_sums,
)
from krasnal.tokens import ELO_TOKENS, TC_TOKENS, load_move_vocab

EVAL_MONTH = "2019-12"
EVAL_GAMES_PER_BIN = 10_000
EVAL_MIN_CLOCK = 30


def _init_preprocess_worker() -> None:
    faulthandler.enable()


def _merge_packed_datasets(part_dirs: list[Path], output_path: Path) -> int:
    if output_path.is_dir():
        shutil.rmtree(output_path)
    elif output_path.exists():
        output_path.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    columns = None
    window_size = None
    merged_shards = []
    total_rows = 0
    shard_idx = 0
    for part_dir in part_dirs:
        with (part_dir / "metadata.json").open() as f:
            metadata = json.load(f)
        if metadata.get("format") != "krasnal-packed-npy":
            raise ValueError(f"Unsupported packed dataset format in {part_dir}")
        part_columns = metadata["columns"]
        part_window_size = int(metadata["window_size"])
        if columns is None:
            columns = part_columns
            window_size = part_window_size
        elif columns != part_columns or window_size != part_window_size:
            raise ValueError(f"Packed dataset metadata mismatch in {part_dir}")

        for shard in metadata["shards"]:
            rows = int(shard["rows"])
            dst_path = output_path / f"part_{shard_idx:04d}"
            (part_dir / shard["path"]).rename(dst_path)
            merged_shards.append({"path": dst_path.name, "rows": rows})
            total_rows += rows
            shard_idx += 1

    if columns is None or window_size is None:
        columns = ["token_ids", "active_clock_ids", "opponent_clock_ids"]
        window_size = 0

    with (output_path / "metadata.json").open("w") as f:
        json.dump(
            {
                "format": "krasnal-packed-npy",
                "version": 1,
                "window_size": window_size,
                "rows": total_rows,
                "columns": columns,
                "shards": merged_shards,
            },
            f,
            indent=2,
        )
        f.write("\n")
    return total_rows


def _sample_eval_games(raw_path: Path, output_path: Path, seed: int) -> None:
    """Maia-style balanced same-bin sampling with time-pressure filtering."""
    df = pl.read_parquet(raw_path)

    # Time-pressure filter: exclude games where any clock value < EVAL_MIN_CLOCK
    df = df.filter(
        (pl.col("clocks_white").list.len() > 0)
        & (pl.col("clocks_black").list.len() > 0)
        & pl.col("clocks_white").list.eval(pl.element().ge(EVAL_MIN_CLOCK)).list.all()
        & pl.col("clocks_black").list.eval(pl.element().ge(EVAL_MIN_CLOCK)).list.all()
    )

    # Same 100-pt Elo bin (≥1500) for both players
    df = df.with_columns(
        (pl.col("white_rating") // 100 * 100).alias("white_bin"),
        (pl.col("black_rating") // 100 * 100).alias("black_bin"),
    ).filter((pl.col("white_bin") == pl.col("black_bin")) & (pl.col("white_bin") >= 1500))

    # Shuffle to ensure unbiased selection across the entire month
    if len(df) > 0:
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    # Sample up to EVAL_GAMES_PER_BIN per bin
    sampled = (
        df.with_columns(pl.int_range(1, pl.len() + 1).over("white_bin").alias("_row_within_bin"))
        .filter(pl.col("_row_within_bin") <= EVAL_GAMES_PER_BIN)
        .drop("white_bin", "black_bin", "_row_within_bin")
    )

    logger.info(
        "Eval sampling: {} -> {} games after time-pressure + same-bin filter",
        len(df),
        len(sampled),
    )
    sampled.write_parquet(output_path)


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
    baseline_cfg = what_is_on_cfg.get("baseline", {})
    build_what_is_on_baseline = bool(baseline_cfg.get("enabled", True))
    what_is_on_baseline_max_games = int(baseline_cfg.get("max_games", 100_000))

    time_control_cfg = cfg.get("time_control", {})
    time_control_enabled = bool(time_control_cfg.get("enabled", True))
    outcome_conditioning_cfg = cfg.get("outcome_conditioning", {})
    outcome_conditioning_enabled = bool(outcome_conditioning_cfg.get("enabled", True))

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
        outcome_conditioning_enabled=outcome_conditioning_enabled,
        move_vocab_path=MOVE_VOCAB_PATH,
    )

    parquet_files = sorted(RAW_UCI_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Aix-filtered games found in {RAW_UCI_DIR}")

    # Maia-style eval sampling: if 2019-12 is present, sample and replace
    temp_dir = PRETRAIN_DATASET_PATH.parent / "temp_preprocess"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    eval_raw = [p for p in parquet_files if EVAL_MONTH in p.name]
    if eval_raw:
        eval_sampled_path = temp_dir / f"eval_sampled_{EVAL_MONTH}.parquet"
        _sample_eval_games(eval_raw[0], eval_sampled_path, seed=seed)
        parquet_files = [eval_sampled_path if EVAL_MONTH in p.name else p for p in parquet_files]

    # --- Build move vocab (skips eval files internally) ---
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

    # --- Parallel tokenization + train packing ---
    total_games = 0
    invalid_clock_skips = 0
    mix_raw: dict[str, int] | None = None
    seq_len_raw: dict[int, int] | None = None
    train_part_dirs: list[Path] = []
    eval_parts: list[Path] = []
    max_workers = int(cfg.preprocess_workers)
    pack_flush_windows = max(1, int(cfg.get("pack_flush_windows", 8_000)))
    stream_batch_size = max(1, int(cfg.get("pack_stream_batch_size", 10_000)))
    logger.info("Processing {} shards with {} workers", len(parquet_files), max_workers)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
        initializer=_init_preprocess_worker,
    ) as executor:
        futures = {}
        for idx, parquet_path in enumerate(parquet_files):
            is_eval = EVAL_MONTH in parquet_path.name
            output_path = temp_dir / (f"eval_{idx:04d}.parquet" if is_eval else f"train_{idx:04d}")
            logger.info(
                "Submitting shard {}/{}: {} -> {}",
                idx + 1,
                len(parquet_files),
                parquet_path.name,
                output_path.name,
            )
            future = executor.submit(
                process_one_shard,
                parquet_path,
                output_path,
                pp_cfg,
                is_eval=is_eval,
                pack_flush_windows=pack_flush_windows,
                batch_size=stream_batch_size,
            )
            futures[future] = parquet_path.name

        for future in as_completed(futures):
            parquet_name = futures[future]
            try:
                (
                    done_name,
                    count,
                    shard_invalid_clock_skips,
                    output_name,
                    shard_mix,
                    shard_lengths,
                    output_rows,
                ) = future.result()
                total_games += count
                invalid_clock_skips += shard_invalid_clock_skips
                mix_raw = merge_token_mix_raw(mix_raw, shard_mix)
                seq_len_raw = merge_seq_len_raw(seq_len_raw, shard_lengths)
                output_path = Path(output_name)
                if output_path.suffix == ".parquet":
                    eval_parts.append(output_path)
                else:
                    train_part_dirs.append(output_path)
                logger.info(
                    "Processed {}: {} games -> {} rows in {} (skipped invalid clock: {})",
                    done_name,
                    count,
                    output_rows,
                    output_name,
                    shard_invalid_clock_skips,
                )
            except Exception as e:
                logger.error("Failed to process {}: {}", parquet_name, e)
                raise

    logger.info("Skipped {} games due to invalid clock data", invalid_clock_skips)

    # --- Sequence length stats ---
    stats = seq_len_stats_from_counts(seq_len_raw or {}, block_size)
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
        "Filtering games longer than context (>{} tokens): "
        "removed {} games ({:.2f}% of input games)",
        block_size,
        over_block_size_count,
        pct_long,
    )

    # --- Token mix report ---
    if mix_raw is None:
        raise RuntimeError("Token mix aggregation failed")
    token_mix = token_mix_from_raw_sums(mix_raw)
    logger.info("Token distribution over original tokenized train games before packing/restarts:")
    logger.info("  total_input_tokens: 100.00% ({})", token_mix["total_tokens"])
    for label, pct_key, count_key in [
        ("moves", "uci_move_pct", "uci_move_count"),
        ("qa_is_check", "check_qa_pct", "check_qa_count"),
        ("qa_whats_on_prompt", "what_is_on_pct", "what_is_on_count"),
        ("qa_whats_on_answer_empty", "empty_pct", "empty_count"),
        ("qa_whats_on_answer_piece", "piece_answer_pct", "piece_answer_count"),
        ("conditioning_prefix", "outcome_prefix_pct", "outcome_prefix_count"),
        ("opponent_material", "material_pct", "material_count"),
        ("game_start", "game_start_pct", "game_start_count"),
        ("game_end", "game_end_pct", "game_end_count"),
    ]:
        logger.info(
            "  {}: {:.2f}% of input tokens ({})",
            label,
            token_mix[pct_key],
            token_mix[count_key],
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

    # --- Finalize datasets ---
    if eval_parts:
        pl.concat(pl.scan_parquet(p) for p in sorted(eval_parts)).sink_parquet(EVAL_DATASET_PATH)
    else:
        logger.warning("No eval games found; writing empty eval dataset")
        pl.DataFrame(
            schema={
                "token_ids": pl.List(pl.UInt16),
                "active_clock_ids": pl.List(pl.UInt32),
                "opponent_clock_ids": pl.List(pl.UInt32),
            }
        ).write_parquet(EVAL_DATASET_PATH)

    PRETRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_window_rows = _merge_packed_datasets(sorted(train_part_dirs), PRETRAIN_DATASET_PATH)

    shutil.rmtree(temp_dir)

    eval_rows = pl.scan_parquet(EVAL_DATASET_PATH).select(pl.len()).collect().item()
    if train_window_rows == 0:
        raise RuntimeError("Train dataset is empty. Increase input data or reduce block_size.")

    if build_what_is_on_baseline:
        build_what_is_on_baseline_counts(
            train_dataset_path=PRETRAIN_DATASET_PATH,
            output_path=WHAT_IS_ON_BASELINE_COUNTS_PATH,
            block_size=block_size,
            max_games=what_is_on_baseline_max_games,
            seed=seed,
        )

    logger.info(
        "Processed {} games -> {} (train windows: {}, eval games: {})",
        stats["total"],
        PRETRAIN_DATASET_PATH.parent,
        train_window_rows,
        eval_rows,
    )

    # --- Eval Elo bin report ---
    if eval_rows > 0:
        eval_df = pl.read_parquet(EVAL_DATASET_PATH)
        logger.info("Eval Dataset Elo Bin Distribution ({} games):", eval_rows)
        # Elo token is at position 3 or 4 in the prefix depending on time control
        # We count elo token occurrences across all games
        from krasnal.tokens import ELO_TOKENS as ELO_MAP

        elo_id_to_name = {v: k for k, v in ELO_MAP.items()}
        elo_counts: dict[str, int] = {name: 0 for name in ELO_MAP}
        for token_ids in eval_df["token_ids"].to_list():
            for tid in token_ids:
                if tid in elo_id_to_name:
                    elo_counts[elo_id_to_name[tid]] += 1
                    break  # first elo token = white elo
        for bucket_name, count in elo_counts.items():
            if count > 0:
                logger.info("  {}: {} games", bucket_name, count)

    run_clock_report(EVAL_DATASET_PATH)

    log_preprocess_to_wandb(
        cfg=cfg,
        token_mix=token_mix,
        seq_stats=stats,
        total_games=stats["total"],
        train_window_rows=train_window_rows,
        eval_rows=eval_rows,
        over_block_size_count=over_block_size_count,
        eval_elo_bins={k: v for k, v in elo_counts.items() if v > 0} if eval_rows > 0 else {},
    )


if __name__ == "__main__":
    main()
