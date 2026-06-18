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
    build_what_is_on_baseline_counts,
    elo_distribution_pcts,
    log_preprocess_to_wandb,
    merge_elo_rating_counts,
    merge_seq_len_raw,
    merge_token_mix_raw,
    process_one_shard,
    run_clock_report,
    seq_len_stats_from_counts,
    token_mix_from_raw_sums,
)
from krasnal.preprocess.eval_sampling import EVAL_MONTH
from krasnal.preprocess.pack import write_packed_dataset_manifest
from krasnal.tokens import ELO_TOKENS, TC_TOKENS, load_move_vocab


def _cap_train_files(
    parquet_files: list[Path],
    target_games: int | None,
    temp_dir: Path,
) -> list[Path]:
    """Limit training input by materializing at most one partial shard."""
    if target_games is None:
        return parquet_files
    if target_games <= 0:
        raise ValueError(f"target_games must be > 0 or null, got {target_games}")

    remaining = target_games
    selected: list[Path] = []
    for path in parquet_files:
        is_eval = EVAL_MONTH in path.name
        if is_eval:
            selected.append(path)
            continue
        if remaining <= 0:
            continue

        row_count = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        if row_count <= remaining:
            selected.append(path)
            remaining -= row_count
        else:
            capped_path = temp_dir / f"limited_{path.name}"
            pl.scan_parquet(path).head(remaining).sink_parquet(capped_path)
            selected.append(capped_path)
            remaining = 0

    if not any(EVAL_MONTH not in path.name for path in selected):
        raise RuntimeError("No training shards selected for preprocessing.")
    if remaining > 0:
        available = target_games - remaining
        logger.warning(
            "target_games={} exceeds available filtered train games {}; using all available games",
            target_games,
            available,
        )
    return selected


def _init_preprocess_worker() -> None:
    faulthandler.enable()


@hydra.main(version_base=None, config_path="../../config", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    piece_aware_moves = bool(cfg.get("piece_aware_moves", False))
    side_prefixed_moves = bool(cfg.get("side_prefixed_moves", True))
    include_elo = bool(cfg.get("include_elo", True))
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

    time_control_token_cfg = cfg.get("time_control_token", {})
    time_control_token_enabled = bool(time_control_token_cfg.get("enabled", True))
    opponent_material_cfg = cfg.get("opponent_material", {})
    opponent_material_enabled = bool(opponent_material_cfg.get("enabled", False))
    outcome_conditioning_cfg = cfg.get("outcome_conditioning", {})
    outcome_conditioning_enabled = bool(outcome_conditioning_cfg.get("enabled", False))
    target_games_raw = cfg.get("target_games")
    target_games = int(target_games_raw) if target_games_raw is not None else None
    report_cfg = cfg.get("report", {})
    report_enabled = bool(report_cfg.get("enabled", False))

    pp_cfg = PreprocessConfig(
        seed=seed,
        block_size=block_size,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
        include_elo=include_elo,
        include_check_qa=include_check_qa,
        check_qa_prob=check_qa_prob,
        include_what_is_on_qa=include_what_is_on_qa,
        what_is_on_prob=what_is_on_prob,
        time_control_token_enabled=time_control_token_enabled,
        opponent_material_enabled=opponent_material_enabled,
        outcome_conditioning_enabled=outcome_conditioning_enabled,
        move_vocab_path=MOVE_VOCAB_PATH,
    )

    parquet_files = sorted(RAW_UCI_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Aix-filtered games found in {RAW_UCI_DIR}")

    temp_dir = PRETRAIN_DATASET_PATH.parent / "temp_preprocess"
    if target_games is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        parquet_files = _cap_train_files(parquet_files, target_games, temp_dir)
        logger.info("Using target_games={} for training shards", target_games)

    PRETRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MOVE_VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"Move vocabulary not found at {MOVE_VOCAB_PATH}. "
            "Run `just download-games` first (builds vocab via DuckDB over filtered parquet)."
        )
    load_move_vocab(
        MOVE_VOCAB_PATH,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )

    eval_staging = PRETRAIN_DATASET_PATH.parent / "eval_staging"
    shutil.rmtree(PRETRAIN_DATASET_PATH, ignore_errors=True)
    shutil.rmtree(eval_staging, ignore_errors=True)
    PRETRAIN_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    eval_staging.mkdir(parents=True, exist_ok=True)

    total_games = 0
    invalid_clock_skips = 0
    mix_raw: dict[str, int] | None = None
    seq_len_raw: dict[int, int] | None = None
    train_elo_counts: dict[str, int] | None = None
    eval_elo_counts: dict[str, int] | None = None
    train_shards: list[tuple[str, int]] = []
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
            eval_output = eval_staging / f"eval_{idx:04d}.parquet"
            logger.info(
                "Submitting shard {}/{}: {} -> {}",
                idx + 1,
                len(parquet_files),
                parquet_path.name,
                eval_output.name if is_eval else PRETRAIN_DATASET_PATH.name,
            )
            future = executor.submit(
                process_one_shard,
                parquet_path,
                pp_cfg,
                is_eval=is_eval,
                file_idx=idx,
                train_output_dir=PRETRAIN_DATASET_PATH,
                eval_output_path=eval_output,
                pack_flush_windows=pack_flush_windows,
                batch_size=stream_batch_size,
                collect_stats=report_enabled,
            )
            futures[future] = parquet_path.name

        for future in as_completed(futures):
            parquet_name = futures[future]
            try:
                (
                    done_name,
                    count,
                    shard_invalid_clock_skips,
                    shard_train_shards,
                    eval_path,
                    shard_mix,
                    shard_lengths,
                    shard_elo_counts,
                    output_rows,
                ) = future.result()
                total_games += count
                invalid_clock_skips += shard_invalid_clock_skips
                if report_enabled:
                    mix_raw = merge_token_mix_raw(mix_raw, shard_mix)
                    seq_len_raw = merge_seq_len_raw(seq_len_raw, shard_lengths)
                    if shard_elo_counts is not None:
                        if eval_path is not None:
                            eval_elo_counts = merge_elo_rating_counts(
                                eval_elo_counts,
                                shard_elo_counts,
                            )
                        else:
                            train_elo_counts = merge_elo_rating_counts(
                                train_elo_counts,
                                shard_elo_counts,
                            )
                if eval_path is not None:
                    eval_parts.append(Path(eval_path))
                elif shard_train_shards:
                    train_shards.extend(shard_train_shards)
                logger.info(
                    "Processed {}: {} games -> {} rows (skipped invalid clock: {})",
                    done_name,
                    count,
                    output_rows,
                    shard_invalid_clock_skips,
                )
            except Exception as e:
                logger.error("Failed to process {}: {}", parquet_name, e)
                raise

    logger.info("Skipped {} games due to invalid clock data", invalid_clock_skips)

    over_block_size_count = 0
    stats: dict = {"total": total_games}
    token_mix: dict = {}
    if report_enabled:
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
        if mix_raw is None:
            raise RuntimeError("Token mix aggregation failed")
        token_mix = token_mix_from_raw_sums(mix_raw)
        logger.info(
            "Token distribution over original tokenized train games before packing/restarts:"
        )
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
        if train_elo_counts and sum(train_elo_counts.values()) > 0:
            train_elo_pcts = elo_distribution_pcts(train_elo_counts)
            for bucket_name in ELO_TOKENS:
                if train_elo_counts[bucket_name] > 0:
                    logger.info("  {}: {:.2f}%", bucket_name, train_elo_pcts[bucket_name])
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

    window_size = block_size + 1
    train_window_rows = write_packed_dataset_manifest(
        PRETRAIN_DATASET_PATH,
        train_shards,
        window_size,
    )
    shutil.rmtree(eval_staging, ignore_errors=True)
    if target_games is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    eval_rows = pl.scan_parquet(EVAL_DATASET_PATH).select(pl.len()).collect().item()
    if train_window_rows == 0:
        raise RuntimeError("Train dataset is empty. Increase input data or reduce block_size.")

    if report_enabled and build_what_is_on_baseline:
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

    if report_enabled:
        if eval_rows > 0 and eval_elo_counts:
            logger.info("Eval Dataset Elo Bin Distribution ({} games):", eval_rows)
            for bucket_name, count in eval_elo_counts.items():
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
            train_elo_counts=train_elo_counts,
            eval_elo_bins={k: v for k, v in (eval_elo_counts or {}).items() if v > 0}
            if eval_rows > 0
            else {},
        )


if __name__ == "__main__":
    main()
