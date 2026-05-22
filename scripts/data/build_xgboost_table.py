#!/usr/bin/env python3
"""Build move-level train/val/test parquet tables for XGBoost.

Input files are expected to be game-level parquet files produced by preprocess
with list columns:
  - ply_list
  - move_clocks_seconds
  - move_time_seconds
"""

from __future__ import annotations

from pathlib import Path
import re

import chess
import hydra
import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig

from krasnal.config import EVAL_DATASET_PATH, PRETRAIN_DATASET_PATH
from krasnal.inference.move_analysis import move_entropy, ply_scaling
from krasnal.lichess_clocks import fetch_lichess_pgn


MAX_TIME_OVER_CLOCK_SECONDS = 300.0
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15


def _bucket_time_initial(time_initial: float | int | None) -> str:
    if time_initial is None:
        return "time_unknown"
    if time_initial < 600:
        return "time_300_599"
    if time_initial < 900:
        return "time_600_899"
    if time_initial < 1200:
        return "time_900_1199"
    return "time_1200_plus"


def _bucket_ply_count(ply_count: float | int | None) -> str:
    if ply_count is None:
        return "ply_unknown"
    if ply_count <= 30:
        return "ply_0_30"
    if ply_count <= 60:
        return "ply_31_60"
    if ply_count <= 90:
        return "ply_61_90"
    return "ply_91_plus"


def _stable_shuffle_key(game_idx: pl.Expr, seed: int = 42) -> pl.Expr:
    # Simple deterministic pseudo-shuffle that keeps the split reproducible.
    return ((game_idx.cast(pl.Int64) * 1103515245) + seed) % 2_147_483_647


def _parse_timecontrol_from_pgn(pgn_text: str) -> tuple[float | None, float | None]:
    """Parse TimeControl PGN tag as (initial_seconds, increment_seconds)."""
    match = re.search(r'\[TimeControl\s+"(\d+)\+(\d+)"\]', pgn_text)
    if not match:
        return None, None
    try:
        return float(match.group(1)), float(match.group(2))
    except Exception:
        return None, None


def _load_games(input_paths: list[Path]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    game_offset = 0

    required_columns = {
        "ply_list",
        "move_clocks_seconds",
        "move_time_seconds",
    }

    for path in input_paths:
        if not path.exists():
            logger.warning("Skipping missing input: {}", path)
            continue

        df = pl.read_parquet(path)
        # Ensure we have a ply list; if not, try to derive from `uci_moves` or from
        # the length of `move_clocks_seconds`.
        if "ply_list" not in df.columns:
            if "uci_moves" in df.columns:
                # derive ply_list as list of indices per game from uci_moves
                ply_lists = []
                for s in df["uci_moves"]:
                    if s is None:
                        ply_lists.append([])
                    else:
                        try:
                            n = len(s.split())
                        except Exception:
                            n = 0
                        ply_lists.append(list(range(n)))
                df = df.with_columns(pl.Series("ply_list", ply_lists, dtype=pl.List(pl.Int64)))
            elif "move_clocks_seconds" in df.columns:
                # derive sequential ply indices per game from clocks length (eager Python fallback)
                ply_lists = []
                for lst in df["move_clocks_seconds"]:
                    if lst is None:
                        ply_lists.append([])
                    else:
                        try:
                            n = len(lst)
                        except Exception:
                            n = 0
                        ply_lists.append(list(range(n)))
                df = df.with_columns(pl.Series("ply_list", ply_lists, dtype=pl.List(pl.Int64)))

        # Ensure we have move_time_seconds (labels). If missing and we have
        # move_clocks_seconds plus time_initial/time_increment, compute labels.
        if "move_time_seconds" not in df.columns and "move_clocks_seconds" in df.columns:
            # compute per-row labels eagerly using Python (safer for varied input types)
            def _compute_labels_row(move_clocks, time_initial, time_increment):
                if not move_clocks:
                    return [None] * 0
                if time_initial is None or time_increment is None:
                    return [None] * len(move_clocks)
                prev = float(time_initial)
                inc = float(time_increment)
                out = []
                for cur in move_clocks:
                    if cur is None:
                        out.append(None)
                    else:
                        used = prev - float(cur) + inc
                        out.append(max(0.0, used))
                        prev = float(cur)
                return out

            labels_list = []
            # prepare python lists to avoid polars Series truth-value issues
            mc_list = df["move_clocks_seconds"].to_list()
            if "time_initial" in df.columns:
                ti_list = df["time_initial"].to_list()
            else:
                ti_list = [None] * len(mc_list)
            if "time_increment" in df.columns:
                tinc_list = df["time_increment"].to_list()
            else:
                tinc_list = [None] * len(mc_list)
            lichess_list = df["lichess_id"].to_list() if "lichess_id" in df.columns else [None] * len(mc_list)

            for mc, ti, tinc, lichess_id in zip(mc_list, ti_list, tinc_list, lichess_list):
                if (ti is None or tinc is None) and lichess_id is not None:
                    try:
                        pgn_text = fetch_lichess_pgn(str(lichess_id), timeout=5, max_retries=1)
                        parsed = _parse_timecontrol_from_pgn(pgn_text)
                        if parsed is not None:
                            if ti is None:
                                ti = parsed[0]
                            if tinc is None:
                                tinc = parsed[1]
                    except Exception:
                        pass

                try:
                    if mc is None:
                        labels = []
                    elif ti is not None and tinc is not None:
                        labels = _compute_labels_row(mc, ti, tinc)
                    else:
                        # fallback: compute per-ply time from adjacent clocks (assume increment=0)
                        labels = []
                        for j in range(len(mc) - 1):
                            a = mc[j]
                            b = mc[j + 1]
                            if a is None or b is None:
                                labels.append(None)
                            else:
                                labels.append(max(0.0, float(a) - float(b)))
                        # last ply has no next clock
                        if len(mc) > 0:
                            labels.append(None)
                        else:
                            labels = []
                except Exception:
                    labels = [None] * (len(mc) if mc else 0)
                labels_list.append(labels)

            df = df.with_columns(pl.Series("move_time_seconds", labels_list, dtype=pl.List(pl.Float64)))

            # Compute prev_clock_seconds and clock_diff_seconds as list columns per game.
            prev_clock_lists = []
            clock_diff_lists = []
            mc_list = df["move_clocks_seconds"].to_list()
            for mc in mc_list:
                if not mc:
                    prev_clock_lists.append([])
                    clock_diff_lists.append([])
                    continue
                prevs = []
                diffs = []
                for i in range(len(mc)):
                    if i >= 2:
                        prev = mc[i - 2]
                    else:
                        prev = None
                    prevs.append(prev)
                    if prev is None or mc[i] is None:
                        diffs.append(None)
                    else:
                        try:
                            diffs.append(max(0.0, float(prev) - float(mc[i])))
                        except Exception:
                            diffs.append(None)
                prev_clock_lists.append(prevs)
                clock_diff_lists.append(diffs)

            df = df.with_columns(
                pl.Series("prev_clock_seconds", prev_clock_lists, dtype=pl.List(pl.Float64)),
                pl.Series("clock_diff_seconds", clock_diff_lists, dtype=pl.List(pl.Float64)),
            )

        # Compute pre-move board-state features from move history.
        if (
            ("is_in_check_before_move" not in df.columns or "total_pieces_before_move" not in df.columns)
            and "ply_list" in df.columns
        ):
            uci_rows = df["uci_moves"].to_list() if "uci_moves" in df.columns else [None] * df.height
            ply_rows = df["ply_list"].to_list()
            check_rows: list[list[int | None]] = []
            piece_rows: list[list[int | None]] = []

            for uci_raw, ply_raw in zip(uci_rows, ply_rows):
                n = len(ply_raw) if ply_raw is not None else 0
                if n == 0:
                    check_rows.append([])
                    piece_rows.append([])
                    continue

                if uci_raw is None:
                    check_rows.append([None] * n)
                    piece_rows.append([None] * n)
                    continue

                if isinstance(uci_raw, str):
                    move_tokens = uci_raw.split()
                elif isinstance(uci_raw, list):
                    move_tokens = [str(m) for m in uci_raw]
                else:
                    check_rows.append([None] * n)
                    piece_rows.append([None] * n)
                    continue

                board = chess.Board()
                row_values: list[int | None] = []
                row_piece_values: list[int | None] = []
                try:
                    for token in move_tokens[:n]:
                        row_values.append(1 if board.is_check() else 0)
                        row_piece_values.append(len(board.piece_map()))
                        board.push_uci(token)
                except Exception:
                    check_rows.append([None] * n)
                    piece_rows.append([None] * n)
                    continue

                if len(row_values) < n:
                    row_values.extend([None] * (n - len(row_values)))
                if len(row_piece_values) < n:
                    row_piece_values.extend([None] * (n - len(row_piece_values)))
                check_rows.append(row_values)
                piece_rows.append(row_piece_values)

            new_columns = []
            if "is_in_check_before_move" not in df.columns:
                new_columns.append(
                    pl.Series("is_in_check_before_move", check_rows, dtype=pl.List(pl.Int8))
                )
            if "total_pieces_before_move" not in df.columns:
                new_columns.append(
                    pl.Series("total_pieces_before_move", piece_rows, dtype=pl.List(pl.Int16))
                )
            if new_columns:
                df = df.with_columns(*new_columns)

        missing = required_columns.difference(df.columns)
        if missing:
            logger.warning("Skipping {} (missing columns: {})", path, sorted(missing))
            continue

        df = df.with_row_index("game_idx", offset=game_offset)
        game_offset += df.height
        frames.append(df)

    if not frames:
        raise ValueError("No valid preprocess parquet inputs found")

    return pl.concat(frames, how="vertical_relaxed")


def _compute_entropy_features(games: pl.DataFrame) -> pl.DataFrame:
    if "model_move_probs" not in games.columns or "move_entropy" in games.columns:
        return games

    if "ply_list" not in games.columns:
        raise ValueError("Cannot compute entropy features without ply_list")

    probs_rows = games["model_move_probs"].to_list()
    ply_rows = games["ply_list"].to_list()

    entropy_rows: list[list[float | None]] = []
    scaled_rows: list[list[float | None]] = []
    flat_entropy: list[float] = []

    for row_idx, (probs_row, ply_row) in enumerate(zip(probs_rows, ply_rows)):
        if ply_row is None:
            entropy_rows.append([])
            scaled_rows.append([])
            continue

        if probs_row is None:
            if len(ply_row) == 0:
                entropy_rows.append([])
                scaled_rows.append([])
                continue
            raise ValueError(f"Missing model_move_probs for row {row_idx}; cannot derive entropy")

        if len(probs_row) != len(ply_row):
            raise ValueError(
                f"Row {row_idx} has {len(probs_row)} probability vectors but {len(ply_row)} plies"
            )

        row_entropy: list[float | None] = []
        row_scaled: list[float | None] = []
        for move_probs, ply in zip(probs_row, ply_row):
            if move_probs is None or len(move_probs) == 0:
                raise ValueError(
                    f"Row {row_idx} contains an empty probability vector; entropy would be fabricated"
                )

            probs_tensor = torch.tensor(move_probs, dtype=torch.float32)
            ent = move_entropy(probs_tensor)
            scaled = ent * ply_scaling(int(ply))
            row_entropy.append(ent)
            row_scaled.append(scaled)
            flat_entropy.append(ent)

        entropy_rows.append(row_entropy)
        scaled_rows.append(row_scaled)

    if flat_entropy:
        entropy_series = pl.Series(flat_entropy, dtype=pl.Float64)
        logger.info(
            "Entropy stats: rows={}, min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}",
            len(flat_entropy),
            float(entropy_series.min()),
            float(entropy_series.max()),
            float(entropy_series.mean()),
            float(entropy_series.std()),
        )

        entropy_min = float(entropy_series.min())
        entropy_max = float(entropy_series.max())
        if entropy_min == entropy_max:
            raise ValueError("move_entropy is constant across the dataset; probabilities look invalid")

        entropy_tensor = torch.tensor(flat_entropy, dtype=torch.float32)
        if torch.allclose(entropy_tensor, torch.ones_like(entropy_tensor), atol=1e-6, rtol=0.0):
            raise ValueError("move_entropy is 1.0 everywhere; model probabilities may be missing")

    return games.with_columns(
        pl.Series("move_entropy", entropy_rows, dtype=pl.List(pl.Float64)),
        pl.Series("entropy_x_ply_scaling", scaled_rows, dtype=pl.List(pl.Float64)),
    )


def _explode_to_moves(games: pl.DataFrame) -> pl.DataFrame:
    select_columns = [
        "game_idx",
        "time_initial",
        "ply_list",
        "move_clocks_seconds",
        "move_time_seconds",
        "prev_clock_seconds",
        "clock_diff_seconds",
        "is_in_check_before_move",
        "total_pieces_before_move",
    ]
    explode_columns = [
        "ply_list",
        "move_clocks_seconds",
        "move_time_seconds",
        "prev_clock_seconds",
        "clock_diff_seconds",
        "is_in_check_before_move",
        "total_pieces_before_move",
    ]
    rename_map = {
        "time_initial": "time_initial",
        "ply_list": "ply",
        "move_clocks_seconds": "clock_after_seconds",
        "move_time_seconds": "target_move_time_seconds",
        "prev_clock_seconds": "prev_clock_seconds",
        "clock_diff_seconds": "clock_diff_seconds",
        "is_in_check_before_move": "is_in_check_before_move",
        "total_pieces_before_move": "total_pieces",
    }

    if "move_entropy" in games.columns:
        select_columns.append("move_entropy")
        explode_columns.append("move_entropy")
        rename_map["move_entropy"] = "move_entropy"

    if "entropy_x_ply_scaling" in games.columns:
        select_columns.append("entropy_x_ply_scaling")
        explode_columns.append("entropy_x_ply_scaling")
        rename_map["entropy_x_ply_scaling"] = "entropy_x_ply_scaling"

    moves = (
        games.select(*select_columns)
        .explode(explode_columns)
        .rename(rename_map)
        .with_columns(
            [
                (pl.col("ply") % 2).cast(pl.Int8).alias("side_to_move"),
                pl.col("ply").cast(pl.Int32),
                pl.col("time_initial").cast(pl.Float64),
                pl.col("clock_after_seconds").cast(pl.Float64),
                pl.col("target_move_time_seconds").cast(pl.Float64),
                pl.col("prev_clock_seconds").cast(pl.Float64),
                pl.col("clock_diff_seconds").cast(pl.Float64),
                pl.col("is_in_check_before_move").cast(pl.Int8),
                pl.col("total_pieces").cast(pl.Int16),
                pl.when(pl.col("time_initial") > 0)
                .then(pl.col("prev_clock_seconds") / pl.col("time_initial"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("clock_fraction_left"),
            ]
        )
        .filter(pl.col("target_move_time_seconds").is_not_null())
    )
    if "move_entropy" in moves.columns:
        moves = moves.with_columns(pl.col("move_entropy").cast(pl.Float64))
    if "entropy_x_ply_scaling" in moves.columns:
        moves = moves.with_columns(pl.col("entropy_x_ply_scaling").cast(pl.Float64))
    return moves


def _filter_implausible_move_times(moves: pl.DataFrame) -> pl.DataFrame:
    """Drop moves whose reported think time exceeds a generous clock-based bound.

    The labels are derived from clock comments, so a move time should not be much
    larger than the available pre-move clock. We keep a wide 300-second margin to
    avoid removing legitimate long thinks while still catching obvious clock
    mismatches and parsing errors.
    """

    if "prev_clock_seconds" not in moves.columns or "target_move_time_seconds" not in moves.columns:
        return moves

    valid = (
        pl.col("target_move_time_seconds").is_not_null()
        & pl.col("prev_clock_seconds").is_not_null()
        & (pl.col("target_move_time_seconds") >= 0)
        & (pl.col("target_move_time_seconds") <= pl.col("prev_clock_seconds") + MAX_TIME_OVER_CLOCK_SECONDS)
    )
    filtered = moves.filter(valid)

    removed = moves.height - filtered.height
    if removed > 0:
        logger.info(
            "Filtered {} implausible move rows where target_move_time_seconds > prev_clock_seconds + {}",
            removed,
            int(MAX_TIME_OVER_CLOCK_SECONDS),
        )

    return filtered


def _split_games(games: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    required = {"game_idx", "time_initial", "ply_count"}
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"Cannot stratify split without columns: {sorted(missing)}")

    with_strata = games.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("time_initial").map_elements(_bucket_time_initial, return_dtype=pl.Utf8),
                    pl.col("ply_count").map_elements(_bucket_ply_count, return_dtype=pl.Utf8),
                ],
                separator="__",
            ).alias("split_stratum"),
            _stable_shuffle_key(pl.col("game_idx")).alias("shuffle_key"),
        ]
    )

    group_sizes = (
        with_strata.group_by("split_stratum")
        .agg(pl.len().alias("games_in_stratum"))
        .sort("games_in_stratum", descending=True)
    )
    logger.info("Strata distribution before split:\n{}", group_sizes)

    train_parts: list[pl.DataFrame] = []
    val_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []

    for stratum, stratum_df in with_strata.partition_by("split_stratum", as_dict=True).items():
        ordered = stratum_df.sort("shuffle_key")
        n_games = ordered.height
        n_train = int(n_games * TRAIN_FRACTION)
        n_val = int(n_games * VAL_FRACTION)
        n_test = n_games - n_train - n_val

        train_parts.append(ordered.slice(0, n_train).drop(["split_stratum", "shuffle_key"]))
        val_parts.append(ordered.slice(n_train, n_val).drop(["split_stratum", "shuffle_key"]))
        test_parts.append(
            ordered.slice(n_train + n_val, n_test).drop(["split_stratum", "shuffle_key"])
        )

    train = pl.concat(train_parts, how="vertical_relaxed") if train_parts else games.head(0)
    val = pl.concat(val_parts, how="vertical_relaxed") if val_parts else games.head(0)
    test = pl.concat(test_parts, how="vertical_relaxed") if test_parts else games.head(0)

    logger.info(
        "Game-level stratified split: train={} val={} test={}",
        train.height,
        val.height,
        test.height,
    )

    return train, val, test


@hydra.main(version_base=None, config_path="../../config", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    pretrain_path = Path(str(cfg.get("pretrain_input", PRETRAIN_DATASET_PATH)))
    eval_path = Path(str(cfg.get("eval_input", EVAL_DATASET_PATH)))
    output_dir = Path(str(cfg.get("xgboost_output_dir", "data/3_xgboost")))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading preprocess outputs: {}, {}", pretrain_path, eval_path)
    games = _load_games([pretrain_path, eval_path])
    logger.info("Loaded {} game rows", games.height)

    games = _compute_entropy_features(games)

    train_games, val_games, test_games = _split_games(games)

    train = _filter_implausible_move_times(_explode_to_moves(train_games))
    val = _filter_implausible_move_times(_explode_to_moves(val_games))
    test = _filter_implausible_move_times(_explode_to_moves(test_games))

    logger.info(
        "Built move rows after stratified split: train={} val={} test={}",
        train.height,
        val.height,
        test.height,
    )

    train_path = output_dir / "xgb_train.parquet"
    val_path = output_dir / "xgb_val.parquet"
    test_path = output_dir / "xgb_test.parquet"

    train.write_parquet(train_path, compression="zstd")
    val.write_parquet(val_path, compression="zstd")
    test.write_parquet(test_path, compression="zstd")

    logger.info("Saved train={} rows -> {}", train.height, train_path)
    logger.info("Saved val={} rows -> {}", val.height, val_path)
    logger.info("Saved test={} rows -> {}", test.height, test_path)


if __name__ == "__main__":
    main()
