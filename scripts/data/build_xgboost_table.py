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
from loguru import logger
from omegaconf import DictConfig

from krasnal.config import EVAL_DATASET_PATH, PRETRAIN_DATASET_PATH
from krasnal.lichess_clocks import fetch_lichess_pgn


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

        # Compute whether side to move is currently in check before each move.
        if "is_in_check_before_move" not in df.columns and "ply_list" in df.columns:
            uci_rows = df["uci_moves"].to_list() if "uci_moves" in df.columns else [None] * df.height
            ply_rows = df["ply_list"].to_list()
            check_rows: list[list[int | None]] = []

            for uci_raw, ply_raw in zip(uci_rows, ply_rows):
                n = len(ply_raw) if ply_raw is not None else 0
                if n == 0:
                    check_rows.append([])
                    continue

                if uci_raw is None:
                    check_rows.append([None] * n)
                    continue

                if isinstance(uci_raw, str):
                    move_tokens = uci_raw.split()
                elif isinstance(uci_raw, list):
                    move_tokens = [str(m) for m in uci_raw]
                else:
                    check_rows.append([None] * n)
                    continue

                board = chess.Board()
                row_values: list[int | None] = []
                try:
                    for token in move_tokens[:n]:
                        row_values.append(1 if board.is_check() else 0)
                        board.push_uci(token)
                except Exception:
                    check_rows.append([None] * n)
                    continue

                if len(row_values) < n:
                    row_values.extend([None] * (n - len(row_values)))
                check_rows.append(row_values)

            df = df.with_columns(
                pl.Series("is_in_check_before_move", check_rows, dtype=pl.List(pl.Int8))
            )

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


def _explode_to_moves(games: pl.DataFrame) -> pl.DataFrame:
    moves = (
        games.select(
            "game_idx",
            "ply_list",
            "move_clocks_seconds",
            "move_time_seconds",
            "prev_clock_seconds",
            "clock_diff_seconds",
            "is_in_check_before_move",
        )
        .explode(
            [
                "ply_list",
                "move_clocks_seconds",
                "move_time_seconds",
                "prev_clock_seconds",
                "clock_diff_seconds",
                "is_in_check_before_move",
            ]
        )
        .rename(
            {
                "ply_list": "ply",
                "move_clocks_seconds": "clock_after_seconds",
                "move_time_seconds": "target_move_time_seconds",
                "prev_clock_seconds": "prev_clock_seconds",
                "clock_diff_seconds": "clock_diff_seconds",
                "is_in_check_before_move": "is_in_check_before_move",
            }
        )
        .with_columns(
            [
                (pl.col("ply") % 2).cast(pl.Int8).alias("side_to_move"),
                pl.col("ply").cast(pl.Int32),
                pl.col("clock_after_seconds").cast(pl.Float64),
                pl.col("target_move_time_seconds").cast(pl.Float64),
                pl.col("prev_clock_seconds").cast(pl.Float64),
                pl.col("clock_diff_seconds").cast(pl.Float64),
                pl.col("is_in_check_before_move").cast(pl.Int8),
            ]
        )
        .filter(pl.col("target_move_time_seconds").is_not_null())
    )
    return moves


def _split_games(moves: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    with_split = moves.with_columns((pl.col("game_idx") % 100).alias("split_bucket"))

    train = with_split.filter(pl.col("split_bucket") < 70).drop("split_bucket")
    val = with_split.filter((pl.col("split_bucket") >= 70) & (pl.col("split_bucket") < 85)).drop(
        "split_bucket"
    )
    test = with_split.filter(pl.col("split_bucket") >= 85).drop("split_bucket")

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

    moves = _explode_to_moves(games)
    logger.info("Built {} move rows with labels", moves.height)

    train, val, test = _split_games(moves)

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
