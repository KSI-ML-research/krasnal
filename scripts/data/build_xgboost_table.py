"""Build move-level train/val/test parquet tables for XGBoost.

Input files are game-level parquet files from data/1_filtered/ (produced by
download_games.py) with Aix clock columns (clocks_white, clocks_black,
time_initial, time_increment). All columns in _REQUIRED_COLUMNS must be present.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import chess
import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from krasnal.config import RAW_UCI_DIR

MAX_TIME_OVER_CLOCK_SECONDS = 300.0
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

_REQUIRED_COLUMNS = {
    "lichess_id",
    "uci_moves",
    "clocks_white",
    "clocks_black",
    "time_initial",
    "time_increment",
    "ply_count",
    "is_check",
    "piece_moved",
    "white_rating",
    "black_rating",
    "result",
}

_READ_COLUMNS = sorted(_REQUIRED_COLUMNS)


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
    return ((game_idx.cast(pl.Int64) * 1103515245) + seed) % 2_147_483_647


def _interleave_clocks(
    clocks_white: list[float] | None,
    clocks_black: list[float] | None,
    n_moves: int,
) -> list[float | None]:
    if clocks_white is None or clocks_black is None:
        return [None] * n_moves
    result: list[float | None] = []
    w_idx = 0
    b_idx = 0
    for ply in range(n_moves):
        if ply % 2 == 0:
            if w_idx < len(clocks_white):
                result.append(clocks_white[w_idx])
                w_idx += 1
            else:
                result.append(None)
        else:
            if b_idx < len(clocks_black):
                result.append(clocks_black[b_idx])
                b_idx += 1
            else:
                result.append(None)
    return result


def _compute_move_time_seconds(
    clocks_white: list[float] | None,
    clocks_black: list[float] | None,
    time_initial: float,
    time_increment: float,
    n_moves: int,
) -> list[float | None]:
    if clocks_white is None or clocks_black is None:
        return [None] * max(n_moves, 0)
    result: list[float | None] = []
    white_prev = time_initial
    black_prev = time_initial
    w_idx = 0
    b_idx = 0
    for ply in range(n_moves):
        if ply % 2 == 0:
            if w_idx < len(clocks_white):
                cur = clocks_white[w_idx]
                w_idx += 1
            else:
                result.append(None)
                continue
            if cur is None:
                result.append(None)
            else:
                used = white_prev - float(cur) + time_increment
                result.append(max(0.0, used))
                white_prev = float(cur)
        else:
            if b_idx < len(clocks_black):
                cur = clocks_black[b_idx]
                b_idx += 1
            else:
                result.append(None)
                continue
            if cur is None:
                result.append(None)
            else:
                used = black_prev - float(cur) + time_increment
                result.append(max(0.0, used))
                black_prev = float(cur)
    return result


def _compute_prev_clock_seconds(
    move_clocks: list[float | None],
) -> list[float | None]:
    result: list[float | None] = []
    for i in range(len(move_clocks)):
        if i >= 2:
            result.append(move_clocks[i - 2])
        else:
            result.append(None)
    return result


def _compute_clock_diff_seconds(
    move_clocks: list[float | None],
) -> list[float | None]:
    result: list[float | None] = []
    for i in range(len(move_clocks)):
        if i >= 2 and move_clocks[i - 2] is not None and move_clocks[i] is not None:
            try:
                diff = max(0.0, float(move_clocks[i - 2]) - float(move_clocks[i]))
                result.append(diff)
            except Exception:
                result.append(None)
        else:
            result.append(None)
    return result


def _load_games(input_paths: list[Path]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    game_offset = 0

    for path in input_paths:
        if not path.exists():
            logger.warning("Skipping missing input: {}", path)
            continue

        df = pl.read_parquet(path, columns=_READ_COLUMNS)

        missing = _REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            logger.warning("Skipping {} (missing columns: {})", path, sorted(missing))
            continue

        clock_missing_count = df.filter(
            pl.col("clocks_white").is_null() | pl.col("clocks_black").is_null()
        ).height
        if clock_missing_count > 0:
            logger.warning(
                "{} has {} games with null clock columns; skipping those rows",
                path.name,
                clock_missing_count,
            )

        df = df.filter(pl.col("clocks_white").is_not_null() & pl.col("clocks_black").is_not_null())

        if df.height == 0:
            logger.warning("No games with clock data in {}, skipping", path)
            continue

        n_moves_list: list[int] = []
        mc_list: list[list[float | None]] = []
        mt_list: list[list[float | None]] = []
        ply_lists: list[list[int]] = []
        prev_clock_lists: list[list[float | None]] = []
        clock_diff_lists: list[list[float | None]] = []

        cw_rows = df["clocks_white"].to_list()
        cb_rows = df["clocks_black"].to_list()
        ti_vals = df["time_initial"].to_list()
        tinc_vals = df["time_increment"].to_list()
        uci_rows = df["uci_moves"].to_list()

        for cw, cb, ti, tinc, uci_str in zip(
            cw_rows, cb_rows, ti_vals, tinc_vals, uci_rows, strict=True
        ):
            if cw is None or cb is None:
                n_moves_list.append(0)
                mc_list.append([])
                mt_list.append([])
                ply_lists.append([])
                prev_clock_lists.append([])
                clock_diff_lists.append([])
                continue
            if uci_str is None:
                n_moves_list.append(0)
                mc_list.append([])
                mt_list.append([])
                ply_lists.append([])
                prev_clock_lists.append([])
                clock_diff_lists.append([])
                continue

            cw_clean = [float(v) if v is not None else None for v in cw]
            cb_clean = [float(v) if v is not None else None for v in cb]
            ti_val = float(ti) if ti is not None else 0.0
            tinc_val = float(tinc) if tinc is not None else 0.0
            moves_str = str(uci_str)
            n_moves = len(moves_str.split())

            mc = _interleave_clocks(cw_clean, cb_clean, n_moves)
            mt = _compute_move_time_seconds(cw_clean, cb_clean, ti_val, tinc_val, n_moves)
            ply_list = list(range(n_moves))
            prev_clocks = _compute_prev_clock_seconds(mc)
            clock_diffs = _compute_clock_diff_seconds(mc)

            n_moves_list.append(n_moves)
            mc_list.append(mc)
            mt_list.append(mt)
            ply_lists.append(ply_list)
            prev_clock_lists.append(prev_clocks)
            clock_diff_lists.append(clock_diffs)

        df = df.with_columns(
            pl.Series("ply_list", ply_lists, dtype=pl.List(pl.Int64)),
            pl.Series("move_clocks_seconds", mc_list, dtype=pl.List(pl.Float64)),
            pl.Series("move_time_seconds", mt_list, dtype=pl.List(pl.Float64)),
            pl.Series("prev_clock_seconds", prev_clock_lists, dtype=pl.List(pl.Float64)),
            pl.Series("clock_diff_seconds", clock_diff_lists, dtype=pl.List(pl.Float64)),
        )

        uci_rows = df["uci_moves"].to_list()
        ply_rows = df["ply_list"].to_list()
        is_check_rows = df["is_check"].to_list()
        check_before: list[list[int | None]] = []
        piece_count_before: list[list[int | None]] = []

        for uci_raw, ply_raw, is_check_raw in zip(uci_rows, ply_rows, is_check_rows, strict=True):
            n = len(ply_raw) if ply_raw is not None else 0
            if n == 0:
                check_before.append([])
                piece_count_before.append([])
                continue

            if uci_raw is None:
                check_before.append([None] * n)
                piece_count_before.append([None] * n)
                continue

            if isinstance(uci_raw, str):
                move_tokens = uci_raw.split()
            elif isinstance(uci_raw, list):
                move_tokens = [str(m) for m in uci_raw]
            else:
                check_before.append([None] * n)
                piece_count_before.append([None] * n)
                continue

            board = chess.Board()
            row_check: list[int | None] = []
            row_pieces: list[int | None] = []

            if is_check_raw is None or len(is_check_raw) < n:
                is_check_safe = [False] * n
            else:
                is_check_safe = list(is_check_raw)
            is_check_safe = is_check_safe[:n]
            while len(is_check_safe) < n:
                is_check_safe.append(False)

            for ply_idx in range(n):
                row_check.append(1 if board.is_check() else 0)
                row_pieces.append(len(board.piece_map()))
                if ply_idx < len(move_tokens):
                    with contextlib.suppress(Exception):
                        board.push_uci(move_tokens[ply_idx])

            check_before.append(row_check)
            piece_count_before.append(row_pieces)

        df = df.with_columns(
            pl.Series("is_in_check_before_move", check_before, dtype=pl.List(pl.Int8)),
            pl.Series("total_pieces_before_move", piece_count_before, dtype=pl.List(pl.Int16)),
        )

        df = df.with_row_index("game_idx", offset=game_offset)
        game_offset += df.height
        frames.append(df)

    if not frames:
        raise ValueError("No valid input parquet files found")

    return pl.concat(frames, how="vertical_relaxed")


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
    return moves


def _filter_implausible_move_times(moves: pl.DataFrame) -> pl.DataFrame:
    if "prev_clock_seconds" not in moves.columns or "target_move_time_seconds" not in moves.columns:
        return moves

    valid = (
        pl.col("target_move_time_seconds").is_not_null()
        & pl.col("prev_clock_seconds").is_not_null()
        & (pl.col("target_move_time_seconds") >= 0)
        & (
            pl.col("target_move_time_seconds")
            <= pl.col("prev_clock_seconds") + MAX_TIME_OVER_CLOCK_SECONDS
        )
    )
    filtered = moves.filter(valid)

    removed = moves.height - filtered.height
    if removed > 0:
        logger.info(
            "Filtered {} implausible move rows where "
            "target_move_time_seconds > prev_clock_seconds + {}",
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

    for _stratum, stratum_df in with_strata.partition_by("split_stratum", as_dict=True).items():
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
    input_raw = str(cfg.get("xgb_input_dir", str(RAW_UCI_DIR)))
    input_dir = Path(input_raw)
    output_dir = Path(str(cfg.get("xgboost_output_dir", "data/3_xgboost")))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    logger.info("Reading filtered games from {} ({} files)", input_dir, len(input_paths))
    games = _load_games(input_paths)
    logger.info("Loaded {} game rows", games.height)

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
