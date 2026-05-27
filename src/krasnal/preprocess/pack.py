"""Packing tokenized games into fixed-length training windows."""

from __future__ import annotations

import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.tokens import GAME_START_ID, PAD_ID

PAD_SEGMENT_ID = -1

_GameRow = tuple[list[int], list[int], list[int]]

_PACKED_SCHEMA = {
    "token_ids": pl.List(pl.Int64),
    "active_clock_ids": pl.List(pl.Int64),
    "opponent_clock_ids": pl.List(pl.Int64),
    "segment_ids": pl.List(pl.Int32),
    "position_ids": pl.List(pl.Int32),
}

_SHARD_COLUMNS = (
    ("token_ids", np.uint16),
    ("active_clock_ids", np.uint32),
    ("opponent_clock_ids", np.uint32),
    ("segment_ids", np.int16),
    ("position_ids", np.int16),
)


def one_row_one_game(lazy_df: pl.LazyFrame, block_size: int) -> pl.LazyFrame:
    window_size = block_size + 1
    columns = ["token_ids", "active_clock_ids", "opponent_clock_ids"]
    return lazy_df.select(
        [pl.col(column).list.slice(0, window_size).alias(column) for column in columns]
    )


def _pad_window_row(
    tokens: list[int],
    active: list[int],
    opp: list[int],
    segments: list[int],
    positions: list[int],
    *,
    window_size: int,
) -> dict[str, list[int]]:
    pad_len = window_size - len(tokens)
    if pad_len > 0:
        tokens.extend([PAD_ID] * pad_len)
        active.extend([CLOCK_IGNORE_ID] * pad_len)
        opp.extend([CLOCK_IGNORE_ID] * pad_len)
        segments.extend([PAD_SEGMENT_ID] * pad_len)
        positions.extend([0] * pad_len)
    return {
        "token_ids": tokens,
        "active_clock_ids": active,
        "opponent_clock_ids": opp,
        "segment_ids": segments,
        "position_ids": positions,
    }


def _append_game_prefix(
    game: _GameRow,
    start: int,
    *,
    window_size: int,
    tokens: list[int],
    active: list[int],
    opp: list[int],
    segments: list[int],
    positions: list[int],
    segment: int,
    pos_in_segment: int,
) -> tuple[int, int, int]:
    """Append game tokens from ``start``; return (segment, pos, resume_index) when full."""
    game_tokens, game_active, game_opp = game
    for idx in range(start, len(game_tokens)):
        if len(tokens) >= window_size:
            return segment, pos_in_segment, idx
        tok = game_tokens[idx]
        if tok == GAME_START_ID:
            segment += 1
            pos_in_segment = 0
        tokens.append(tok)
        active.append(game_active[idx])
        opp.append(game_opp[idx])
        segments.append(segment)
        positions.append(pos_in_segment)
        pos_in_segment += 1
    return segment, pos_in_segment, len(game_tokens)


class PackedWindowBuilder:
    """Memory-bounded packer: feeds games incrementally and spills fixed-array shards."""

    def __init__(self, block_size: int, *, flush_every: int = 8_000) -> None:
        self.window_size = block_size + 1
        self.flush_every = max(1, flush_every)
        self.pending: _GameRow | None = None
        self._games: deque[_GameRow] = deque()
        self.row_buffer: list[dict[str, list[int]]] = []
        self.shard_paths: list[tuple[Path, int]] = []

    def feed_from_columns(
        self,
        token_ids_col: list[list[int]],
        active_col: list[list[int]],
        opp_col: list[list[int]],
    ) -> None:
        """Queue games from three parallel list-of-lists (e.g. from Polars ``.to_list()``)."""
        for tokens, active, opp in zip(token_ids_col, active_col, opp_col, strict=True):
            if len(tokens) > self.window_size:
                raise ValueError(
                    f"Game length {len(tokens)} exceeds packed window size {self.window_size}; "
                    "filter games before packing"
                )
            self._games.append((tokens, active, opp))

    def feed_dataframe(self, games: pl.DataFrame, shuffle_seed: int) -> None:
        """Queue all games from a frame, then call ``drain()`` (used in tests)."""
        if games.is_empty():
            return
        shuffled = games.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)
        self.feed_from_columns(
            shuffled["token_ids"].to_list(),
            shuffled["active_clock_ids"].to_list(),
            shuffled["opponent_clock_ids"].to_list(),
        )

    def drain(self) -> None:
        while self._emit_one_window():
            pass

    def _emit_one_window(self) -> bool:
        if self.pending is None and not self._games:
            return False

        tokens: list[int] = []
        active: list[int] = []
        opp: list[int] = []
        segments: list[int] = []
        positions: list[int] = []
        segment = -1
        pos_in_segment = 0

        if self.pending is not None:
            segment, pos_in_segment, resume = _append_game_prefix(
                self.pending,
                0,
                window_size=self.window_size,
                tokens=tokens,
                active=active,
                opp=opp,
                segments=segments,
                positions=positions,
                segment=segment,
                pos_in_segment=pos_in_segment,
            )
            self.pending = self.pending if resume < len(self.pending[0]) else None

        while self.pending is None and self._games and len(tokens) < self.window_size:
            game = self._games.popleft()
            segment, pos_in_segment, resume = _append_game_prefix(
                game,
                0,
                window_size=self.window_size,
                tokens=tokens,
                active=active,
                opp=opp,
                segments=segments,
                positions=positions,
                segment=segment,
                pos_in_segment=pos_in_segment,
            )
            if resume < len(game[0]):
                self.pending = game

        self.row_buffer.append(
            _pad_window_row(
                tokens,
                active,
                opp,
                segments,
                positions,
                window_size=self.window_size,
            )
        )
        return True

    def maybe_flush(self, part_dir: Path) -> None:
        if len(self.row_buffer) < self.flush_every:
            return
        part_dir.mkdir(parents=True, exist_ok=True)
        path = part_dir / f"part_{len(self.shard_paths):04d}"
        rows = _write_packed_shard(path, self.row_buffer, self.window_size)
        self.shard_paths.append((path, rows))
        self.row_buffer.clear()

    def finish(self, output_path: Path, *, part_dir: Path) -> None:
        self.drain()
        part_dir.mkdir(parents=True, exist_ok=True)
        if self.row_buffer:
            path = part_dir / f"part_{len(self.shard_paths):04d}"
            rows = _write_packed_shard(path, self.row_buffer, self.window_size)
            self.shard_paths.append((path, rows))
            self.row_buffer.clear()

        _write_packed_dataset_manifest(output_path, self.shard_paths, self.window_size)


def pack_games_into_windows(games: pl.DataFrame, block_size: int, seed: int) -> pl.DataFrame:
    """Pack games into fixed windows; split games restart from ``<game_start>`` in the next row."""
    if games.is_empty():
        return pl.DataFrame(schema=_PACKED_SCHEMA)

    builder = PackedWindowBuilder(block_size, flush_every=len(games) + 1)
    builder.feed_dataframe(games, shuffle_seed=seed)
    builder.drain()
    if not builder.row_buffer:
        return pl.DataFrame(schema=_PACKED_SCHEMA)
    return pl.DataFrame(builder.row_buffer)


def _rows_to_arrays(
    rows: list[dict[str, list[int]]],
    window_size: int,
) -> dict[str, np.ndarray]:
    arrays = {}
    for column, dtype in _SHARD_COLUMNS:
        arr = np.asarray([row[column] for row in rows], dtype=dtype)
        if arr.ndim != 2 or arr.shape[1] != window_size:
            raise ValueError(f"{column} shard has invalid shape {arr.shape}")
        arrays[column] = arr
    return arrays


def _write_packed_shard(
    path: Path,
    rows: list[dict[str, list[int]]],
    window_size: int,
) -> int:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    arrays = _rows_to_arrays(rows, window_size)
    for column, arr in arrays.items():
        np.save(path / f"{column}.npy", arr)
    return len(rows)


def _write_packed_dataset_manifest(
    output_path: Path,
    shard_paths: list[tuple[Path, int]],
    window_size: int,
) -> None:
    if output_path.is_dir():
        shutil.rmtree(output_path)
    elif output_path.exists():
        output_path.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    shards = []
    total_rows = 0
    for idx, (src_path, rows) in enumerate(shard_paths):
        dst_path = output_path / f"part_{idx:04d}"
        if src_path.resolve() != dst_path.resolve():
            shutil.copytree(src_path, dst_path)
        total_rows += rows
        shards.append({"path": dst_path.name, "rows": rows})

    with (output_path / "metadata.json").open("w") as f:
        json.dump(
            {
                "format": "krasnal-packed-npy",
                "version": 1,
                "window_size": window_size,
                "rows": total_rows,
                "columns": [name for name, _dtype in _SHARD_COLUMNS],
                "shards": shards,
            },
            f,
            indent=2,
        )
        f.write("\n")
