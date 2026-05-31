"""Packing tokenized games into fixed-length training windows."""

from __future__ import annotations

import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.tokens import PAD_ID

_GameRow = tuple[list[int], list[int], list[int]]

_PACKED_SCHEMA = {
    "token_ids": pl.List(pl.Int64),
    "active_clock_ids": pl.List(pl.Int64),
    "opponent_clock_ids": pl.List(pl.Int64),
}

_SHARD_COLUMNS = (
    ("token_ids", np.uint16),
    ("active_clock_ids", np.uint32),
    ("opponent_clock_ids", np.uint32),
)


def one_row_one_game(lazy_df: pl.LazyFrame, block_size: int) -> pl.LazyFrame:
    window_size = block_size + 1
    columns = ["token_ids", "active_clock_ids", "opponent_clock_ids"]
    return lazy_df.select(
        [pl.col(column).list.slice(0, window_size).alias(column) for column in columns]
    )


def _append_game_prefix(
    game: _GameRow,
    start: int,
    *,
    window_size: int,
    tokens: list[int],
    active: list[int],
    opp: list[int],
) -> int:
    """Append game tokens from ``start``; return resume_index when full."""
    game_tokens, game_active, game_opp = game
    for idx in range(start, len(game_tokens)):
        if len(tokens) >= window_size:
            return idx
        tokens.append(game_tokens[idx])
        active.append(game_active[idx])
        opp.append(game_opp[idx])
    return len(game_tokens)


class PackedWindowBuilder:
    """Memory-bounded packer: feeds games incrementally and spills fixed-array shards."""

    def __init__(self, block_size: int, *, flush_every: int = 8_000) -> None:
        self.window_size = block_size + 1
        self.flush_every = max(1, flush_every)
        self.pending: _GameRow | None = None
        self._games: deque[_GameRow] = deque()
        self._buffer = self._new_buffer()
        self._buffer_rows = 0
        self.shard_paths: list[tuple[Path, int]] = []

    def _new_buffer(self, rows: int | None = None) -> dict[str, np.ndarray]:
        row_count = self.flush_every if rows is None else rows
        return {
            column: np.empty((row_count, self.window_size), dtype=dtype)
            for column, dtype in _SHARD_COLUMNS
        }

    @property
    def _buffer_capacity(self) -> int:
        return next(iter(self._buffer.values())).shape[0]

    def feed_from_columns(
        self,
        token_ids_col,
        active_col,
        opp_col,
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

    def drain(self, part_dir: Path | None = None) -> None:
        while True:
            if part_dir is None:
                if self._buffer_rows >= self._buffer_capacity:
                    self._grow_buffer()
            elif self._buffer_rows >= self.flush_every:
                self._flush(part_dir)
            if not self._emit_one_window():
                return

    def _emit_one_window(self) -> bool:
        if self.pending is None and not self._games:
            return False

        tokens: list[int] = []
        active: list[int] = []
        opp: list[int] = []

        if self.pending is not None:
            resume = _append_game_prefix(
                self.pending,
                0,
                window_size=self.window_size,
                tokens=tokens,
                active=active,
                opp=opp,
            )
            self.pending = self.pending if resume < len(self.pending[0]) else None

        while self.pending is None and self._games and len(tokens) < self.window_size:
            game = self._games.popleft()
            resume = _append_game_prefix(
                game,
                0,
                window_size=self.window_size,
                tokens=tokens,
                active=active,
                opp=opp,
            )
            if resume < len(game[0]):
                self.pending = game

        self._append_window(tokens, active, opp)
        return True

    def _append_window(self, tokens: list[int], active: list[int], opp: list[int]) -> None:
        row = self._buffer_rows
        if row >= self._buffer_capacity:
            raise RuntimeError("Packed window buffer is full; flush before appending")

        length = len(tokens)
        self._buffer["token_ids"][row, :length] = tokens
        self._buffer["active_clock_ids"][row, :length] = active
        self._buffer["opponent_clock_ids"][row, :length] = opp
        if length < self.window_size:
            self._buffer["token_ids"][row, length:] = PAD_ID
            self._buffer["active_clock_ids"][row, length:] = CLOCK_IGNORE_ID
            self._buffer["opponent_clock_ids"][row, length:] = CLOCK_IGNORE_ID
        self._buffer_rows += 1

    def maybe_flush(self, part_dir: Path) -> None:
        if self._buffer_rows < self.flush_every:
            return
        self._flush(part_dir)

    def _flush(self, part_dir: Path) -> None:
        part_dir.mkdir(parents=True, exist_ok=True)
        path = part_dir / f"part_{len(self.shard_paths):04d}"
        rows = _write_packed_shard(path, self._buffer, self._buffer_rows, self.window_size)
        self.shard_paths.append((path, rows))
        self._buffer = self._new_buffer()
        self._buffer_rows = 0

    def _grow_buffer(self) -> None:
        grown = self._new_buffer(rows=self._buffer_rows * 2)
        for column, _dtype in _SHARD_COLUMNS:
            grown[column][: self._buffer_rows] = self._buffer[column][: self._buffer_rows]
        self._buffer = grown

    def finish(self, output_path: Path, *, part_dir: Path) -> None:
        self.drain(part_dir)
        part_dir.mkdir(parents=True, exist_ok=True)
        if self._buffer_rows:
            self._flush(part_dir)

        _write_packed_dataset_manifest(output_path, self.shard_paths, self.window_size)

    def to_dataframe(self) -> pl.DataFrame:
        if self._buffer_rows == 0:
            return pl.DataFrame(schema=_PACKED_SCHEMA)
        return pl.DataFrame(
            {
                column: self._buffer[column][: self._buffer_rows].tolist()
                for column, _dtype in _SHARD_COLUMNS
            },
            schema=_PACKED_SCHEMA,
        )


def pack_games_into_windows(games: pl.DataFrame, block_size: int, seed: int) -> pl.DataFrame:
    """Pack games into fixed windows; split games restart from ``<game_start>`` in the next row."""
    if games.is_empty():
        return pl.DataFrame(schema=_PACKED_SCHEMA)

    builder = PackedWindowBuilder(block_size, flush_every=len(games) + 1)
    builder.feed_dataframe(games, shuffle_seed=seed)
    builder.drain()
    return builder.to_dataframe()


def _write_packed_shard(
    path: Path,
    arrays: dict[str, np.ndarray],
    rows: int,
    window_size: int,
) -> int:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for column, arr in arrays.items():
        shard = np.ascontiguousarray(arr[:rows])
        if shard.ndim != 2 or shard.shape[1] != window_size:
            raise ValueError(f"{column} shard has invalid shape {shard.shape}")
        np.save(path / f"{column}.npy", shard)
    return rows


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
