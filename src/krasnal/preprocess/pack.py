"""Packing tokenized games into fixed-length training windows."""

from __future__ import annotations

from pathlib import Path

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
    """Memory-bounded packer: feeds games incrementally and spills windows to parquet parts."""

    def __init__(self, block_size: int, *, flush_every: int = 8_000) -> None:
        self.window_size = block_size + 1
        self.flush_every = max(1, flush_every)
        self.pending: _GameRow | None = None
        self._games: list[_GameRow] = []
        self.row_buffer: list[dict[str, list[int]]] = []
        self.part_paths: list[Path] = []

    @staticmethod
    def _parse_game_row(row: dict[str, object]) -> _GameRow:
        tokens = [int(x) for x in row["token_ids"]]
        active = [int(x) for x in row["active_clock_ids"]]
        opp = [int(x) for x in row["opponent_clock_ids"]]
        if not (len(tokens) == len(active) == len(opp)):
            raise ValueError("Clock/token length mismatch while packing games")
        return tokens, active, opp

    def feed_game(self, game: _GameRow) -> None:
        if len(game[0]) > self.window_size:
            raise ValueError(
                f"Game length {len(game[0])} exceeds packed window size {self.window_size}; "
                "filter games before packing"
            )
        self._games.append(game)
        while self._emit_one_window():
            pass

    def feed_dataframe(self, games: pl.DataFrame, shuffle_seed: int) -> None:
        """Queue all games from a frame, then call ``drain()`` (used in tests)."""
        if games.is_empty():
            return
        shuffled = games.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)
        for row in shuffled.iter_rows(named=True):
            game = self._parse_game_row(row)
            if len(game[0]) > self.window_size:
                raise ValueError(
                    f"Game length {len(game[0])} exceeds packed window size {self.window_size}; "
                    "filter games before packing"
                )
            self._games.append(game)

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
            game = self._games.pop(0)
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
        path = part_dir / f"part_{len(self.part_paths):04d}.parquet"
        pl.DataFrame(self.row_buffer).write_parquet(path)
        self.part_paths.append(path)
        self.row_buffer.clear()

    def finish(self, output_path: Path, *, part_dir: Path | None = None) -> None:
        self.drain()
        if part_dir is not None:
            part_dir.mkdir(parents=True, exist_ok=True)
            if self.row_buffer:
                path = part_dir / f"part_{len(self.part_paths):04d}.parquet"
                pl.DataFrame(self.row_buffer).write_parquet(path)
                self.part_paths.append(path)
                self.row_buffer.clear()

        if self.part_paths:
            pl.concat(pl.scan_parquet(p) for p in self.part_paths).sink_parquet(output_path)
            return

        if self.row_buffer:
            pl.DataFrame(self.row_buffer).write_parquet(output_path)
            return

        pl.DataFrame(schema=_PACKED_SCHEMA).write_parquet(output_path)


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
