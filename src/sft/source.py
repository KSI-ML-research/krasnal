from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import polars as pl
from tqdm.auto import tqdm

from config import RAW_DATA_DIR, ChessGPTConfig
from tokenizer import Tokenizer

from .format import build_cot_row


@dataclass
class SampleStats:
    """Track generation attempts and accepted rows."""

    attempts: int = 0
    accepted: int = 0

    @property
    def rejection_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return (self.attempts - self.accepted) / self.attempts


def build_board(prefix_moves: list[str]) -> chess.Board | None:
    """Reconstruct a board from a prefix of UCI moves."""
    board = chess.Board()
    try:
        for move in prefix_moves:
            board.push_uci(move)
    except ValueError:
        return None
    return board


def load_raw_games() -> pl.DataFrame:
    """Load the raw game columns needed for CoT generation."""
    return pl.scan_parquet(f"{RAW_DATA_DIR}/*.parquet").select("result", "moves").collect()


class OnlineCotDataSource:
    """Generate synthetic CoT rows from real-game positions with Stockfish."""

    def __init__(
        self,
        *,
        games: pl.DataFrame,
        tokenizer: Tokenizer,
        stockfish_path: Path,
        multipv_min: int,
        multipv_max: int,
        depth: int | None,
        movetime_ms: int | None,
        seed: int,
        max_attempts_per_sample: int = 100,
    ) -> None:
        if multipv_min <= 0 or multipv_max < multipv_min:
            raise ValueError("Invalid MultiPV range")
        if (depth is None) == (movetime_ms is None):
            raise ValueError("Use exactly one of --depth or --movetime-ms")
        self.games = games
        self.tokenizer = tokenizer
        self.stockfish_path = stockfish_path
        self.multipv_min = multipv_min
        self.multipv_max = multipv_max
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.max_attempts_per_sample = max_attempts_per_sample
        self.max_len = ChessGPTConfig.block_size
        self.rng = random.Random(seed)
        self.limit = chess.engine.Limit(
            depth=depth,
            time=None if movetime_ms is None else movetime_ms / 1000,
        )
        self.engine: chess.engine.SimpleEngine | None = None
        self.stats = SampleStats()

    def __enter__(self) -> OnlineCotDataSource:
        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.stockfish_path))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def _require_engine(self) -> chess.engine.SimpleEngine:
        if self.engine is None:
            raise RuntimeError("OnlineCotDataSource must be used as a context manager")
        return self.engine

    def sample_row(self) -> dict[str, int | str | list[int] | None]:
        """Generate one accepted CoT row."""
        engine = self._require_engine()
        for _ in range(self.max_attempts_per_sample):
            self.stats.attempts += 1
            game_index = self.rng.randrange(self.games.height)
            result, moves_str = self.games.row(game_index)
            moves = str(moves_str).split()
            if len(moves) < 2:
                continue

            move_index = self.rng.randint(1, len(moves) - 1)
            prefix_moves = moves[:move_index]
            actual_move = moves[move_index]
            board = build_board(prefix_moves)
            if board is None or actual_move not in self.tokenizer.move_to_id:
                continue

            multipv = self.rng.randint(self.multipv_min, self.multipv_max)
            infos = engine.analyse(board, self.limit, multipv=multipv)
            if isinstance(infos, dict):
                infos = [infos]

            pv_lines: list[list[str]] = []
            top_score_cp: int | None = None
            for info in infos:
                pv = [
                    move.uci()
                    for move in info.get("pv", [])
                    if move.uci() in self.tokenizer.move_to_id
                ]
                if pv:
                    pv_lines.append(pv)
                if top_score_cp is None and "score" in info:
                    top_score_cp = info["score"].pov(board.turn).score(mate_score=100000)

            if not pv_lines:
                continue

            row = build_cot_row(
                tokenizer=self.tokenizer,
                result=int(result),
                prefix_moves=prefix_moves,
                pv_lines=pv_lines,
                actual_move=actual_move,
                depth=self.depth,
                movetime_ms=self.movetime_ms,
                stockfish_score_cp=top_score_cp,
                source_game_index=game_index,
            )
            if len(row["token_ids"]) > self.max_len:
                continue

            self.stats.accepted += 1
            return row
        raise RuntimeError(
            f"Could not generate a valid CoT row after {self.max_attempts_per_sample} attempts"
        )

    def sample_rows(self, num_rows: int) -> list[dict[str, int | str | list[int] | None]]:
        """Generate multiple accepted CoT rows."""
        return [self.sample_row() for _ in range(num_rows)]


def sample_cot_rows(
    *,
    games: pl.DataFrame,
    tokenizer: Tokenizer,
    num_samples: int,
    multipv_min: int,
    multipv_max: int,
    stockfish_path: Path,
    depth: int | None,
    movetime_ms: int | None,
    seed: int,
) -> list[dict[str, int | str | list[int] | None]]:
    """Generate a fixed offline set of CoT rows."""
    if num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    with OnlineCotDataSource(
        games=games,
        tokenizer=tokenizer,
        stockfish_path=stockfish_path,
        multipv_min=multipv_min,
        multipv_max=multipv_max,
        depth=depth,
        movetime_ms=movetime_ms,
        seed=seed,
    ) as source:
        rows: list[dict[str, int | str | list[int] | None]] = []
        with tqdm(total=num_samples, desc="generate-sft-cot", unit="sample") as progress:
            for _ in range(num_samples):
                rows.append(source.sample_row())
                progress.update(1)
        return rows
