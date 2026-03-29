from __future__ import annotations

import random
from pathlib import Path

import chess
import chess.engine
import polars as pl
from tqdm.auto import tqdm

from krasnal.config import RAW_UCI_DIR
from krasnal.sft.generation.format import build_cot_row
from krasnal.tokens import BLACK_PREFIX, MOVE_TO_ID, WHITE_PREFIX


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
    return pl.scan_parquet(str(RAW_UCI_DIR / "*.parquet")).select("result", "moves").collect()


class OnlineCotDataSource:
    """Generate synthetic CoT rows from real-game positions with Stockfish."""

    def __init__(
        self,
        *,
        games: pl.DataFrame,
        stockfish_path: Path,
        multipv_min: int,
        multipv_max: int,
        depth: int,
        seed: int,
        max_len: int,
        max_attempts_per_sample: int = 100,
    ) -> None:
        if multipv_min <= 0 or multipv_max < multipv_min:
            raise ValueError("Invalid MultiPV range")
        if depth is None or depth <= 0:
            raise ValueError("--depth must be > 0")
        self.games = games
        self.stockfish_path = stockfish_path
        self.multipv_min = multipv_min
        self.multipv_max = multipv_max
        self.depth = depth
        self.max_attempts_per_sample = max_attempts_per_sample
        self.max_len = max_len
        self.rng = random.Random(seed)
        self.limit = chess.engine.Limit(depth=depth)
        self.engine: chess.engine.SimpleEngine | None = None

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
            game_index = self.rng.randrange(self.games.height)
            result, moves_str = self.games.row(game_index)
            moves = str(moves_str).split()
            if len(moves) < 2:
                continue

            move_index = self.rng.randint(1, len(moves) - 1)
            prefix_moves = moves[:move_index]
            actual_move = moves[move_index]
            suffix_moves = moves[move_index + 1 :]
            board = build_board(prefix_moves)

            ply = move_index - 1
            actual_prefixed = (WHITE_PREFIX if ply % 2 == 0 else BLACK_PREFIX) + actual_move

            if board is None or actual_prefixed not in MOVE_TO_ID:
                continue

            multipv = self.rng.randint(self.multipv_min, self.multipv_max)
            infos = engine.analyse(board, self.limit, multipv=multipv)
            if isinstance(infos, dict):
                infos = [infos]

            pv_lines: list[list[str]] = []
            top_score_cp: int | None = None
            for info in infos:
                raw_pv = info.get("pv", [])
                if raw_pv:
                    move = raw_pv[0].uci()
                    pv_prefix = WHITE_PREFIX if board.turn == chess.WHITE else BLACK_PREFIX
                    prefixed_move = pv_prefix + move
                    if prefixed_move in MOVE_TO_ID:
                        pv_lines.append([move])
                if top_score_cp is None and "score" in info:
                    top_score_cp = info["score"].pov(board.turn).score(mate_score=100000)

            if len(pv_lines) != multipv:
                continue

            row = build_cot_row(
                result=int(result),
                prefix_moves=prefix_moves,
                pv_lines=pv_lines,
                actual_move=actual_move,
                suffix_moves=suffix_moves,
                depth=self.depth,
                stockfish_score_cp=top_score_cp,
                source_game_index=game_index,
            )
            if len(row["token_ids"]) > self.max_len:
                continue

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
    num_samples: int,
    multipv_min: int,
    multipv_max: int,
    stockfish_path: Path,
    depth: int,
    seed: int,
    max_len: int,
) -> list[dict[str, int | str | list[int] | None]]:
    """Generate a fixed offline set of CoT rows."""
    if num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    with OnlineCotDataSource(
        games=games,
        stockfish_path=stockfish_path,
        multipv_min=multipv_min,
        multipv_max=multipv_max,
        depth=depth,
        seed=seed,
        max_len=max_len,
    ) as source:
        rows: list[dict[str, int | str | list[int] | None]] = []
        with tqdm(total=num_samples, desc="generate-sft-cot", unit="sample") as progress:
            for _ in range(num_samples):
                rows.append(source.sample_row())
                progress.update(1)
        return rows
