from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass

import chess
import chess.engine

from .tokenizer import (
    DRAW_ID,
    SOS_ID,
    SPECIAL_TOKENS,
    STEP_BACK_ID,
    THINK_END_ID,
    THINK_START_ID,
    WIN_BLACK_ID,
    WIN_WHITE_ID,
    Tokenizer,
)


@dataclass(frozen=True)
class StockfishCoTConfig:
    min_prefix: int = 4
    think_min: int = 8
    think_max: int = 16
    tail_len: int = 0
    max_seq_len: int = 256
    backtrack_prob: float = 0.15

    stockfish_path: str = "stockfish"
    stockfish_time: float = 0.02
    stockfish_depth: int = 0
    stockfish_nodes: int = 0
    multipv: int = 2
    threads: int = 1
    hash_mb: int = 128
    cache_size: int = 50000
    pv_temperature: float = 2.0
    pv_explore_prob: float = 0.35
    branch_prob: float = 0.35


class StockfishCoTGenerator:
    """Generate synthetic CoT samples from real games using Stockfish MultiPV.

    The engine process is kept alive for the whole preprocessing run and optional
    in-memory LRU caching avoids repeated analysis for identical prefixes.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: StockfishCoTConfig,
        engine: chess.engine.SimpleEngine | None = None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self._own_engine = engine is None
        self.engine = engine or chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self._cache: OrderedDict[str, list[list[str]]] = OrderedDict()

        # Engines differ in supported UCI options; best-effort configuration.
        options = {"Threads": max(1, config.threads), "Hash": max(1, config.hash_mb)}
        try:
            self.engine.configure(options)
        except Exception:
            pass

    def close(self) -> None:
        if self._own_engine:
            self.engine.quit()

    def __enter__(self) -> StockfishCoTGenerator:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def build_sample(self, seq: list[int], rng: random.Random) -> list[int] | None:
        cfg = self.config
        special_ids = set(SPECIAL_TOKENS)
        outcome_ids = {WIN_WHITE_ID, WIN_BLACK_ID, DRAW_ID, SOS_ID}
        outcome = seq[0] if seq and seq[0] in outcome_ids else SOS_ID
        moves = [t for t in seq if t not in special_ids]

        if len(moves) < cfg.min_prefix + 1:
            return None

        max_prefix = len(moves) - 1
        if max_prefix < cfg.min_prefix:
            return None

        prefix_len = rng.randint(cfg.min_prefix, max_prefix)
        board = chess.Board()
        for t in moves[:prefix_len]:
            uci = self.tokenizer.id_to_move.get(t, "")
            if not uci:
                return None
            try:
                board.push_uci(uci)
            except Exception:
                return None

        pv_lines = self._analyze_multipv(board)
        if not pv_lines:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            pv_lines = [[legal_moves[0].uci()]]

        chosen_line = self._pick_pv_line(pv_lines, rng)
        if not chosen_line:
            return None

        think_len = rng.randint(max(1, cfg.think_min), max(cfg.think_min, cfg.think_max))

        second_line: list[str] | None = None
        if len(pv_lines) > 1 and rng.random() < cfg.branch_prob:
            other_lines = [line for line in pv_lines if line != chosen_line]
            if other_lines:
                second_line = self._pick_pv_line(other_lines, rng)

        think_tokens = self._build_think_tokens(
            prefix_board=board,
            primary_line=chosen_line,
            secondary_line=second_line,
            think_len=think_len,
            rng=rng,
        )

        if not think_tokens:
            return None

        final_move_id = self.tokenizer.move_to_id.get(chosen_line[0])
        if final_move_id is None:
            return None

        tokens = (
            [outcome]
            + moves[:prefix_len]
            + [THINK_START_ID]
            + think_tokens
            + [THINK_END_ID]
            + [final_move_id]
        )

        if cfg.tail_len > 0 and len(chosen_line) > 1:
            tail_start = 1
            tail_end = min(tail_start + cfg.tail_len, len(chosen_line))
            for uci in chosen_line[tail_start:tail_end]:
                move_id = self.tokenizer.move_to_id.get(uci)
                if move_id is not None:
                    tokens.append(move_id)

        return tokens[: cfg.max_seq_len]

    def _analyze_multipv(self, board: chess.Board) -> list[list[str]]:
        key = board.fen()
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        limit = chess.engine.Limit(
            time=self.config.stockfish_time if self.config.stockfish_time > 0 else None,
            depth=self.config.stockfish_depth if self.config.stockfish_depth > 0 else None,
            nodes=self.config.stockfish_nodes if self.config.stockfish_nodes > 0 else None,
        )

        try:
            info = self.engine.analyse(board, limit=limit, multipv=max(1, self.config.multipv))
        except Exception:
            self._cache_put(key, [])
            return []

        infos = info if isinstance(info, list) else [info]
        lines: list[list[str]] = []
        for line_info in infos:
            pv = line_info.get("pv", [])
            ucis = [mv.uci() for mv in pv if mv is not None]
            if ucis:
                lines.append(ucis)

        self._cache_put(key, lines)
        return lines

    def _pick_pv_line(self, pv_lines: list[list[str]], rng: random.Random) -> list[str]:
        if len(pv_lines) == 1:
            return pv_lines[0]

        if rng.random() < max(0.0, min(1.0, self.config.pv_explore_prob)):
            return rng.choice(pv_lines)

        # Temperature-softened rank weighting: higher temp => more uniform sampling.
        temp = max(self.config.pv_temperature, 1e-6)
        weights = [math.exp(-(rank / temp)) for rank in range(len(pv_lines))]
        idx = rng.choices(range(len(pv_lines)), weights=weights, k=1)[0]
        return pv_lines[idx]

    def _build_think_tokens(
        self,
        prefix_board: chess.Board,
        primary_line: list[str],
        secondary_line: list[str] | None,
        think_len: int,
        rng: random.Random,
    ) -> list[int]:
        think_tokens: list[int] = []
        board = prefix_board.copy()

        if think_len <= 0:
            return think_tokens

        if secondary_line is None:
            self._append_line_moves(board, primary_line, 0, think_len, think_tokens, rng)
            return think_tokens

        primary_budget = rng.randint(1, max(1, think_len - 1))
        primary_played = self._append_line_moves(
            board, primary_line, 0, primary_budget, think_tokens, rng
        )

        branch_depth = min(self._common_prefix_len(primary_line, secondary_line), primary_played)
        back_steps = primary_played - branch_depth
        for _ in range(back_steps):
            if board.move_stack:
                think_tokens.append(STEP_BACK_ID)
                board.pop()

        remaining_budget = think_len - primary_played
        self._append_line_moves(
            board,
            secondary_line,
            branch_depth,
            max(0, remaining_budget),
            think_tokens,
            rng,
        )
        return think_tokens

    def _append_line_moves(
        self,
        board: chess.Board,
        line: list[str],
        start_idx: int,
        budget: int,
        think_tokens: list[int],
        rng: random.Random,
    ) -> int:
        played = 0
        if budget <= 0:
            return played

        for uci in line[start_idx:]:
            if played >= budget:
                break
            move_id = self.tokenizer.move_to_id.get(uci)
            if move_id is None:
                continue
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move not in board.legal_moves:
                break

            think_tokens.append(move_id)
            board.push(move)
            played += 1

            # Keep some local revision noise for robustness even in linear branches.
            if rng.random() < self.config.backtrack_prob and board.move_stack:
                think_tokens.append(STEP_BACK_ID)
                board.pop()
                think_tokens.append(move_id)
                board.push(move)

        return played

    @staticmethod
    def _common_prefix_len(line_a: list[str], line_b: list[str]) -> int:
        n = min(len(line_a), len(line_b))
        i = 0
        while i < n and line_a[i] == line_b[i]:
            i += 1
        return i

    def _cache_get(self, key: str) -> list[list[str]] | None:
        value = self._cache.get(key)
        if value is None:
            return None
        self._cache.move_to_end(key)
        return value

    def _cache_put(self, key: str, value: list[list[str]]) -> None:
        if self.config.cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)
