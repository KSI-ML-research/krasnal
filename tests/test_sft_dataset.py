import random
import sys
from pathlib import Path
from unittest.mock import MagicMock

import chess
import pytest

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import MOVES_FILE
from src.sft_cot_generator import StockfishCoTConfig, StockfishCoTGenerator
from src.tokenizer import (
    SPECIAL_TOKENS,
    STEP_BACK_ID,
    THINK_END_ID,
    THINK_START_ID,
    WIN_WHITE_ID,
    Tokenizer,
)

SPECIAL_IDS = set(SPECIAL_TOKENS)


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer(MOVES_FILE)


def _make_pv(board, depth=4):
    """Build a proper alternating PV (each move legal at that point)."""
    pv, b = [], board.copy()
    for _ in range(depth):
        legal = list(b.legal_moves)
        if not legal:
            break
        mv = legal[0]
        pv.append(mv)
        b.push(mv)
    return pv


def _engine():
    """Mock engine that returns a proper alternating PV without Stockfish."""
    mock = MagicMock()
    mock.analyse.side_effect = lambda board, **_kw: {"pv": _make_pv(board)}
    return mock


def _seq(tokenizer, ucis):
    return [WIN_WHITE_ID] + [tokenizer.move_to_id[uci] for uci in ucis]


def _assert_legal_think_tree(result, tokenizer):
    """Replay the think section and assert every move was legal at that point."""
    ts = result.index(THINK_START_ID)
    te = result.index(THINK_END_ID)
    assert ts < te

    # Reconstruct the board at the end of the prefix (everything before <think>)
    board = chess.Board()
    for tid in result[:ts]:
        if tid not in SPECIAL_IDS:
            board.push_uci(tokenizer.id_to_move[tid])

    for tok in result[ts + 1 : te]:
        if tok == STEP_BACK_ID:
            assert board.move_stack, "<back> with empty move stack"
            board.pop()
        else:
            move = chess.Move.from_uci(tokenizer.id_to_move[tok])
            assert move in board.legal_moves, (
                f"Illegal move {tokenizer.id_to_move[tok]} in <think> block"
            )
            board.push(move)


def test_think_block_is_legal_search_tree(tokenizer):
    seq = _seq(tokenizer, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"])
    cfg = StockfishCoTConfig(
        min_prefix=2,
        think_min=2,
        think_max=4,
        tail_len=0,
        max_seq_len=256,
        backtrack_prob=0.0,
    )
    generator = StockfishCoTGenerator(tokenizer, cfg, engine=_engine())
    result = generator.build_sample(
        seq,
        random.Random(0),
    )
    assert result is not None
    assert THINK_START_ID in result
    assert THINK_END_ID in result
    _assert_legal_think_tree(result, tokenizer)


def test_think_block_with_backtracking(tokenizer):
    seq = _seq(tokenizer, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"])
    cfg = StockfishCoTConfig(
        min_prefix=2,
        think_min=3,
        think_max=3,
        tail_len=0,
        max_seq_len=256,
        backtrack_prob=1.0,
    )
    generator = StockfishCoTGenerator(tokenizer, cfg, engine=_engine())
    result = generator.build_sample(
        seq,
        random.Random(0),
    )
    assert result is not None
    assert STEP_BACK_ID in result
    _assert_legal_think_tree(result, tokenizer)
