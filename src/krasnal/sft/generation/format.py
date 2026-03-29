from __future__ import annotations

from krasnal.tokens import (
    BLACK_PREFIX,
    GAME_END_ID,
    GAME_START_ID,
    MOVE_TO_ID,
    THINK_END_ID,
    THINK_START_ID,
    WHITE_PREFIX,
    result_to_token_id,
)


def _get_prefixed_move(move: str, ply: int) -> str:
    prefix = WHITE_PREFIX if ply % 2 == 0 else BLACK_PREFIX
    return prefix + move


def flatten_multipv_moves(pv_lines: list[list[str]], start_ply: int = 0) -> list[int]:
    """Flatten MultiPV root lines into consecutive move tokens."""
    token_ids: list[int] = []
    for pv in pv_lines:
        for ply, move in enumerate(pv):
            token_ids.append(MOVE_TO_ID[_get_prefixed_move(move, start_ply + ply)])
    return token_ids


def build_cot_sequence(
    *,
    result: int,
    prefix_moves: list[str],
    pv_lines: list[list[str]],
    actual_move: str,
    suffix_moves: list[str],
) -> list[int]:
    """Build one CoT training sequence."""
    token_ids = [GAME_START_ID, result_to_token_id(result)]
    prefix_len = len(prefix_moves)
    token_ids.extend(
        MOVE_TO_ID[_get_prefixed_move(move, ply)] for ply, move in enumerate(prefix_moves)
    )
    token_ids.append(THINK_START_ID)
    pv_start_ply = prefix_len
    token_ids.extend(flatten_multipv_moves(pv_lines, pv_start_ply))
    token_ids.append(THINK_END_ID)
    actual_ply = pv_start_ply + sum(len(pv) for pv in pv_lines)
    token_ids.append(MOVE_TO_ID[_get_prefixed_move(actual_move, actual_ply)])
    token_ids.extend(
        MOVE_TO_ID[_get_prefixed_move(move, ply)]
        for ply, move in enumerate(suffix_moves, start=actual_ply + 1)
    )
    token_ids.append(GAME_END_ID)
    return token_ids


def build_cot_row(
    *,
    result: int,
    prefix_moves: list[str],
    pv_lines: list[list[str]],
    actual_move: str,
    suffix_moves: list[str],
    depth: int,
    stockfish_score_cp: int | None,
    source_game_index: int,
) -> dict[str, int | str | list[int] | None]:
    """Build one persisted CoT sample row."""
    return {
        "token_ids": build_cot_sequence(
            result=result,
            prefix_moves=prefix_moves,
            pv_lines=pv_lines,
            actual_move=actual_move,
            suffix_moves=suffix_moves,
        ),
        "prefix_len": 2 + len(prefix_moves),
        "target_move": actual_move,
        "pv_count": len(pv_lines),
        "depth": depth,
        "stockfish_score_cp": stockfish_score_cp,
        "source_game_index": source_game_index,
    }
