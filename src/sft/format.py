from __future__ import annotations

from tokenizer import Tokenizer


def result_to_token_id(result: int, tokenizer: Tokenizer) -> int:
    """Map a game result label to its special token."""
    if result == 1:
        return tokenizer.win_white_id
    if result == -1:
        return tokenizer.win_black_id
    return tokenizer.draw_id


def serialize_pv_tokens(tokenizer: Tokenizer, pv_lines: list[list[str]]) -> list[int]:
    """Flatten PV lines into a `<branch>`-separated token list."""
    token_ids: list[int] = []
    for index, pv in enumerate(pv_lines):
        if index > 0:
            token_ids.append(tokenizer.step_back_id)
        token_ids.extend(tokenizer.move_to_id[move] for move in pv)
    return token_ids


def build_cot_sequence(
    *,
    tokenizer: Tokenizer,
    result: int,
    prefix_moves: list[str],
    pv_lines: list[list[str]],
    actual_move: str,
) -> list[int]:
    """Build one CoT training sequence."""
    token_ids = [result_to_token_id(result, tokenizer)]
    token_ids.extend(tokenizer.move_to_id[move] for move in prefix_moves)
    token_ids.append(tokenizer.think_start_id)
    token_ids.extend(serialize_pv_tokens(tokenizer, pv_lines))
    token_ids.append(tokenizer.think_end_id)
    token_ids.append(tokenizer.move_to_id[actual_move])
    token_ids.append(tokenizer.eos_id)
    return token_ids


def build_cot_row(
    *,
    tokenizer: Tokenizer,
    result: int,
    prefix_moves: list[str],
    pv_lines: list[list[str]],
    actual_move: str,
    depth: int | None,
    movetime_ms: int | None,
    stockfish_score_cp: int | None,
    source_game_index: int,
) -> dict[str, int | str | list[int] | None]:
    """Build one persisted CoT sample row."""
    return {
        "token_ids": build_cot_sequence(
            tokenizer=tokenizer,
            result=result,
            prefix_moves=prefix_moves,
            pv_lines=pv_lines,
            actual_move=actual_move,
        ),
        "prefix_len": 1 + len(prefix_moves),
        "target_move": actual_move,
        "pv_count": len(pv_lines),
        "depth": depth,
        "movetime_ms": movetime_ms,
        "stockfish_score_cp": stockfish_score_cp,
        "source_game_index": source_game_index,
    }
