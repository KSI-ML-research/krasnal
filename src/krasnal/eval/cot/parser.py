import bulletchess

from krasnal.inference.utils import get_legal_token_ids
from krasnal.tokens import (
    GAME_END_ID,
    GAME_START_ID,
    ID_TO_MOVE,
    THINK_END_ID,
    THINK_START_ID,
)


def parse_cot_sample(token_ids: list[int]) -> dict | None:
    """Extract CoT game data from token sequence."""
    if not token_ids or token_ids[0] != GAME_START_ID or token_ids[-1] != GAME_END_ID:
        return None

    try:
        think_start = token_ids.index(THINK_START_ID)
    except ValueError:
        return None

    prompt_tokens = token_ids[:think_start]
    if len(prompt_tokens) < 2 or prompt_tokens[0] != GAME_START_ID:
        return None

    board = bulletchess.Board()
    for token_id in token_ids[2:think_start]:
        if token_id in {THINK_START_ID, THINK_END_ID}:
            return None
        try:
            uci_move = ID_TO_MOVE[token_id][1:]
            move = bulletchess.Move.from_uci(uci_move)
        except Exception:
            return None
        board.apply(move)

    target_think_tokens: list[int] = []
    in_think = False
    last_think_end = None
    for idx, token_id in enumerate(token_ids[think_start:], start=think_start):
        if token_id == THINK_START_ID:
            if in_think:
                return None
            in_think = True
            continue
        if token_id == THINK_END_ID:
            if not in_think:
                return None
            in_think = False
            last_think_end = idx
            continue
        if in_think:
            target_think_tokens.append(token_id)

    if in_think or last_think_end is None:
        return None

    post_think_actual_token = None
    for token_id in token_ids[last_think_end + 1 : -1]:
        if token_id in {THINK_START_ID, THINK_END_ID}:
            continue
        post_think_actual_token = token_id
        break
    if post_think_actual_token is None:
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "target_think_tokens": target_think_tokens,
        "post_think_actual_token": post_think_actual_token,
        "post_think_legal_ids": get_legal_token_ids(board),
    }


def is_valid_cot_sequence(tokens: list[int]) -> bool:
    """Check if token sequence has valid CoT format."""
    if not tokens or tokens[0] != GAME_START_ID or tokens[-1] != GAME_END_ID:
        return False
    if tokens.count(GAME_START_ID) != 1 or tokens.count(GAME_END_ID) != 1:
        return False

    in_think = False
    think_span_len = 0
    think_span_count = 0
    for token_id in tokens[1:-1]:
        if token_id == THINK_START_ID:
            if in_think:
                return False
            in_think = True
            think_span_len = 0
            continue
        if token_id == THINK_END_ID:
            if not in_think or think_span_len == 0:
                return False
            in_think = False
            think_span_count += 1
            continue
        if in_think:
            think_span_len += 1
    return not in_think and think_span_count > 0


def extract_think_tokens(tokens: list[int]) -> list[int]:
    """Extract tokens between THINK_START and THINK_END."""
    think_tokens: list[int] = []
    in_think = False
    for token_id in tokens:
        if token_id == THINK_START_ID:
            in_think = True
            continue
        if token_id == THINK_END_ID:
            in_think = False
            continue
        if in_think:
            think_tokens.append(token_id)
    return think_tokens
