from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import bulletchess

from krasnal.config import MOVES_FILE

GAME_START_ID = 0
GAME_END_ID = 1
PAD_ID = 2

WHITE_WON_ID = 3
BLACK_WON_ID = 4
DRAW_ID = 5
UNKNOWN_RESULT_ID = 6

THINK_START_ID = 7
THINK_END_ID = 8

ELO_BELOW_1000_ID = 9
ELO_1000_1499_ID = 10
ELO_1500_1999_ID = 11
ELO_2000_2499_ID = 12
ELO_2500_2999_ID = 13
ELO_ABOVE_3000_ID = 14
ELO_UNKNOWN_ID = 15

IS_CHECK_ID = 16
YES_CHECK_ID = 17
NO_CHECK_ID = 18
PIECE_TYPE_MOVED_ID = 19
PAWN_ID = 20
KNIGHT_ID = 21
BISHOP_ID = 22
ROOK_ID = 23
QUEEN_ID = 24
KING_ID = 25
EMPTY_ID = 26

WHATS_ON_SQUARE = {
    f"<whats_on_{chr(f + 97)}{r + 1}>": 27 + r * 8 + f for r in range(8) for f in range(8)
}
WHATS_ON_SQUARE_TOKEN_IDS = frozenset(WHATS_ON_SQUARE.values())

COLORED_PIECE_TOKENS = {}
_next_id = 91
for _color in ["w", "b"]:
    for _piece in ["pawn", "knight", "bishop", "rook", "queen", "king"]:
        COLORED_PIECE_TOKENS[f"<{_color}:{_piece}>"] = _next_id
        _next_id += 1

WHITE_PREFIX = "w:"
BLACK_PREFIX = "b:"
SIDE_PREFIXED_MOVES_DEFAULT: Final[bool] = True
SIDE_PREFIXED_MOVES = SIDE_PREFIXED_MOVES_DEFAULT

OUTCOME_TOKENS = {
    "<white_won>": WHITE_WON_ID,
    "<black_won>": BLACK_WON_ID,
    "<draw>": DRAW_ID,
    "<result_unknown>": UNKNOWN_RESULT_ID,
}

ELO_TOKENS = {
    "<elo_below_1000>": ELO_BELOW_1000_ID,
    "<elo_1000_1499>": ELO_1000_1499_ID,
    "<elo_1500_1999>": ELO_1500_1999_ID,
    "<elo_2000_2499>": ELO_2000_2499_ID,
    "<elo_2500_2999>": ELO_2500_2999_ID,
    "<elo_above_3000>": ELO_ABOVE_3000_ID,
    "<elo_unknown>": ELO_UNKNOWN_ID,
}

# Loss-mask targets: model always receives result + Elo as a fixed prefix at inference.
CONDITIONING_METADATA_TARGET_MASK_IDS: Final[frozenset[int]] = frozenset(
    (*OUTCOME_TOKENS.values(), *ELO_TOKENS.values())
)

THINKING_TOKENS = {
    "<think_start>": THINK_START_ID,
    "<think_end>": THINK_END_ID,
}

QA_TOKENS = {
    "<is_check>": IS_CHECK_ID,
    "<yes_check>": YES_CHECK_ID,
    "<no_check>": NO_CHECK_ID,
    "<piece_type_moved>": PIECE_TYPE_MOVED_ID,
    "<pawn>": PAWN_ID,
    "<knight>": KNIGHT_ID,
    "<bishop>": BISHOP_ID,
    "<rook>": ROOK_ID,
    "<queen>": QUEEN_ID,
    "<king>": KING_ID,
    "<empty>": EMPTY_ID,
    **WHATS_ON_SQUARE,
    **COLORED_PIECE_TOKENS,
}
WHATS_ON_PROMPT_TOKEN_IDS = WHATS_ON_SQUARE_TOKEN_IDS

QA_TOKEN_IDS = frozenset(QA_TOKENS.values())

SPECIAL_TOKENS = {
    "<game_start>": GAME_START_ID,
    "<game_end>": GAME_END_ID,
    "<pad>": PAD_ID,
    **QA_TOKENS,
    **OUTCOME_TOKENS,
    **ELO_TOKENS,
    **THINKING_TOKENS,
}


def _load_vocabulary(*, side_prefixed_moves: bool) -> tuple[dict[str, int], dict[int, str]]:
    move_to_id = dict(SPECIAL_TOKENS)

    with open(MOVES_FILE) as f:
        all_uci_moves = [line.strip() for line in f if line.strip()]

    next_id = max(SPECIAL_TOKENS.values()) + 1
    for move in all_uci_moves:
        if side_prefixed_moves:
            move_to_id[WHITE_PREFIX + move] = next_id
            next_id += 1
            move_to_id[BLACK_PREFIX + move] = next_id
        else:
            move_to_id[move] = next_id
        next_id += 1

    id_to_move = {v: k for k, v in move_to_id.items()}
    return move_to_id, id_to_move


MOVE_TO_ID, ID_TO_MOVE = _load_vocabulary(side_prefixed_moves=SIDE_PREFIXED_MOVES)
VOCAB_SIZE = len(MOVE_TO_ID)


def set_side_prefixed_moves(enabled: bool) -> None:
    global SIDE_PREFIXED_MOVES, VOCAB_SIZE
    SIDE_PREFIXED_MOVES = bool(enabled)
    move_to_id, id_to_move = _load_vocabulary(side_prefixed_moves=SIDE_PREFIXED_MOVES)
    MOVE_TO_ID.clear()
    MOVE_TO_ID.update(move_to_id)
    ID_TO_MOVE.clear()
    ID_TO_MOVE.update(id_to_move)
    VOCAB_SIZE = len(MOVE_TO_ID)


def get_vocab_size() -> int:
    return VOCAB_SIZE


def get_elo_bucket(elo: int) -> int:
    if elo < 1000:
        return ELO_BELOW_1000_ID
    if elo < 1500:
        return ELO_1000_1499_ID
    if elo < 2000:
        return ELO_1500_1999_ID
    if elo < 2500:
        return ELO_2000_2499_ID
    if elo < 3000:
        return ELO_2500_2999_ID
    return ELO_ABOVE_3000_ID


def result_to_token_id(result: str | int) -> int:
    if result in (1, "1-0", "white_won", WHITE_WON_ID):
        return WHITE_WON_ID
    if result in (-1, "0-1", "black_won", BLACK_WON_ID):
        return BLACK_WON_ID
    if result in (0, "1/2-1/2", "draw", DRAW_ID):
        return DRAW_ID
    if result in ("result_unknown", UNKNOWN_RESULT_ID):
        return UNKNOWN_RESULT_ID
    raise ValueError(f"Unsupported game result: {result!r}")


def save_to_json(path: Path) -> None:
    with open(path, "w") as f:
        json.dump(MOVE_TO_ID, f)


def get_moves_only(token_ids: list[int]) -> list[int]:
    moves: list[int] = []
    in_think = False
    for token_id in token_ids:
        if token_id == THINK_START_ID:
            in_think = True
            continue
        if token_id == THINK_END_ID:
            in_think = False
            continue
        if in_think:
            continue
        if token_id not in SPECIAL_TOKENS.values():
            moves.append(token_id)
    return moves


def to_uci(token_id: int) -> str:
    token = ID_TO_MOVE.get(token_id, "")
    if token.startswith(WHITE_PREFIX):
        return token[len(WHITE_PREFIX) :]
    if token.startswith(BLACK_PREFIX):
        return token[len(BLACK_PREFIX) :]
    return token


def move_key_for_ply(uci: str, ply: int) -> str:
    if not SIDE_PREFIXED_MOVES:
        return uci
    prefix = WHITE_PREFIX if ply % 2 == 0 else BLACK_PREFIX
    return prefix + uci


def move_key_for_turn(uci: str, turn: object) -> str:
    if not SIDE_PREFIXED_MOVES:
        return uci
    prefix = WHITE_PREFIX if str(turn) == "White" else BLACK_PREFIX
    return prefix + uci


def move_token_id_for_ply(uci: str, ply: int) -> int | None:
    return MOVE_TO_ID.get(move_key_for_ply(uci, ply))


def move_token_id_for_turn(uci: str, turn: object) -> int | None:
    return MOVE_TO_ID.get(move_key_for_turn(uci, turn))


def token_to_uci(token_id: int) -> str | None:
    token = ID_TO_MOVE.get(token_id)
    if token is None:
        return None
    return to_uci(token_id)


def uci_to_token_id(uci: str, turn: object) -> int | None:
    return move_token_id_for_turn(uci, turn)


def legal_token_ids(board: bulletchess.Board) -> list[int]:
    return [
        token_id
        for move in board.legal_moves()
        if (uci := move.uci()) and (token_id := move_token_id_for_turn(uci, board.turn)) is not None
    ]
