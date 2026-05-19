from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import bulletchess

from krasnal.config import MOVE_VOCAB_PATH

GAME_START_ID = 0
GAME_END_ID = 1
PAD_ID = 2

WHITE_WON_ID = 3
BLACK_WON_ID = 4
DRAW_ID = 5
UNKNOWN_RESULT_ID = 6

ELO_BELOW_1000_ID = 9
ELO_1000_1099_ID = 10
ELO_1100_1199_ID = 11
ELO_1200_1299_ID = 12
ELO_1300_1399_ID = 13
ELO_1400_1499_ID = 14
ELO_1500_1599_ID = 15
ELO_1600_1699_ID = 16
ELO_1700_1799_ID = 17
ELO_1800_1899_ID = 18
ELO_1900_1999_ID = 19
ELO_2000_2099_ID = 20
ELO_2100_2199_ID = 21
ELO_ABOVE_2200_ID = 22
ELO_UNKNOWN_ID = 23

TC_BLITZ_NO_INC_ID = 111
TC_BLITZ_INC_ID = 112
TC_RAPID_NO_INC_ID = 113
TC_RAPID_INC_ID = 114
TC_CLASSICAL_ID = 115
TC_UNKNOWN_ID = 116

IS_CHECK_ID = 24
YES_CHECK_ID = 25
NO_CHECK_ID = 26
PIECE_TYPE_MOVED_ID = 27
PAWN_ID = 28
KNIGHT_ID = 29
BISHOP_ID = 30
ROOK_ID = 31
QUEEN_ID = 32
KING_ID = 33
EMPTY_ID = 34

WHATS_ON_SQUARE = {
    f"<whats_on_{chr(f + 97)}{r + 1}>": 35 + r * 8 + f for r in range(8) for f in range(8)
}
WHATS_ON_SQUARE_TOKEN_IDS = frozenset(WHATS_ON_SQUARE.values())

COLORED_PIECE_TOKENS = {}
_next_id = 99
for _color in ["w", "b"]:
    for _piece in ["pawn", "knight", "bishop", "rook", "queen", "king"]:
        COLORED_PIECE_TOKENS[f"<{_color}:{_piece}>"] = _next_id
        _next_id += 1

WHITE_PREFIX = "w:"
BLACK_PREFIX = "b:"
PIECE_AWARE_MOVES_DEFAULT: Final[bool] = False
SIDE_PREFIXED_MOVES_DEFAULT: Final[bool] = True
PIECE_AWARE_MOVES = PIECE_AWARE_MOVES_DEFAULT
SIDE_PREFIXED_MOVES = SIDE_PREFIXED_MOVES_DEFAULT

PIECE_TYPE_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "pawn": ("pawn", "p"),
    "knight": ("knight", "n"),
    "bishop": ("bishop", "b"),
    "rook": ("rook", "r"),
    "queen": ("queen", "q"),
    "king": ("king", "k"),
}
PIECE_TYPES: Final[tuple[str, ...]] = tuple(PIECE_TYPE_ALIASES)
PIECE_TYPE_LOOKUP: Final[dict[str, str]] = {
    alias: canonical for canonical, aliases in PIECE_TYPE_ALIASES.items() for alias in aliases
}

OUTCOME_TOKENS = {
    "<white_won>": WHITE_WON_ID,
    "<black_won>": BLACK_WON_ID,
    "<draw>": DRAW_ID,
    "<result_unknown>": UNKNOWN_RESULT_ID,
}

ELO_TOKENS = {
    "<elo_below_1000>": ELO_BELOW_1000_ID,
    "<elo_1000_1099>": ELO_1000_1099_ID,
    "<elo_1100_1199>": ELO_1100_1199_ID,
    "<elo_1200_1299>": ELO_1200_1299_ID,
    "<elo_1300_1399>": ELO_1300_1399_ID,
    "<elo_1400_1499>": ELO_1400_1499_ID,
    "<elo_1500_1599>": ELO_1500_1599_ID,
    "<elo_1600_1699>": ELO_1600_1699_ID,
    "<elo_1700_1799>": ELO_1700_1799_ID,
    "<elo_1800_1899>": ELO_1800_1899_ID,
    "<elo_1900_1999>": ELO_1900_1999_ID,
    "<elo_2000_2099>": ELO_2000_2099_ID,
    "<elo_2100_2199>": ELO_2100_2199_ID,
    "<elo_above_2200>": ELO_ABOVE_2200_ID,
    "<elo_unknown>": ELO_UNKNOWN_ID,
}

TC_TOKENS = {
    "<tc_blitz_no_inc>": TC_BLITZ_NO_INC_ID,
    "<tc_blitz_inc>": TC_BLITZ_INC_ID,
    "<tc_rapid_no_inc>": TC_RAPID_NO_INC_ID,
    "<tc_rapid_inc>": TC_RAPID_INC_ID,
    "<tc_classical>": TC_CLASSICAL_ID,
    "<tc_unknown>": TC_UNKNOWN_ID,
}

ELO_BUCKETS = {
    ELO_BELOW_1000_ID: "below_1000",
    ELO_1000_1099_ID: "1000_1099",
    ELO_1100_1199_ID: "1100_1199",
    ELO_1200_1299_ID: "1200_1299",
    ELO_1300_1399_ID: "1300_1399",
    ELO_1400_1499_ID: "1400_1499",
    ELO_1500_1599_ID: "1500_1599",
    ELO_1600_1699_ID: "1600_1699",
    ELO_1700_1799_ID: "1700_1799",
    ELO_1800_1899_ID: "1800_1899",
    ELO_1900_1999_ID: "1900_1999",
    ELO_2000_2099_ID: "2000_2099",
    ELO_2100_2199_ID: "2100_2199",
    ELO_ABOVE_2200_ID: "above_2200",
    ELO_UNKNOWN_ID: "unknown",
}

# Loss-mask targets: model always receives result, Elo, and time control as a fixed prefix.
CONDITIONING_METADATA_TARGET_MASK_IDS: Final[frozenset[int]] = frozenset(
    (*OUTCOME_TOKENS.values(), *ELO_TOKENS.values(), *TC_TOKENS.values())
)


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
    **TC_TOKENS,
}

MOVE_TO_ID = dict(SPECIAL_TOKENS)
ID_TO_MOVE = {v: k for k, v in MOVE_TO_ID.items()}
VOCAB_SIZE = len(MOVE_TO_ID)
MOVE_VOCAB_MANIFEST: dict[str, Any] | None = None
MOVE_VOCAB_SOURCE_PATH: Path | None = None


def _now_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def normalize_piece_type(piece_type: object) -> str:
    if piece_type is None:
        raise ValueError("piece_moved is missing")
    normalized = str(piece_type).strip().lower()
    if not normalized:
        raise ValueError("piece_moved is empty")
    try:
        return PIECE_TYPE_LOOKUP[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported piece_moved value: {piece_type!r}") from exc


def _side_prefix_for(ply: int | None, turn: object | None) -> str:
    if ply is not None and turn is not None:
        raise ValueError("Provide either ply or turn, not both")
    if ply is not None:
        if ply < 0:
            raise ValueError(f"ply must be >= 0, got {ply}")
        return WHITE_PREFIX if ply % 2 == 0 else BLACK_PREFIX
    if turn is None:
        raise ValueError("side-prefixed move keys require ply or turn")

    turn_str = str(turn).strip().lower()
    if turn_str == "white":
        return WHITE_PREFIX
    if turn_str == "black":
        return BLACK_PREFIX
    raise ValueError(f"Unsupported board turn: {turn!r}")


def build_move_key(
    uci: str,
    mover_piece_type: object,
    *,
    ply: int | None = None,
    turn: object | None = None,
    piece_aware_moves: bool | None = None,
    side_prefixed_moves: bool | None = None,
) -> str:
    key = str(uci).strip()
    if not key:
        raise ValueError("uci move is empty")

    use_piece_aware_moves = (
        PIECE_AWARE_MOVES if piece_aware_moves is None else bool(piece_aware_moves)
    )
    use_side_prefix = (
        SIDE_PREFIXED_MOVES if side_prefixed_moves is None else bool(side_prefixed_moves)
    )

    if use_piece_aware_moves:
        key = f"{normalize_piece_type(mover_piece_type)}:{key}"
    if use_side_prefix:
        key = f"{_side_prefix_for(ply, turn)}{key}"
    return key


def _build_vocabulary(move_keys: Iterable[str]) -> dict[str, int]:
    vocab = dict(SPECIAL_TOKENS)
    next_id = max(SPECIAL_TOKENS.values()) + 1
    for token in sorted(set(move_keys)):
        if token in vocab:
            raise ValueError(f"Move token collides with special token: {token}")
        vocab[token] = next_id
        next_id += 1
    return vocab


def make_move_vocab_artifact(
    move_keys: Iterable[str],
    *,
    piece_aware_moves: bool,
    side_prefixed_moves: bool,
    generation_timestamp: str | None = None,
) -> dict[str, Any]:
    vocab = _build_vocabulary(move_keys)
    return {
        "manifest": {
            "piece_aware_moves": bool(piece_aware_moves),
            "side_prefixed_moves": bool(side_prefixed_moves),
            "generation_timestamp": generation_timestamp or _now_timestamp(),
            "vocab_size": len(vocab),
        },
        "vocab": vocab,
    }


def _validate_move_vocab_artifact(artifact: object) -> tuple[dict[str, Any], dict[str, int]]:
    if not isinstance(artifact, dict):
        raise ValueError("move_vocab.json must contain a JSON object")
    if set(artifact) != {"manifest", "vocab"}:
        raise ValueError("move_vocab.json must contain exactly 'manifest' and 'vocab'")

    manifest = artifact["manifest"]
    vocab = artifact["vocab"]
    if not isinstance(manifest, dict):
        raise ValueError("move_vocab.json manifest must be an object")
    if not isinstance(vocab, dict):
        raise ValueError("move_vocab.json vocab must be an object")

    required_manifest_keys = {
        "piece_aware_moves",
        "side_prefixed_moves",
        "generation_timestamp",
        "vocab_size",
    }
    if set(manifest) != required_manifest_keys:
        raise ValueError(
            "move_vocab.json manifest must contain only: "
            f"{', '.join(sorted(required_manifest_keys))}"
        )

    normalized_vocab: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in vocab.items():
        if not isinstance(token, str):
            raise ValueError(f"Vocabulary token must be a string, got {token!r}")
        if not isinstance(token_id, int):
            raise ValueError(f"Vocabulary id for {token!r} must be an integer")
        if token_id in seen_ids:
            raise ValueError(f"Duplicate vocabulary id: {token_id}")
        seen_ids.add(token_id)
        normalized_vocab[token] = token_id

    for token, token_id in SPECIAL_TOKENS.items():
        if normalized_vocab.get(token) != token_id:
            raise ValueError(f"Vocabulary has invalid id for special token {token!r}")

    manifest_vocab_size = manifest["vocab_size"]
    if not isinstance(manifest_vocab_size, int):
        raise ValueError("move_vocab.json manifest vocab_size must be an integer")
    if manifest_vocab_size != len(normalized_vocab):
        raise ValueError(
            "move_vocab.json manifest vocab_size does not match vocab length: "
            f"{manifest_vocab_size} != {len(normalized_vocab)}"
        )
    if not isinstance(manifest["piece_aware_moves"], bool):
        raise ValueError("move_vocab.json manifest piece_aware_moves must be a boolean")
    if not isinstance(manifest["side_prefixed_moves"], bool):
        raise ValueError("move_vocab.json manifest side_prefixed_moves must be a boolean")
    if not isinstance(manifest["generation_timestamp"], str):
        raise ValueError("move_vocab.json manifest generation_timestamp must be a string")

    return dict(manifest), normalized_vocab


def _validate_manifest_match(
    manifest: dict[str, Any],
    *,
    piece_aware_moves: bool,
    side_prefixed_moves: bool,
) -> None:
    expected = {
        "piece_aware_moves": bool(piece_aware_moves),
        "side_prefixed_moves": bool(side_prefixed_moves),
    }
    actual = {
        "piece_aware_moves": manifest["piece_aware_moves"],
        "side_prefixed_moves": manifest["side_prefixed_moves"],
    }
    if actual != expected:
        raise ValueError(
            "move_vocab.json manifest does not match runtime config: "
            f"expected {expected}, found {actual}"
        )


def install_move_vocab_artifact(
    artifact: object,
    *,
    source_path: Path | None = None,
    piece_aware_moves: bool | None = None,
    side_prefixed_moves: bool | None = None,
    require_manifest_match: bool = True,
) -> None:
    manifest, vocab = _validate_move_vocab_artifact(artifact)
    expected_piece_aware_moves = (
        manifest["piece_aware_moves"] if piece_aware_moves is None else bool(piece_aware_moves)
    )
    expected_side_prefixed = (
        manifest["side_prefixed_moves"]
        if side_prefixed_moves is None
        else bool(side_prefixed_moves)
    )
    if require_manifest_match:
        _validate_manifest_match(
            manifest,
            piece_aware_moves=expected_piece_aware_moves,
            side_prefixed_moves=expected_side_prefixed,
        )

    global PIECE_AWARE_MOVES, SIDE_PREFIXED_MOVES, VOCAB_SIZE, MOVE_VOCAB_MANIFEST
    global MOVE_VOCAB_SOURCE_PATH

    PIECE_AWARE_MOVES = bool(manifest["piece_aware_moves"])
    SIDE_PREFIXED_MOVES = bool(manifest["side_prefixed_moves"])
    MOVE_TO_ID.clear()
    MOVE_TO_ID.update(vocab)
    ID_TO_MOVE.clear()
    ID_TO_MOVE.update({v: k for k, v in vocab.items()})
    VOCAB_SIZE = len(MOVE_TO_ID)
    MOVE_VOCAB_MANIFEST = manifest
    MOVE_VOCAB_SOURCE_PATH = source_path


def load_move_vocab(
    path: Path = MOVE_VOCAB_PATH,
    *,
    piece_aware_moves: bool,
    side_prefixed_moves: bool,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Move vocabulary not found at {path}. Run preprocessing to generate it."
        )
    with path.open() as f:
        artifact = json.load(f)
    install_move_vocab_artifact(
        artifact,
        source_path=path,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
        require_manifest_match=True,
    )


def save_move_vocab(
    path: Path,
    move_keys: Iterable[str],
    *,
    piece_aware_moves: bool,
    side_prefixed_moves: bool,
) -> dict[str, Any]:
    artifact = make_move_vocab_artifact(
        move_keys,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")
    return artifact


def get_vocab_size() -> int:
    return VOCAB_SIZE


def get_elo_bucket(elo: int) -> int:
    if elo < 1000:
        return ELO_BELOW_1000_ID
    if elo >= 2200:
        return ELO_ABOVE_2200_ID

    # Map [1000, 2199] to 100-point buckets
    bucket_index = (elo - 1000) // 100
    # The IDs are sequential starting from 10
    return 10 + bucket_index


def get_time_control_bucket(
    time_initial: int | float | None,
    time_increment: int | float | None,
) -> int:
    if time_initial is None or time_increment is None:
        return TC_UNKNOWN_ID

    initial = int(time_initial)
    increment = int(time_increment)
    estimated_duration = initial + 40 * increment

    if estimated_duration < 480:
        return TC_BLITZ_NO_INC_ID if increment == 0 else TC_BLITZ_INC_ID
    if estimated_duration < 1500:
        return TC_RAPID_NO_INC_ID if increment == 0 else TC_RAPID_INC_ID
    return TC_CLASSICAL_ID


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


def get_moves_only(token_ids: list[int]) -> list[int]:
    return [tid for tid in token_ids if tid not in SPECIAL_TOKENS.values()]


def _token_string_to_uci(token: str) -> str:
    if token.startswith(WHITE_PREFIX):
        token = token[len(WHITE_PREFIX) :]
    elif token.startswith(BLACK_PREFIX):
        token = token[len(BLACK_PREFIX) :]

    piece_prefix, sep, uci = token.partition(":")
    if sep and piece_prefix in PIECE_TYPES:
        return uci
    return token


def to_uci(token_id: int) -> str:
    token = ID_TO_MOVE.get(token_id, "")
    return _token_string_to_uci(token)


def move_key_for_ply(
    uci: str,
    ply: int,
    mover_piece_type: object | None = None,
    *,
    piece_aware_moves: bool | None = None,
    side_prefixed_moves: bool | None = None,
) -> str:
    return build_move_key(
        uci,
        mover_piece_type,
        ply=ply,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )


def move_key_for_turn(
    uci: str,
    turn: object,
    mover_piece_type: object | None = None,
    *,
    piece_aware_moves: bool | None = None,
    side_prefixed_moves: bool | None = None,
) -> str:
    return build_move_key(
        uci,
        mover_piece_type,
        turn=turn,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )


def move_token_id_for_ply(
    uci: str,
    ply: int,
    mover_piece_type: object | None = None,
) -> int | None:
    return MOVE_TO_ID.get(move_key_for_ply(uci, ply, mover_piece_type))


def move_token_id_for_turn(
    uci: str,
    turn: object,
    mover_piece_type: object | None = None,
) -> int | None:
    return MOVE_TO_ID.get(move_key_for_turn(uci, turn, mover_piece_type))


def token_to_uci(token_id: int) -> str | None:
    token = ID_TO_MOVE.get(token_id)
    if token is None:
        return None
    return _token_string_to_uci(token)


def uci_to_token_id(
    uci: str,
    turn: object,
    mover_piece_type: object | None = None,
) -> int | None:
    return move_token_id_for_turn(uci, turn, mover_piece_type)


def _piece_type_for_board_move(board: bulletchess.Board, move: bulletchess.Move) -> str:
    piece = board[move.origin]
    if piece is None:
        raise ValueError(f"No piece on move origin for legal move {move.uci()}")
    return normalize_piece_type(piece.piece_type)


def legal_token_ids(board: bulletchess.Board) -> list[int]:
    token_ids: list[int] = []
    for move in board.legal_moves():
        uci = move.uci()
        if not uci:
            continue
        piece_type = _piece_type_for_board_move(board, move)
        token_id = move_token_id_for_turn(uci, board.turn, piece_type)
        if token_id is not None:
            token_ids.append(token_id)
    return token_ids
