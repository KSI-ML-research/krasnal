from __future__ import annotations

from dataclasses import dataclass, field

import bulletchess

from krasnal.tokens import (
    DRAW_ID,
    ELO_UNKNOWN_ID,
    GAME_START_ID,
    OUTCOME_TOKENS,
    TC_UNKNOWN_ID,
    normalize_piece_type,
    token_to_uci,
    uci_to_token_id,
)


@dataclass
class Game:
    white_elo_token: int = ELO_UNKNOWN_ID
    black_elo_token: int = ELO_UNKNOWN_ID
    time_control_token: int = TC_UNKNOWN_ID
    target_outcome_token: int = DRAW_ID
    moves_uci: list[str] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    board: bulletchess.Board = field(default_factory=bulletchess.Board)

    def clear(self) -> None:
        """Reset to the initial position while preserving prompt metadata."""
        self.moves_uci.clear()
        self.tokens.clear()
        self.board = bulletchess.Board()

    def context_tokens(self) -> list[int]:
        """Return prompt tokens plus synchronized move tokens."""
        return [
            GAME_START_ID,
            self.time_control_token,
            self.target_outcome_token,
            self.white_elo_token,
            self.black_elo_token,
            *self.tokens,
        ]

    def legal_moves(self) -> list[str]:
        return [move.uci() for move in self.board.legal_moves()]

    def feed_uci(self, uci: str) -> None:
        move = self._parse_and_validate_uci(uci)
        piece = self.board[move.origin]
        if piece is None:
            raise ValueError(f"No piece on move origin for move: {uci}")
        token_id = uci_to_token_id(uci, self.board.turn, normalize_piece_type(piece.piece_type))
        if token_id is None:
            raise ValueError(f"No token for move: {uci}")

        self.board.apply(move)
        self.moves_uci.append(uci)
        self.tokens.append(token_id)

    def feed_token(self, token_id: int) -> None:
        token = token_to_uci(token_id)
        if token is None:
            raise ValueError(f"Unknown token id: {token_id}")
        if token_id in OUTCOME_TOKENS.values():
            raise ValueError(f"Token id {token_id} is not a move token")
        self.feed_uci(token)

    def len_tokens(self) -> int:
        return len(self.tokens)

    def is_empty_position(self) -> bool:
        return not self.tokens

    def _parse_and_validate_uci(self, uci: str) -> bulletchess.Move:
        try:
            move = bulletchess.Move.from_uci(uci)
        except Exception as exc:
            raise ValueError(f"Invalid UCI move: {uci}") from exc

        legal_moves = set(self.legal_moves())
        if uci not in legal_moves:
            raise ValueError(f"Illegal move for current position: {uci}")

        return move
