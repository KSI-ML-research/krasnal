from dataclasses import dataclass


@dataclass
class EvalContext:
    """Context for metric computation.

    Attributes:
        sequence: Token sequence for inference.
        piece_type: Type of piece that was moved (1=pawn, ..., 6=king).
        actual_token: The actual token that was played (ground truth).
        in_check: Whether the player's king is in check.
        phase: Game phase (opening, middlegame, endgame).
        player_elo_token: Elo bucket token for the side to move.
        gives_check: Whether the move gives check to opponent.
        fen: FEN string of the position before the move.
        active_clock_seconds: Remaining seconds for the side to move (if known from data).
        opponent_clock_seconds: Opponent remaining seconds at the same ply (if known).
    """

    sequence: list[int] | None = None
    piece_type: int | None = None
    actual_token: int | None = None
    in_check: bool | None = None
    phase: str | None = None
    player_elo_token: int | None = None
    gives_check: bool | None = None
    fen: str | None = None
    post_move_fen: str | None = None
    what_is_on_game_key: str | None = None
    what_is_on_ply: int | None = None
    active_clock_seconds: int | None = None
    opponent_clock_seconds: int | None = None
    active_clock_sequence: list[int] | None = None
    opponent_clock_sequence: list[int] | None = None
