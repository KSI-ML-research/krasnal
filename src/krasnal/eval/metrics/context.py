from dataclasses import dataclass

import torch


@dataclass
class EvalContext:
    """Context for metric computation.

    Attributes:
        probs: Model's probability distribution over tokens.
        legal_ids: List of legal move token IDs in current position.
        sequence: Token sequence for inference.
        piece_type: Type of piece that was moved (1=pawn, ..., 6=king).
        actual_token: The actual token that was played (ground truth).
        in_check: Whether the player's king is in check.
        phase: Game phase (opening, middlegame, endgame).
        gives_check: Whether the move gives check to opponent.
        fen: FEN string of the position before the move.
        top1_fen: FEN string after applying model's top-1 legal move.
    """

    probs: torch.Tensor | None = None
    legal_ids: list[int] | None = None
    sequence: list[int] | None = None
    piece_type: int | None = None
    actual_token: int | None = None
    in_check: bool | None = None
    phase: str | None = None
    gives_check: bool | None = None
    fen: str | None = None
    post_move_fen: str | None = None
    top1_fen: str | None = None
    top1_move_uci: str | None = None
    cot_format_valid: bool | None = None
    cot_post_think_probs: torch.Tensor | None = None
    cot_post_think_actual_token: int | None = None
    cot_post_think_legal_ids: list[int] | None = None
    target_think_tokens: list[int] | None = None
    generated_think_tokens: list[int] | None = None
