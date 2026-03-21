from __future__ import annotations

import chess
import torch

from inference.abstracts import BaseGenerator, BaseInferenceSession, BaseSampler
from tokenizer import Tokenizer


class MoveGenerator(BaseGenerator):
    """Generator that selects a move directly from the current policy distribution.

    Given the session's next-token probabilities, this class:
        1. Identifies all legal UCI moves for the current board position.
        2. Masks out illegal moves and re-normalizes the distribution.
        3. Samples a move via the provided sampler.

    This is a single-token generator: it consumes exactly one token from the model
    per call to `generate_move`. It does not generate intermediate reasoning tokens.

    CoT inference: a future `CoTGenerator` subclass will loop for up to `max_tokens`,
    letting the model emit `<think> ... ` reasoning tokens before producing a legal
    move. It will re-sample whenever an illegal token appears until a legal move
    token is emitted.
    """

    def generate_move(
        self,
        session: BaseInferenceSession,
        board: chess.Board,
        tokenizer: Tokenizer,
        sampler: BaseSampler,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> str:
        del max_tokens
        legal_ids = [tokenizer.move_to_id[m.uci()] for m in board.legal_moves]
        if not legal_ids:
            return "0000"

        probs = session.get_probs()

        mask = torch.zeros_like(probs)
        mask[legal_ids] = 1.0
        filtered_probs = (probs * mask) / (probs[legal_ids].sum() + 1e-10)

        token_id = sampler.sample(filtered_probs, temperature, top_p)
        session.feed(token_id)
        return tokenizer.id_to_move[token_id]
