from __future__ import annotations

from typing import TYPE_CHECKING

import bulletchess
import torch

from inference.abstracts import BaseGenerator, BaseInferenceSession, BaseSampler
from inference.utils import get_legal_token_ids

if TYPE_CHECKING:
    from tokenizer import Tokenizer


class MoveGenerator(BaseGenerator):
    """Generator that selects a move directly from the current policy distribution.

    Given the session's next-token probabilities, this class:
        1. Identifies all legal UCI moves for the current board position.
        2. Masks out illegal moves and re-normalizes the distribution.
        3. Samples a move via the provided sampler.

    This is a single-token generator: it consumes exactly one token from the model
    per call to `generate_move`. It does not generate intermediate reasoning tokens.

    Note: `generate_move` calls `session.feed(token_id)` internally to advance the
    session context after selecting a move.

    CoT inference: a future `CoTGenerator` subclass will loop for up to `max_tokens`,
    letting the model emit `<think> ... ` reasoning tokens before producing a legal
    move. It will re-sample whenever an illegal token appears until a legal move
    token is emitted.
    """

    def generate_move(
        self,
        session: BaseInferenceSession,
        board: bulletchess.Board,
        tokenizer: Tokenizer,
        sampler: BaseSampler,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        legal_ids = get_legal_token_ids(board, tokenizer)
        if not legal_ids:
            return "0000"

        probs = session.get_probs()

        legal_mask = torch.zeros(len(probs), dtype=torch.bool, device=probs.device)
        legal_mask[legal_ids] = True
        filtered_probs = probs.clone()
        filtered_probs[~legal_mask] = 0
        filtered_probs = filtered_probs / (filtered_probs.sum() + 1e-10)

        token_id = sampler.sample(filtered_probs, temperature, top_p)
        # fallback if sampler returns an illegal move because top-p had no legal moves
        if token_id not in legal_ids:
            token_id = int(torch.argmax(filtered_probs).item())
        session.feed(token_id)
        return tokenizer.id_to_move[token_id]
