from __future__ import annotations

import chess
import torch

from ..tokenizer import SPECIAL_TOKENS, Tokenizer
from .abstracts import BaseGenerator, BaseInferenceSession, BaseSampler


class SimpleGenerator(BaseGenerator):
    """Basic generator that selects a move directly from legal options."""

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

        # Mask illegal moves and re-normalize
        mask = torch.zeros_like(probs)
        mask[legal_ids] = 1.0
        filtered_probs = (probs * mask) / (probs[legal_ids].sum() + 1e-10)

        token_id = sampler.sample(filtered_probs, temperature, top_p)
        session.feed(token_id)
        return tokenizer.id_to_move[token_id]


class CoTGenerator(BaseGenerator):
    """Generator that allows for optional <think>...</think> blocks."""

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
        legal_ids = [tokenizer.move_to_id[m.uci()] for m in board.legal_moves]
        if not legal_ids:
            return "0000"

        special_ids = set(SPECIAL_TOKENS)
        thinking = False
        saw_think = False
        think_complete = False

        for _ in range(max_tokens):
            probs = session.get_probs()
            # Sample from full vocab to allow <think> and </think>
            next_id = sampler.sample(probs, temperature, top_p)

            if next_id == tokenizer.think_start_id:
                session.feed(next_id)
                thinking = True
                saw_think = True
                continue

            if next_id == tokenizer.think_end_id:
                session.feed(next_id)
                thinking = False
                think_complete = True
                continue

            if thinking:
                session.feed(next_id)
                continue

            if next_id in special_ids:
                # We generally don't want to loop on other special tokens
                # but if the model emits one, we feed it and continue.
                session.feed(next_id)
                continue

            if saw_think and not think_complete:
                # Still inside the logical "think" even if model forgot tag
                session.feed(next_id)
                continue

            # If we reached here, we want a legal move.
            # Mask illegal moves and re-sample just to be sure.
            mask = torch.zeros_like(probs)
            mask[legal_ids] = 1.0
            filtered_probs = (probs * mask) / (probs[legal_ids].sum() + 1e-10)

            final_id = sampler.sample(filtered_probs, temperature, top_p)
            session.feed(final_id)
            return tokenizer.id_to_move[final_id]

        return "0000"
