from __future__ import annotations

from typing import Protocol

import chess
import torch

from ..tokenizer import Tokenizer


class BaseInferenceSession(Protocol):
    """Protocol for stateful model inference (token context management)."""

    def reset(self, outcome_token: int) -> None:
        """Clear context and start a new game."""

    def feed(self, token_id: int) -> None:
        """Append a token to the context."""

    def get_probs(self) -> torch.Tensor:
        """Return probability distribution over the next token (vocab_size,)."""


class BaseSampler(Protocol):
    """Protocol for selecting a token ID from a probability distribution."""

    def sample(
        self,
        probs: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> int:
        """Sample a token ID from probabilities."""


class BaseGenerator(Protocol):
    """Protocol for generating moves (potentially with CoT) using a session."""

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
        """Select a legal move and return it in UCI notation."""
