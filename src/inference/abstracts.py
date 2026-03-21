from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import chess
import torch

if TYPE_CHECKING:
    from tokenizer import Tokenizer


class BaseInferenceSession(Protocol):
    """Manages model state and context during inference.

    This is the lowest-level abstraction in the inference stack. It holds the
    token sequence fed to the model so far and produces a probability distribution
    over the next token via `get_probs()`. Subclasses may add KV-cache support
    for efficient prefix reuse (planned).

    Why separate from Generator?
        The session knows nothing about chess rules or move legality. It purely
        handles token-level state. This separation allows the same session to be
        reused by different generators (e.g. direct move selection, CoT-based).
    """

    def reset(self, outcome_token: int) -> None:
        """Clear context and start a new game."""

    def feed(self, token_id: int) -> None:
        """Append a token to the context."""

    def get_probs(self) -> torch.Tensor:
        """Return probability distribution over the next token (vocab_size,)."""


class BaseSampler(Protocol):
    """Selects a token ID from a probability distribution.

    Implementations may apply temperature scaling, nucleus (top-p) sampling,
    or greedy selection. The sampler is intentionally decoupled from both the
    session and the generator so it can be swapped independently (e.g. for
    deterministic inference with temperature=0, or for diverse sampling).
    """

    def sample(
        self,
        probs: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> int:
        """Sample a token ID from probabilities."""


class BaseGenerator(Protocol):
    """Selects a legal move from the model using a session and sampler.

    The generator is the highest-level inference abstraction. It ties together
    a session (which provides probability distributions), a sampler (which
    converts probs to a token ID), and chess board state (to enforce move legality).

    CoT inference: Future subclasses (e.g. CoTGenerator) will loop for up to
    `max_tokens`, letting the model generate reasoning tokens before a legal move
    appears. The base protocol intentionally includes `max_tokens` to signal this
    planned capability.
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
        """Select a legal move and return it in UCI notation."""
