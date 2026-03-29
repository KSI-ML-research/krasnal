from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from krasnal.inference.game import Game
from krasnal.inference.utils import create_amp_context
from krasnal.model import GPT
from krasnal.tokens import (
    ELO_UNKNOWN_ID,
    GAME_START_ID,
    THINK_END_ID,
    THINK_START_ID,
    legal_token_ids,
)


class InferenceSession:
    """Concrete inference session that runs a full forward pass per `get_raw_logits()`.

    The session owns a synchronized `Game` for chess state and a raw `context`
    token stream for actual model input. Non-move tokens (for example CoT
    control tokens) are preserved in the raw context without mutating `Game`.

    Future improvements:
        - KV-cache: cache key/value activations for the growing prefix to avoid
          re-computing already-seen tokens on every forward pass.
        - CoT awareness: a future session variant may store structured reasoning
          state alongside the raw token sequence.
    """

    def __init__(
        self,
        model: GPT,
        device: torch.device,
        game: Game | None = None,
        outcome_token: int | None = None,
        white_elo_token: int = ELO_UNKNOWN_ID,
        black_elo_token: int = ELO_UNKNOWN_ID,
    ):
        self.model = model
        self.device = device
        self._amp_ctx = create_amp_context(device)
        self.kv_cache = None

        if game is None:
            if outcome_token is None:
                raise ValueError("outcome_token must be provided when game is not supplied")
            game = Game(
                white_elo_token=white_elo_token,
                black_elo_token=black_elo_token,
                target_outcome_token=outcome_token,
            )

        self.new_game(game)

    def reset(
        self,
        outcome_token: int,
        white_elo_token: int = ELO_UNKNOWN_ID,
        black_elo_token: int = ELO_UNKNOWN_ID,
    ) -> None:
        """Backward-compatible reset that rebuilds the underlying Game."""
        self.new_game(
            Game(
                white_elo_token=white_elo_token,
                black_elo_token=black_elo_token,
                target_outcome_token=outcome_token,
            )
        )

    def new_game(self, game: Game) -> None:
        """Replace the stored game and reset runtime-only state."""
        self.game = game
        self.context = self.game.context_tokens()
        self._in_think_block = False
        self.kv_cache = None

    def feed_token(self, token_id: int) -> None:
        """Append a token to model context and update game if it is a legal move token."""
        self.context.append(token_id)
        if token_id == THINK_START_ID:
            self._in_think_block = True
            self.kv_cache = None
            return
        if token_id == THINK_END_ID:
            self._in_think_block = False
            self.kv_cache = None
            return
        if self._in_think_block:
            self.kv_cache = None
            return
        with contextlib.suppress(ValueError):
            self.game.feed_token(token_id)
        self.kv_cache = None

    def feed_uci(self, uci: str) -> None:
        """Append a UCI move, updating both game state and model context."""
        self.game.feed_uci(uci)
        self.context.append(self.game.tokens[-1])
        self.kv_cache = None

    def get_raw_logits(self) -> torch.Tensor:
        """Return next-token logits for the current model context."""
        if not self.context:
            self.context = [GAME_START_ID]

        block_size = self.model.config.block_size
        context_window = self.context[-block_size:]  # sliding window context
        x = torch.tensor([context_window], dtype=torch.long, device=self.device)
        with torch.inference_mode(), self._amp_ctx:
            logits, _ = self.model(x)
        return logits[0, -1]

    def get_legal_logits(self) -> torch.Tensor:
        """Return next-token logits with illegal moves masked out."""
        logits = self.get_raw_logits()
        legal_ids = legal_token_ids(self.game.board)
        masked = torch.full_like(logits, float("-inf"))
        if legal_ids:
            masked[legal_ids] = logits[legal_ids]
        return masked

    def get_raw_probs(self) -> torch.Tensor:
        return F.softmax(self.get_raw_logits(), dim=-1)

    def get_legal_probs(self) -> torch.Tensor:
        legal_logits = self.get_legal_logits()
        return F.softmax(legal_logits, dim=-1)
