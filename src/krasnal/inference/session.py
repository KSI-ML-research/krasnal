from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from krasnal.inference.game import Game
from krasnal.inference.kv_cache import KVCache
from krasnal.inference.utils import create_amp_context
from krasnal.model import GPT
from krasnal.tokens import (
    ELO_UNKNOWN_ID,
    GAME_START_ID,
    QA_TOKEN_IDS,
    TC_UNKNOWN_ID,
    THINK_END_ID,
    THINK_START_ID,
    legal_token_ids,
)


class InferenceSession:
    """Concrete inference session that incrementally decodes with KV cache when possible.

    The session owns a synchronized `Game` for chess state and a raw `context`
    token stream for actual model input. Non-move tokens (for example CoT
    control tokens) are preserved in the raw context without mutating `Game`.
    """

    def __init__(
        self,
        model: GPT,
        device: torch.device,
        game: Game | None = None,
        outcome_token: int | None = None,
        white_elo_token: int = ELO_UNKNOWN_ID,
        black_elo_token: int = ELO_UNKNOWN_ID,
        time_control_token: int = TC_UNKNOWN_ID,
    ):
        self.model = model
        self.device = device
        self._amp_ctx = create_amp_context(device)

        if game is None:
            if outcome_token is None:
                raise ValueError("outcome_token must be provided when game is not supplied")
            game = Game(
                white_elo_token=white_elo_token,
                black_elo_token=black_elo_token,
                time_control_token=time_control_token,
                target_outcome_token=outcome_token,
            )

        self.new_game(game)

    def reset(
        self,
        outcome_token: int,
        white_elo_token: int = ELO_UNKNOWN_ID,
        black_elo_token: int = ELO_UNKNOWN_ID,
        time_control_token: int = TC_UNKNOWN_ID,
    ) -> None:
        """Backward-compatible reset that rebuilds the underlying Game."""
        self.new_game(
            Game(
                white_elo_token=white_elo_token,
                black_elo_token=black_elo_token,
                time_control_token=time_control_token,
                target_outcome_token=outcome_token,
            )
        )

    def new_game(self, game: Game) -> None:
        """Replace the stored game and reset runtime-only state."""
        self.game = game
        self.context = self.game.context_tokens()
        self._in_think_block = False
        self._reset_cache_state()

    def _reset_cache_state(self) -> None:
        self.kv_cache: KVCache | None = None
        self._cached_window_start = 0
        self._cached_window_len = 0
        self._last_logits: torch.Tensor | None = None

    def _kv_cache_dtype(self) -> torch.dtype:
        if self.device.type == "cuda":
            return torch.bfloat16
        return next(self.model.parameters()).dtype

    def _build_kv_cache(self) -> KVCache:
        cfg = self.model.config
        return KVCache(
            batch_size=1,
            num_layers=cfg.n_layer,
            num_heads=cfg.n_head,
            head_dim=cfg.n_embd // cfg.n_head,
            max_seq_len=cfg.block_size,
            device=self.device,
            dtype=self._kv_cache_dtype(),
        )

    def feed_token(self, token_id: int) -> None:
        """Append a token to model context and update game if it is a legal move token."""
        self.context.append(token_id)
        if token_id == THINK_START_ID:
            self._in_think_block = True
            self._last_logits = None
            return
        if token_id == THINK_END_ID:
            self._in_think_block = False
            self._last_logits = None
            return
        if self._in_think_block:
            self._last_logits = None
            return
        with contextlib.suppress(ValueError):
            self.game.feed_token(token_id)
        self._last_logits = None

    def feed_uci(self, uci: str) -> None:
        """Append a UCI move, updating both game state and model context."""
        self.game.feed_uci(uci)
        self.context.append(self.game.tokens[-1])
        self._last_logits = None

    def get_raw_logits(self) -> torch.Tensor:
        """Return next-token logits for the current model context."""
        if not self.context:
            self.context = [GAME_START_ID]

        block_size = self.model.config.block_size
        context_window = self.context[-block_size:]  # sliding window context
        window_start = len(self.context) - len(context_window)

        if (
            self.kv_cache is not None
            and self._last_logits is not None
            and self._cached_window_start == window_start
            and self._cached_window_len == len(context_window)
        ):
            return self._last_logits

        if (
            self.kv_cache is None
            or self._cached_window_start != window_start
            or self._cached_window_len > len(context_window)
        ):
            self.kv_cache = self._build_kv_cache()
            tokens_to_process = context_window
            self._cached_window_start = window_start
            self._cached_window_len = 0
        else:
            tokens_to_process = context_window[self._cached_window_len :]

        if not tokens_to_process:
            if self._last_logits is None:
                raise RuntimeError("Missing cached logits for current context window")
            return self._last_logits

        x = torch.tensor([tokens_to_process], dtype=torch.long, device=self.device)
        with torch.inference_mode(), self._amp_ctx:
            logits, _ = self.model(x, past_kv=self.kv_cache)

        self._cached_window_len = len(context_window)
        self._last_logits = logits[0, -1]
        return self._last_logits

    def get_legal_logits(self) -> torch.Tensor:
        """Return next-token logits with illegal moves and Q&A tokens masked out."""
        logits = self.get_raw_logits()
        legal_ids = legal_token_ids(self.game.board)
        masked = torch.full_like(logits, float("-inf"))
        if legal_ids:
            masked[legal_ids] = logits[legal_ids]
        masked[list(QA_TOKEN_IDS)] = float("-inf")
        return masked

    def get_raw_probs(self) -> torch.Tensor:
        return F.softmax(self.get_raw_logits(), dim=-1)

    def get_legal_probs(self) -> torch.Tensor:
        legal_logits = self.get_legal_logits()
        return F.softmax(legal_logits, dim=-1)
