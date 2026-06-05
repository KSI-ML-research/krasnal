from __future__ import annotations

import bulletchess
import torch
import torch.nn.functional as F

from krasnal.inference.game import Game
from krasnal.inference.kv_cache import KVCache
from krasnal.inference.utils import create_amp_context
from krasnal.model import GPT
from krasnal.time_conditioning import (
    clock_pairs_for_window,
    new_clock_tracks,
    sync_prefix_clock_tracks,
)
from krasnal.tokens import (
    ELO_ABOVE_2200_ID,
    GAME_START_ID,
    QA_TOKEN_IDS,
    TC_UNKNOWN_ID,
    legal_token_ids,
)
from krasnal.uci_engine.go_params import GoParams, uci_ms_to_clock_seconds


class InferenceSession:
    """Concrete inference session that incrementally decodes with KV cache when possible.

    The game owns the model context. Feeding a move appends the move token and,
    when enabled, its deterministic post-move material annotation.

    When ``use_clock_encodings`` is enabled, clock tensors follow the target
    alignment used during training: input token at global index ``g`` is paired
    with the clock row stored for token ``g + 1``. The leaf step uses clocks from
    ``prepare_go_clocks`` (UCI ``wtime`` / ``btime``).
    """

    def __init__(
        self,
        model: GPT,
        device: torch.device,
        game: Game | None = None,
        outcome_token: int | None = None,
        white_elo_token: int = ELO_ABOVE_2200_ID,
        black_elo_token: int = ELO_ABOVE_2200_ID,
        time_control_token: int = TC_UNKNOWN_ID,
        clock_initial_seconds: int | None = None,
    ):
        self.model = model
        self.device = device
        self._amp_ctx = create_amp_context(device)
        self._clock_initial_seconds = clock_initial_seconds

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

    def _init_clock_tracks(self) -> None:
        enabled = self.model.config.use_clock_encodings
        initial = self._clock_initial_seconds
        if enabled and initial is None:
            raise ValueError(
                "clock_initial_seconds is required when use_clock_encodings is enabled"
            )
        (
            self._per_token_active,
            self._per_token_opp,
            self._go_active_sec,
            self._go_opp_sec,
        ) = new_clock_tracks(len(self.context), enabled=enabled, initial_seconds=initial)

    def new_game(self, game: Game) -> None:
        """Replace the stored game and reset runtime-only state."""
        self.game = game
        self.context = self.game.context_tokens()
        self._init_clock_tracks()
        self._reset_cache_state()

    def sync_prefix_tokens_from_game(self) -> None:
        """Refresh fixed prefix tokens (Elo / TC) after ``Game`` metadata changes."""
        prefix = self.game.prefix_tokens()
        self.context = prefix + self.context[len(prefix) :]
        if self.model.config.use_clock_encodings:
            self._per_token_active, self._per_token_opp = sync_prefix_clock_tracks(
                self._per_token_active,
                self._per_token_opp,
                prefix_len=len(prefix),
                total_len=len(self.context),
                prefix_clock_seconds=self._clock_initial_seconds,
            )
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

    def prepare_go_clocks(self, go: GoParams | None) -> None:
        """Set leaf clock pair from UCI ``go`` (milliseconds) for side to move."""
        if not self.model.config.use_clock_encodings:
            return
        if go is None or go.wtime_ms is None or go.btime_ms is None:
            raise ValueError("go must include wtime and btime when use_clock_encodings is enabled")
        w_s = uci_ms_to_clock_seconds(go.wtime_ms)
        b_s = uci_ms_to_clock_seconds(go.btime_ms)
        if self.game.board.turn == bulletchess.WHITE:
            self._go_active_sec, self._go_opp_sec = w_s, b_s
        else:
            self._go_active_sec, self._go_opp_sec = b_s, w_s
        self._reset_cache_state()

    def _append_clock_tracks(self, clock_active: int, clock_opponent: int) -> None:
        self._per_token_active.append(int(clock_active))
        self._per_token_opp.append(int(clock_opponent))

    def _append_context_tokens(
        self,
        token_ids: list[int],
        *,
        clock_active: int | None,
        clock_opponent: int | None,
    ) -> None:
        self.context.extend(token_ids)
        if self.model.config.use_clock_encodings:
            if clock_active is None or clock_opponent is None:
                raise ValueError("clock_active and clock_opponent are required")
            for _ in token_ids:
                self._append_clock_tracks(clock_active, clock_opponent)

    def feed_token(
        self,
        token_id: int,
        *,
        clock_active: int | None = None,
        clock_opponent: int | None = None,
    ) -> None:
        """Append a token to model context and update game if it is a legal move token."""
        before = len(self.game.tokens)
        self.game.feed_token(token_id)
        self._append_context_tokens(
            self.game.tokens[before:],
            clock_active=clock_active,
            clock_opponent=clock_opponent,
        )
        self._last_logits = None

    def feed_uci(
        self,
        uci: str,
        clock_active: int | None = None,
        clock_opponent: int | None = None,
    ) -> None:
        """Append a UCI move, updating both game state and model context."""
        before = len(self.game.tokens)
        self.game.feed_uci(uci)
        self._append_context_tokens(
            self.game.tokens[before:],
            clock_active=clock_active,
            clock_opponent=clock_opponent,
        )
        self._last_logits = None

    def get_raw_logits(self) -> torch.Tensor:
        """Return next-token logits for the current model context."""
        if not self.context:
            self.context = [GAME_START_ID]
            self._init_clock_tracks()

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

        first_g = len(self.context) - len(tokens_to_process)
        extend_incrementally = self._cached_window_len > 0 and len(tokens_to_process) > 1
        token_batches = (
            [[token] for token in tokens_to_process]
            if extend_incrementally
            else [tokens_to_process]
        )

        logits = None
        processed = 0
        for token_batch in token_batches:
            x = torch.tensor([token_batch], dtype=torch.long, device=self.device)
            with torch.inference_mode(), self._amp_ctx:
                if not self.model.config.use_clock_encodings:
                    logits, _ = self.model(x, past_kv=self.kv_cache)
                    processed += len(token_batch)
                    continue

                act, opp = clock_pairs_for_window(
                    first_g + processed,
                    len(token_batch),
                    context_len=len(self.context),
                    per_token_active=self._per_token_active,
                    per_token_opp=self._per_token_opp,
                    go_active_sec=self._go_active_sec,
                    go_opp_sec=self._go_opp_sec,
                    enabled=True,
                )
                active_t = torch.tensor([act], dtype=torch.long, device=self.device)
                opp_t = torch.tensor([opp], dtype=torch.long, device=self.device)
                logits, _ = self.model(
                    x,
                    past_kv=self.kv_cache,
                    active_clock_ids=active_t,
                    opponent_clock_ids=opp_t,
                )
                processed += len(token_batch)

        self._cached_window_len = len(context_window)
        if logits is None:
            raise RuntimeError("Model returned no logits")
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
