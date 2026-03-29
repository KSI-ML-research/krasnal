from __future__ import annotations

import inspect
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from config import SOS_ID
from inference.abstracts import BaseInferenceSession
from inference.kv_cache import KVCache
from model import GPT


class InferenceSession(BaseInferenceSession):
    """Concrete inference session that runs a full forward pass per `get_probs()`.

        Uses a KV cache to avoid redundant computation when the model supports it.

        Future optimizations:
        - CoT awareness: a future session variant may store structured reasoning
        state alongside the raw token sequence.
    """

    def __init__(
        self,
        model: GPT,
        device: torch.device,
        outcome_token: int = SOS_ID,
        use_kv_cache: bool = True,
    ):
        self.model = model
        self.device = device
        self.use_kv_cache = use_kv_cache
        self._model_supports_kv = "past_kv" in inspect.signature(self.model.forward).parameters
        self.kv_cache: KVCache | None = None
        self._amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        self.reset(outcome_token)

    def _build_kv_cache(self) -> KVCache:
        config = self.model.config
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        return KVCache(
            batch_size=1,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            head_dim=config.n_embd // config.n_head,
            max_seq_len=config.block_size,
            device=self.device,
            dtype=dtype,
        )

    def reset(self, outcome_token: int = SOS_ID) -> None:
        """Clear context and start a new game."""
        self.context: list[int] = [outcome_token]
        self._cached_context: list[int] = []
        if self.use_kv_cache:
            self.kv_cache = self._build_kv_cache()
        else:
            self.kv_cache = None

    def feed(self, token_id: int | list[int]) -> None:
        """Append one or more tokens to the context."""
        if isinstance(token_id, list):
            self.context.extend(token_id)
        else:
            self.context.append(token_id)

    def get_probs(self) -> torch.Tensor:
        """Return probability distribution over the next token."""
        block_size = self.model.config.block_size
        context_window = self.context[-block_size:]  # sliding window context

        x_tokens = context_window
        if self.use_kv_cache and self._model_supports_kv and self.kv_cache is not None:
            cached_len = self.kv_cache.get_seq_len()
            # If cached context doesn't match the current context, reset the cache.
            # This can happen if the session context was truncated (e.g. due to block size limit)
            # or if the session was reset with a different outcome token.
            if (
                cached_len != len(self._cached_context)
                or context_window[:cached_len] != self._cached_context[:cached_len]
                or cached_len > len(context_window)
            ):
                self.kv_cache.reset()
                self._cached_context = []
                cached_len = 0

            if cached_len < len(context_window):
                x_tokens = context_window[cached_len:]
            else:
                self.kv_cache.reset()
                self._cached_context = []
                x_tokens = context_window

        x = torch.tensor([x_tokens], dtype=torch.long, device=self.device)
        with torch.inference_mode(), self._amp_ctx:
            if self.use_kv_cache and self._model_supports_kv and self.kv_cache is not None:
                logits, _ = self.model(x, past_kv=self.kv_cache)
                self._cached_context = list(context_window)
            else:
                logits, _ = self.model(x)
        return F.softmax(logits[0, -1], dim=-1)
