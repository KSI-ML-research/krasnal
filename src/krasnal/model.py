"""
Krasnal GPT model definition, based on NanoGPT by Andrej Karpathy:
https://github.com/karpathy/nanoGPT/blob/master/model.py

Differences from the original NanoGPT implementation:
- Replaced LayerNorm with RMSNorm, which is more efficient. (paper: https://arxiv.org/abs/1910.07467)
- Replaced absolute positional embeddings with RoPE, which is better at extrapolation.
- Replaced ReLU activation with GeLU which is smoother and performs better in practice.
- Added KV Cache (Kyryllo Goroshenko) to speed up long context inference.
- Conditional log-clock TimeConditioning (optional)
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from krasnal.inference.kv_cache import KVCache
from krasnal.config import CLOCK_IGNORE_ID, GPTConfig, MlpActivation

MLP_ACTIVATIONS: frozenset[MlpActivation] = frozenset({"gelu", "swiglu", "relu2"})


def _swiglu_hidden_dim(n_embd: int) -> int:
    """Intermediate width for SwiGLU FFN (~same params as 4*n_embd GELU MLP)."""
    hidden = int(8 * n_embd / 3)
    return ((hidden + 7) // 8) * 8


class RoPE(nn.Module):
    """Rotary Position Embeddings (RoPE) implementation.

    RoPE encodes positional information by rotating the Query and Key vectors in 2D
    planes of the embedding space, preserving relative distances.
    Paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # precompute RoPE matrices for the maximum sequence length
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        seq_idx = torch.arange(max_seq_len).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # cache [1, 1, max_seq_len, head_dim] for broadcasting over (B, nh, T, hs)
        self.register_buffer("cos_cached", idx_theta2.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", idx_theta2.sin()[None, None, :, :], persistent=False)

    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates a vector by pi/2.

        Splits the feature dimension in half: [x1, y1] -> [-y1, x1].
        """
        d_2 = self.head_dim // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding (RoPE) to Query and Key tensors."""
        assert q.shape == k.shape, "q and k shapes must match"
        assert len(q.shape) == 4, "q and k must be 4D tensors of shape (B, nh, T, hs)"
        assert q.shape[3] == self.head_dim, f"head_dim must match {self.head_dim}"
        T = q.shape[2]

        if position_ids is None:
            assert position_offset >= 0, "position_offset must be non-negative"
            end = position_offset + T
            assert self.max_seq_len >= T, (
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )
            assert end <= self.max_seq_len, (
                f"end index {end} exceeds max_seq_len {self.max_seq_len}"
            )
            cos = self.cos_cached[:, :, position_offset:end].to(device=q.device, dtype=q.dtype)
            sin = self.sin_cached[:, :, position_offset:end].to(device=q.device, dtype=q.dtype)
        else:
            assert position_ids.shape == (q.shape[0], T), (
                f"position_ids must be (B, T), got {tuple(position_ids.shape)}"
            )
            # Clamp for index safety; avoid data-dependent asserts (breaks torch.compile).
            pos = position_ids.clamp(min=0, max=self.max_seq_len - 1)
            cache = self.cos_cached[0, 0].to(device=q.device, dtype=q.dtype)
            cos = cache[pos].unsqueeze(1)
            sin = self.sin_cached[0, 0].to(device=q.device, dtype=q.dtype)[pos].unsqueeze(1)

        q_rope = (q * cos) + (self._neg_half(q) * sin)
        k_rope = (k * cos) + (self._neg_half(k) * sin)

        return q_rope, k_rope


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by head count"
        head_dim = config.n_embd // config.n_head
        assert head_dim % 8 == 0, (
            f"Head dimension ({head_dim}) must be a multiple of 8 for optimal performance"
        )
        # Flash attention requires PyTorch >= 2.0
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("PyTorch >= 2.0 required for scaled_dot_product_attention")

        # Combined key, query, and value projections in a single batch Linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Regularization / Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.rope = RoPE(config.n_embd // config.n_head, config.block_size)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        layer_idx: int | None = None,
        segment_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert len(x.shape) == 3
        assert x.shape[2] == self.n_embd

        B, T, C = x.size()  # batch size, sequence length, embedding dim (n_embd)

        # Calculate Q, K, V for all heads in batch and move heads forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        past_len = 0
        if past_kv is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when using past_kv cache")
            past_len = past_kv.get_seq_len()

        assert past_len + T <= self.rope.max_seq_len, (
            f"Total sequence length (past_len={past_len} + current_T={T}) "
            f"exceeds model block_size limit ({self.rope.max_seq_len})"
        )

        if past_kv is not None and position_ids is not None:
            raise ValueError("position_ids are not supported with KV-cache inference")
        if past_kv is not None and segment_ids is not None:
            raise ValueError("segment_ids are not supported with KV-cache inference")

        q, k = self.rope(
            q,
            k,
            position_offset=past_len,
            position_ids=position_ids,
        )

        # Append to KV-Cache if provided
        if past_kv is not None:
            past_kv.append_layer(layer_idx, k, v)
            # Get full k and v directly from cache tensors after append
            k_full = past_kv.key_cache[layer_idx, :, :, : past_len + T, :]
            v_full = past_kv.value_cache[layer_idx, :, :, : past_len + T, :]
        else:
            k_full = k
            v_full = v

        attn_mask = None
        is_causal = True
        if segment_ids is not None:
            if position_ids is None:
                raise ValueError("position_ids are required when segment_ids are provided")
            seg_q = segment_ids.unsqueeze(-1)
            seg_k = segment_ids.unsqueeze(-2)
            pos_q = position_ids.unsqueeze(-1)
            pos_k = position_ids.unsqueeze(-2)
            # (B, 1, T, T) for scaled_dot_product_attention with q shape (B, nh, T, hs)
            attn_mask = ((seg_q == seg_k) & (pos_k <= pos_q)).unsqueeze(1)
            is_causal = False
        elif past_kv is not None:
            is_causal = False
            if T > 1:
                q_pos = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(-1)
                k_pos = torch.arange(0, past_len + T, device=x.device).unsqueeze(0)
                attn_mask = k_pos <= q_pos

        # Flash Attention forward pass
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Re-assemble head outputs and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network component of the Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        activation = config.mlp_activation
        if activation not in MLP_ACTIVATIONS:
            raise ValueError(
                f"mlp_activation must be one of {sorted(MLP_ACTIVATIONS)}, got {activation!r}"
            )
        self.activation = activation
        self.dropout = nn.Dropout(config.dropout)

        if activation == "swiglu":
            hidden = _swiglu_hidden_dim(config.n_embd)
            self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
            self.c_up = nn.Linear(config.n_embd, hidden, bias=False)
            self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        else:
            hidden = 4 * config.n_embd
            self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
            self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
            if activation == "gelu":
                self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            x = F.silu(self.c_gate(x)) * self.c_up(x)
        elif self.activation == "gelu":
            x = self.act(self.c_fc(x))
        else:
            x = F.relu(self.c_fc(x)).square()
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        layer_idx: int | None = None,
        segment_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x),
            past_kv=past_kv,
            layer_idx=layer_idx,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class TimeConditioning(nn.Module):
    """Encodes active and opponent clocks into continuous embeddings."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        h = int(config.time_conditioning_hidden)
        assert 1 <= h <= config.n_embd, (
            f"time_conditioning_hidden must satisfy 1 <= h <= n_embd ({config.n_embd}), got {h}"
        )
        self.mlp = nn.Sequential(
            nn.Linear(2, h),
            nn.GELU(),
            nn.Linear(h, config.n_embd, bias=False),
        )

    def _clock_pair_features(
        self,
        active_clock_ids: torch.Tensor,
        opponent_clock_ids: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """Computes logarithmic clock features of shape (B, T, 2) from clock tokens."""
        shape = idx.shape
        if active_clock_ids.shape != shape or opponent_clock_ids.shape != shape:
            raise ValueError(
                f"Clock tensors must match idx shape {tuple(shape)}, "
                f"got active={tuple(active_clock_ids.shape)}, "
                f"opponent={tuple(opponent_clock_ids.shape)}"
            )

        def encode(clock_ids: torch.Tensor) -> torch.Tensor:
            valid = (clock_ids != CLOCK_IGNORE_ID).to(torch.float32).unsqueeze(-1)
            t = torch.where(
                clock_ids != CLOCK_IGNORE_ID, clock_ids.to(torch.float32), 0.0
            ).unsqueeze(-1)
            return torch.log1p(t) * valid

        return torch.cat([encode(active_clock_ids), encode(opponent_clock_ids)], dim=-1)

    def forward(
        self,
        active_clock_ids: torch.Tensor,
        opponent_clock_ids: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        assert active_clock_ids.shape == idx.shape, "active_clock_ids shape must match idx"
        assert opponent_clock_ids.shape == idx.shape, "opponent_clock_ids shape must match idx"
        assert active_clock_ids.device == idx.device, "active_clock_ids device must match idx"

        pair = self._clock_pair_features(active_clock_ids, opponent_clock_ids, idx)
        return self.mlp(pair.to(dtype=self.mlp[0].weight.dtype))


class GPT(nn.Module):
    """Transformer-based language model"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified"
        assert config.block_size is not None, "block_size must be specified"
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Optional clock conditioning projection
        if config.use_time_conditioning:
            self.time_conditioning = TimeConditioning(config)

        # Weight tying (https://paperswithcode.com/method/weight-tying)
        self.transformer.wte.weight = self.lm_head.weight

        # Parameter initialization
        self.apply(self._init_weights)

        # Apply specialized scaled initialization to residual projections (GPT-2 standard)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights with normal distribution standard deviation 0.02."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -1,
        past_kv: KVCache | None = None,
        active_clock_ids: torch.Tensor | None = None,
        opponent_clock_ids: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_all_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transformer forward pass with optional targets and time conditioning."""
        assert len(idx.shape) == 2, "idx must be a 2D tensor of shape (B, T)"
        _b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size limit is {self.config.block_size}"
        )
        if past_kv is not None and targets is not None:
            raise ValueError(
                "KV-cache mode is inference-only and does not support target loss computation"
            )
        if targets is not None:
            assert targets.shape == idx.shape, "targets shape must match idx shape (B, T)"
            assert targets.dtype == torch.long, "targets must be a LongTensor (class indices)"

        # Base token embeddings
        x = self.transformer.wte(idx)  # (b, t, n_embd)

        if self.config.use_time_conditioning:
            if active_clock_ids is None or opponent_clock_ids is None:
                raise ValueError(
                    "active_clock_ids and opponent_clock_ids are required when "
                    "use_time_conditioning is True "
                    "(pass CLOCK_IGNORE_ID values for unknown tokens)."
                )
            x = x + self.time_conditioning(active_clock_ids, opponent_clock_ids, idx)

        x = self.transformer.drop(x)

        # Forward pass through all Block layers
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(
                x,
                past_kv=past_kv,
                layer_idx=layer_idx,
                segment_ids=segment_ids,
                position_ids=position_ids,
            )

        if past_kv is not None:
            past_kv.advance(t)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
        elif return_all_logits:
            logits = self.lm_head(x)
            loss = None
        else:
            # Inference: only compute the language modeling head on the last token to save FLOPs
            logits = self.lm_head(x[:, [-1], :])  # preserves the sequence dimension shape (B, 1, V)
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Sets up weight decay optimizer parameters.

        Applies weight decay to all 2D parameter tensors (weights of MatMul and Embeddings)
        while omitting biases and 1D normalization tensors from decay.
        """
        # Collect all parameters requiring gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate into weight-decay eligible (dim >= 2) and ineligible (dim < 2)
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Check for fused AdamW support (available in modern PyTorch + CUDA)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
