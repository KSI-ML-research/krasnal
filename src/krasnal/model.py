"""
Krasnal GPT model definition, based on NanoGPT by Andrej Karpathy:
https://github.com/karpathy/nanoGPT/blob/master/model.py

Differences from the original NanoGPT implementation:
- Replaced LayerNorm with RMSNorm, which is more efficient. (paper: https://arxiv.org/abs/1910.07467)
- Replaced absolute positional embeddings with RoPE, which is better at extrapolation.
- Replaced ReLU activation with GeLU which is smoother and performs better in practice.
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
from krasnal.config import GPTConfig


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim

        # precompute RoPE matrices for the maximum sequence length
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        seq_idx = torch.arange(max_seq_len).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # cache [1, 1, max_seq_len, head_dim] for broadcasting over (B, nh, T, hs)
        self.register_buffer("cos_cached", idx_theta2.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", idx_theta2.sin()[None, None, :, :], persistent=False)

    def _neg_half(self, x: torch.Tensor):
        """
        Returns a vector rotated by pi/2.
        If considered as a 2D vector x = [x1, y1], this returns [-y1, x1].
        """
        d_2 = self.head_dim // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, q, k, position_offset: int = 0):
        """
        Applies Rotary Position Embedding (RoPE) to queries and keys.

        The standard 2D rotation matrix formula is:
        Rot(theta, (x, y)) = (x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta))

        Here, we split our feature dimension into two halves: the first half is 'x' and
        the second half is 'y'. Let's denote q = [x, y].
        Then _neg_half(q) = [-y, x].

        The rotation is applied as:
        q_rope = (q * cos) + (_neg_half(q) * sin)
               = ([x, y] * [cos, cos]) + ([-y, x] * [sin, sin])
               = [x*cos - y*sin, x*sin + y*cos]

        This precisely matches the mathematical 2D rotation, allowing us to compute
        it highly efficiently using element-wise vector operations.
        """
        # q, k: (B, nh, T, hs)
        T = q.shape[2]
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        end = position_offset + T
        if end > self.cos_cached.shape[2]:
            raise ValueError("position_offset + T exceeds max_seq_len")

        cos = self.cos_cached[:, :, position_offset:end].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[:, :, position_offset:end].to(device=q.device, dtype=q.dtype)

        neg_half_q = self._neg_half(q)
        q_rope = (q * cos) + (neg_half_q * sin)

        neg_half_k = self._neg_half(k)
        k_rope = (k * cos) + (neg_half_k * sin)

        return q_rope, k_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.rope = RoPE(config.n_embd // config.n_head, config.block_size)
        # flash attention requires PyTorch >= 2.0
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("PyTorch >= 2.0 required for scaled_dot_product_attention")

    def forward(self, x, past_kv: KVCache | None = None, layer_idx: int | None = None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate Q, K, V for all heads in batch and move heads forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        past_len = 0
        if past_kv is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when using past_kv")
            past_len = past_kv.get_seq_len()

        # apply RoPE per head
        q, k = self.rope(q, k, position_offset=past_len)

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
        if past_kv is not None:
            # For cached decoding, explicit mask is only needed when T > 1.
            is_causal = False
            if T > 1:
                q_pos = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(-1)
                k_pos = torch.arange(0, past_len + T, device=x.device).unsqueeze(0)
                attn_mask = k_pos <= q_pos

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_kv: KVCache | None = None, layer_idx: int | None = None):
        x = x + self.attn(self.ln_1(x), past_kv=past_kv, layer_idx=layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
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

        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self) -> int:
        """
        Return the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, ignore_index=-1, past_kv: KVCache | None = None):
        _b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        if past_kv is not None and targets is not None:
            raise ValueError("KV-cache mode is inference-only and does not support targets")

        tok_emb = self.transformer.wte(idx)  # token embeddings (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x, past_kv=past_kv, layer_idx=layer_idx)
        if past_kv is not None:
            past_kv.advance(t)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Create AdamW optimizer and use the fused version if available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
