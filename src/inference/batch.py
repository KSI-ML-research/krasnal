"""Batch inference for evaluating multiple positions at once."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from inference.abstracts import BaseInferenceSession
from inference.utils import create_amp_context
from tokenizer import PAD_ID

if TYPE_CHECKING:
    from model import GPT


class BatchInferenceSession(BaseInferenceSession):
    """Batched inference session for evaluating multiple positions simultaneously.

    This class processes multiple token sequences in a single forward pass,
    which is significantly faster than sequential inference for evaluation tasks.

    Example:
        >>> session = BatchInferenceSession(model, device)
        >>> sequences = [[SOS_ID, 100, 200], [SOS_ID, 150, 220, 300]]
        >>> probs = session.get_probs_batch(sequences)
        >>> # probs shape: (2, vocab_size)
    """

    def __init__(
        self,
        model: GPT,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self._amp_ctx = create_amp_context(device)

    def get_probs_batch(
        self,
        sequences: list[list[int]],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Forward pass on a batch of sequences in chunks.

        Args:
            sequences: List of token sequences. Each sequence is a list of token IDs.
                       The first token is assumed to be the start token (e.g., SOS_ID).
            batch_size: Number of sequences to process at once. Default 256.

        Returns:
            Tensor of shape (batch_size, vocab_size) containing probability
            distributions for the next token at each position.
        """
        if not sequences:
            raise ValueError("sequences cannot be empty")

        block_size = self.model.config.block_size

        all_probs = []

        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i : i + batch_size]

            max_len = max(len(seq) for seq in chunk)

            if max_len > block_size:
                raise ValueError(f"Sequence length {max_len} exceeds block_size {block_size}")

            padded = torch.full(
                (len(chunk), max_len),
                fill_value=PAD_ID,
                dtype=torch.long,
                device=self.device,
            )
            lengths = torch.empty(len(chunk), dtype=torch.long, device=self.device)

            for j, seq in enumerate(chunk):
                padded[j, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                lengths[j] = len(seq)

            with torch.inference_mode(), self._amp_ctx:
                logits, _ = self.model(padded, padded, ignore_index=PAD_ID)

            last_token_idx = lengths - 1
            batch_idx = torch.arange(len(chunk), device=self.device)
            next_token_logits = logits[batch_idx, last_token_idx, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            all_probs.append(probs)

        return torch.cat(all_probs, dim=0)

    def get_probs(self) -> torch.Tensor:
        raise NotImplementedError(
            "Use get_probs_batch() for batched inference. "
            "For sequential inference, use InferenceSession."
        )
