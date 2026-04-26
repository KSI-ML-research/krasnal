"""Batch inference for evaluating multiple positions at once."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from krasnal.inference.game import Game
from krasnal.inference.utils import create_amp_context
from krasnal.tokens import PAD_ID, QA_TOKEN_IDS, legal_token_ids

if TYPE_CHECKING:
    from krasnal.model import GPT


class StatelessBatchInferenceSession:
    """Stateless batched inference for evaluating multiple positions simultaneously.

    This class processes multiple token sequences in a single forward pass,
    which is significantly faster than sequential inference for evaluation tasks.

    Example:
        >>> session = StatelessBatchInferenceSession(model, device)
        >>> sequences = [[GAME_START_ID, 100, 200], [GAME_START_ID, 150, 220, 300]]
        >>> probs = session.get_raw_probs_batch(sequences)
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

    def get_raw_logits_batch(
        self,
        sequences: list[list[int]],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Run forward pass on a batch of sequences in chunks.

        Processes multiple token sequences in a single forward pass for
        improved throughput. Sequences are padded to the maximum length
        within each chunk.

        Args:
            sequences: List of token sequences. Each sequence is a list of token IDs.
                       The first token is assumed to be the game start token.
            batch_size: Number of sequences to process at once. Default 256.

        Returns:
            Tensor of shape (batch_size, vocab_size) containing next-token
            logits at each position.
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
                seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
                padded[j, : len(seq)] = seq_tensor
                lengths[j] = len(seq)

            with torch.inference_mode(), self._amp_ctx:
                logits, _ = self.model(padded, padded, ignore_index=PAD_ID)

            last_token_idx = lengths - 1
            batch_idx = torch.arange(len(chunk), device=self.device)
            next_token_logits = logits[batch_idx, last_token_idx, :]
            all_probs.append(next_token_logits)

        return torch.cat(all_probs, dim=0)

    def get_raw_probs_batch(
        self,
        sequences: list[list[int]],
        batch_size: int = 256,
    ) -> torch.Tensor:
        return torch.softmax(self.get_raw_logits_batch(sequences, batch_size=batch_size), dim=-1)

    def get_legal_logits_batch(
        self,
        games: list[Game],
        batch_size: int = 256,
    ) -> torch.Tensor:
        logits = self.get_raw_logits_batch(
            [game.context_tokens() for game in games],
            batch_size=batch_size,
        )
        masked = torch.full_like(logits, float("-inf"))
        for idx, game in enumerate(games):
            legal_ids = legal_token_ids(game.board)
            if legal_ids:
                masked[idx, legal_ids] = logits[idx, legal_ids]
        masked[:, list(QA_TOKEN_IDS)] = float("-inf")
        return masked

    def get_legal_probs_batch(
        self,
        games: list[Game],
        batch_size: int = 256,
    ) -> torch.Tensor:
        return torch.softmax(
            self.get_legal_logits_batch(games, batch_size=batch_size),
            dim=-1,
        )
