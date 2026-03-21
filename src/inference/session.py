from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from ..model import GPT
from ..tokenizer import SOS_ID, SPECIAL_TOKENS, Tokenizer
from .abstracts import BaseInferenceSession


class InferenceSession(BaseInferenceSession):
    """Stateful model inference session managing token context."""

    def __init__(
        self,
        model: GPT,
        device: torch.device,
        outcome_token: int = SOS_ID,
    ):
        self.model = model
        self.device = device
        self._amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        self.reset(outcome_token)

    def reset(self, outcome_token: int = SOS_ID) -> None:
        """Clear context and start a new game."""
        self.context: list[int] = [outcome_token]

    def feed(self, token_id: int) -> None:
        """Append a token to the context."""
        self.context.append(token_id)

    def revert_last_move(self, tokenizer: Tokenizer) -> bool:
        """Revert the most recent move and its preceding reasoning block."""
        if len(self.context) <= 1:
            return False

        special_ids = set(SPECIAL_TOKENS)
        move_idx = None

        for i in range(len(self.context) - 1, 0, -1):
            if self.context[i] not in special_ids:
                move_idx = i
                break

        if move_idx is None:
            return False

        del self.context[move_idx:]

        while self.context and self.context[-1] == tokenizer.think_end_id:
            start_idx = None
            for i in range(len(self.context) - 2, -1, -1):
                if self.context[i] == tokenizer.think_start_id:
                    start_idx = i
                    break
            if start_idx is None:
                break
            del self.context[start_idx:]

        return True

    def get_probs(self) -> torch.Tensor:
        """Return probability distribution over the next token."""
        x = torch.tensor([self.context], dtype=torch.long, device=self.device)
        with torch.inference_mode(), self._amp_ctx:
            logits, _ = self.model(x)
        return F.softmax(logits[0, -1], dim=-1)
