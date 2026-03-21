from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from inference.abstracts import BaseInferenceSession
from model import GPT
from tokenizer import SOS_ID


class InferenceSession(BaseInferenceSession):
    """Concrete inference session that runs a full forward pass per `get_probs()`.

    Tokens are accumulated in `self.context` and the entire sequence is re-passed
    through the model on each call. This is the baseline implementation.

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

    def get_probs(self) -> torch.Tensor:
        """Return probability distribution over the next token."""
        x = torch.tensor([self.context], dtype=torch.long, device=self.device)
        with torch.inference_mode(), self._amp_ctx:
            logits, _ = self.model(x)
        return F.softmax(logits[0, -1], dim=-1)
