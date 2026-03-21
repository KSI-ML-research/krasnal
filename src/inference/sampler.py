from __future__ import annotations

import torch
import torch.nn.functional as F

from inference.abstracts import BaseSampler


class DefaultSampler(BaseSampler):
    """Temperature and nucleus (top-p) sampler for token selection.

    Applies temperature scaling to logits/probabilities, then performs nucleus
    (top-p) sampling to select a token ID. Falls back to greedy (argmax) when
    temperature is zero. This is the default sampler used during inference.
    """

    def sample(
        self,
        probs: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> int:
        if temperature == 0.0:
            return int(torch.argmax(probs).item())

        if temperature != 1.0:
            probs = F.softmax(torch.log(probs + 1e-10) / temperature, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumulative_probs > top_p
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = False
            probs[sorted_indices[to_remove]] = 0
            probs = probs / (probs.sum() + 1e-10)

        return int(torch.multinomial(probs, num_samples=1).item())
