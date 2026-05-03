from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch

from krasnal.config import GPTConfig
from krasnal.model import GPT
from krasnal.tokens import get_vocab_size


def create_amp_context(device: torch.device) -> AbstractContextManager:
    """Create AMP autocast context for CUDA devices.

    Returns nullcontext for non-CUDA devices.
    """
    return (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def load_model(model_path: str, device: torch.device, config: GPTConfig) -> GPT:
    """Load a trained chess model from a checkpoint."""
    if config.vocab_size is None:
        config.vocab_size = get_vocab_size()
    model = GPT(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
