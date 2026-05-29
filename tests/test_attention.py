import torch

from krasnal.config import GPTConfig
from krasnal.model import GPT


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        use_time_conditioning=False,
        time_conditioning_hidden=8,
        vocab_size=64,
        dropout=0.0,
    )


def test_forward_uses_standard_causal_attention():
    model = GPT(_tiny_config())
    idx = torch.tensor([[10, 11, 20, 21]], dtype=torch.long)
    targets = torch.tensor([[11, 12, 21, 22]], dtype=torch.long)

    _, loss = model(idx, targets)

    assert loss is not None
    assert torch.isfinite(loss)
