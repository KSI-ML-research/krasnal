import pytest
import torch

from krasnal.config import GPTConfig
from krasnal.model import GPT, MLP, _swiglu_hidden_dim


def _tiny_config(*, mlp_activation: str = "swiglu") -> GPTConfig:
    return GPTConfig(
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=64,
        use_clock_encodings=False,
        clock_encoding_hidden=32,
        vocab_size=128,
        mlp_activation=mlp_activation,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("activation", ["gelu", "swiglu", "relu2"])
def test_mlp_forward_preserves_shape(activation: str) -> None:
    config = _tiny_config(mlp_activation=activation)
    mlp = MLP(config)
    x = torch.randn(2, 5, config.n_embd)
    assert mlp(x).shape == x.shape


def test_swiglu_hidden_dim_aligns_to_multiple_of_8() -> None:
    assert _swiglu_hidden_dim(512) % 8 == 0


@pytest.mark.parametrize("activation", ["swiglu", "relu2"])
def test_gpt_mlp_activation_forward(activation: str) -> None:
    config = _tiny_config(mlp_activation=activation)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 4))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 4, config.vocab_size)
    assert loss is not None


def test_invalid_mlp_activation_raises() -> None:
    with pytest.raises(ValueError, match="mlp_activation"):
        MLP(_tiny_config(mlp_activation="relu"))  # type: ignore[arg-type]
