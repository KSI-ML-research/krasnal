from contextlib import nullcontext

import pytest

from trainer import cosine_warmup_lr, setup_runtime


class MockTrainConfig:
    def __init__(self):
        self.learning_rate = 1e-3
        self.min_lr = 1e-5
        self.warmup_iters = 100
        self.max_iters = 1000


def test_cosine_warmup_lr_warmup_phase():
    cfg = MockTrainConfig()
    lr_at_0 = cosine_warmup_lr(0, cfg)
    lr_at_50 = cosine_warmup_lr(50, cfg)
    lr_at_100 = cosine_warmup_lr(100, cfg)

    assert lr_at_0 == 0.0
    assert lr_at_50 == pytest.approx(cfg.learning_rate * 0.5)
    assert lr_at_100 == pytest.approx(cfg.learning_rate)


def test_cosine_warmup_lr_decay_phase():
    cfg = MockTrainConfig()
    lr_at_100 = cosine_warmup_lr(100, cfg)
    lr_at_550 = cosine_warmup_lr(550, cfg)

    assert lr_at_100 == pytest.approx(cfg.learning_rate)
    assert lr_at_550 < lr_at_100
    assert lr_at_550 > cfg.min_lr


def test_cosine_warmup_lr_at_max_iters():
    cfg = MockTrainConfig()
    lr = cosine_warmup_lr(1000, cfg)
    assert lr == pytest.approx(cfg.min_lr)


def test_cosine_warmup_lr_validation_warmup_not_positive():
    cfg = MockTrainConfig()
    cfg.warmup_iters = 0
    with pytest.raises(ValueError, match="warmup_iters must be positive"):
        cosine_warmup_lr(0, cfg)


def test_cosine_warmup_lr_validation_max_less_than_warmup():
    cfg = MockTrainConfig()
    cfg.warmup_iters = 100
    cfg.max_iters = 100
    with pytest.raises(ValueError, match="must be greater than"):
        cosine_warmup_lr(100, cfg)


def test_setup_runtime_cpu():
    device, device_type, dtype, ctx, scaler = setup_runtime(device="cpu")
    assert device == "cpu"
    assert device_type == "cpu"
    assert dtype == "float32"
    assert isinstance(ctx, nullcontext)
