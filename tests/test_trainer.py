from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from trainer import cosine_warmup_lr, run_supervised_training, setup_runtime


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


def test_run_supervised_training_smoke():
    class MockConfig:
        def __init__(self):
            self.learning_rate = 1e-3
            self.min_lr = 1e-5
            self.warmup_iters = 2
            self.max_iters = 10

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(8, 10)

        def forward(self, x, y, ignore_index=-100):
            logits = self.net(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=ignore_index
            )
            return logits, loss

        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())

    dataset = TensorDataset(torch.randn(20, 8), torch.randint(0, 10, (20,)))
    loader = DataLoader(dataset, batch_size=4)

    model = MockModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    device = "cpu"
    config = MockConfig()

    log_calls = []

    def lr_fn(iter_num):
        return cosine_warmup_lr(iter_num, config)

    def log_fn(iter_num, loss, epoch):
        log_calls.append((iter_num, loss, epoch))

    result = run_supervised_training(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        max_iters=10,
        steps_per_epoch=5,
        device=device,
        ctx=nullcontext(),
        scaler=MagicMock(),
        grad_clip=0.0,
        pad_id=-100,
        lr_fn=lr_fn,
        desc="test",
        log_interval=2,
        log_fn=log_fn,
    )

    assert result is not None
    assert len(log_calls) == 5
    for iter_num, loss, epoch in log_calls:
        assert isinstance(iter_num, int)
        assert isinstance(loss, float)
        assert isinstance(epoch, float)
