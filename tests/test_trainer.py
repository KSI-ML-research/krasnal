from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from krasnal.dataset import make_collate_fn
from krasnal.tokens import IS_CHECK_ID
from krasnal.trainer import cosine_warmup_lr, run_supervised_training


class MockConfig:
    def __init__(self):
        self.learning_rate = 1e-3
        self.min_lr = 1e-5
        self.warmup_iters = 100
        self.max_iters = 1000


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 10)

    def forward(self, x, y, ignore_index=-100):
        logits = self.linear(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=ignore_index
        )
        return logits, loss


def test_warmup_at_zero():
    cfg = MockConfig()
    assert cosine_warmup_lr(0, cfg) == 0.0


def test_warmup_at_half():
    cfg = MockConfig()
    assert cosine_warmup_lr(50, cfg) == pytest.approx(cfg.learning_rate * 0.5)


def test_warmup_complete():
    cfg = MockConfig()
    assert cosine_warmup_lr(100, cfg) == pytest.approx(cfg.learning_rate)


def test_decay():
    cfg = MockConfig()
    lr_100 = cosine_warmup_lr(100, cfg)
    lr_550 = cosine_warmup_lr(550, cfg)
    assert lr_550 < lr_100
    assert lr_550 > cfg.min_lr


def test_at_max_iters():
    cfg = MockConfig()
    assert cosine_warmup_lr(1000, cfg) == pytest.approx(cfg.min_lr)


def test_warmup_validation():
    cfg = MockConfig()
    cfg.warmup_iters = 0
    with pytest.raises(ValueError, match="warmup_iters must be positive"):
        cosine_warmup_lr(0, cfg)


def test_max_less_than_warmup_validation():
    cfg = MockConfig()
    cfg.max_iters = 100
    with pytest.raises(ValueError, match="must be greater than"):
        cosine_warmup_lr(100, cfg)


def test_run_training_smoke():
    from krasnal.config import TrainConfig

    dataset = TensorDataset(torch.randn(20, 8), torch.randint(0, 10, (20,)))
    loader = DataLoader(dataset, batch_size=4)

    model = MockModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    config = TrainConfig(
        learning_rate=1e-3,
        min_lr=1e-5,
        epochs=1.0,
        warmup_iters=2,
        batch_size=4,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        grad_clip=0.0,
        log_interval=2,
        eval_interval=100,
        max_iters=10,
        steps_per_epoch=5,
    )

    result = run_supervised_training(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_config=config,
        device="cpu",
        ctx=nullcontext(),
        scaler=MagicMock(),
        lr_fn=lambda _: 1e-3,
        desc="test",
        log_fn=lambda *_: None,
        eval_fn=lambda *_: {},
        eval_log_fn=lambda *_: None,
        val_loader=loader,
    )

    assert result is not None


def test_run_training_keeps_eval_mode_during_eval_callback():
    from krasnal.config import TrainConfig

    class EvalModeModel(MockModel):
        def __init__(self):
            super().__init__()
            self.modes: list[bool] = []

        def eval(self):
            self.modes.append(False)
            return super().eval()

        def train(self, mode: bool = True):
            self.modes.append(mode)
            return super().train(mode)

    dataset = TensorDataset(torch.randn(20, 8), torch.randint(0, 10, (20,)))
    loader = DataLoader(dataset, batch_size=4)

    model = EvalModeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    config = TrainConfig(
        learning_rate=1e-3,
        min_lr=1e-5,
        epochs=1.0,
        warmup_iters=2,
        batch_size=4,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        grad_clip=0.0,
        log_interval=2,
        eval_interval=1,
        max_iters=1,
        steps_per_epoch=5,
    )

    seen_eval_modes: list[bool] = []

    def eval_fn(_model, _iter_num):
        seen_eval_modes.append(model.training)
        return {}

    run_supervised_training(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_config=config,
        device="cpu",
        ctx=nullcontext(),
        scaler=MagicMock(),
        lr_fn=lambda _: 1e-3,
        desc="test",
        log_fn=lambda *_: None,
        eval_fn=eval_fn,
        eval_log_fn=lambda *_: None,
        val_loader=loader,
    )

    assert seen_eval_modes == [False, False]


def test_collate_masks_is_check_targets():
    collate = make_collate_fn()
    x, y = collate([torch.tensor([10, IS_CHECK_ID, 11], dtype=torch.long)])

    assert x.tolist() == [[10, IS_CHECK_ID]]
    assert y.tolist() == [[-100, 11]]
