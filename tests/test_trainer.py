from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from krasnal.dataset import make_collate_fn
from krasnal.tokens import (
    ELO_2000_2099_ID,
    IS_CHECK_ID,
    MOVE_TO_ID,
    PAWN_ID,
    TC_RAPID_INC_ID,
    WHATS_ON_PROMPT_TOKEN_IDS,
    WHITE_WON_ID,
)
from krasnal.trainer import cosine_warmup_lr, run_supervised_training
from krasnal.utils import format_eval_metric_key


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


def test_collate_masks_conditioning_metadata_targets():
    collate = make_collate_fn()
    x, y = collate(
        [torch.tensor([0, TC_RAPID_INC_ID, WHITE_WON_ID, ELO_2000_2099_ID, 500], dtype=torch.long)]
    )

    assert x.tolist() == [[0, TC_RAPID_INC_ID, WHITE_WON_ID, ELO_2000_2099_ID]]
    assert y.tolist() == [[-100, -100, -100, 500]]


def test_collate_masks_is_check_targets():
    collate = make_collate_fn()
    x, y = collate([torch.tensor([10, IS_CHECK_ID, PAWN_ID], dtype=torch.long)])

    assert x.tolist() == [[10, IS_CHECK_ID]]
    assert y.tolist() == [[-100, PAWN_ID]]


def test_collate_masks_whats_on_prompt_tokens():
    collate = make_collate_fn()
    whats_on_e4 = MOVE_TO_ID["<whats_on_e4>"]
    answer = MOVE_TO_ID["<w:pawn>"]

    assert whats_on_e4 in WHATS_ON_PROMPT_TOKEN_IDS
    assert answer not in WHATS_ON_PROMPT_TOKEN_IDS

    x, y = collate([torch.tensor([10, whats_on_e4, answer], dtype=torch.long)])

    assert x.tolist() == [[10, whats_on_e4]]
    assert y.tolist() == [[-100, answer]]


def test_format_eval_metric_key_groups_game_metrics():
    assert format_eval_metric_key("illegal_mass") == "eval/game/illegal_mass"
    assert format_eval_metric_key("top1_legal") == "eval/game/top1_legal"
    key = "qa/what_is_on/accuracy_matrix"
    assert format_eval_metric_key(key) == "eval/qa/what_is_on/accuracy_matrix"
    assert format_eval_metric_key("val_loss") == "eval/val_loss"
