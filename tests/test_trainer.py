from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from krasnal.config import CLOCK_IGNORE_ID, GPTConfig, TrainConfig
from krasnal.dataset import make_collate_fn
from krasnal.model import GPT
from krasnal.tokens import (
    ELO_2000_2099_ID,
    IS_CHECK_ID,
    MOVE_TO_ID,
    TC_RAPID_INC_ID,
    WHATS_ON_PROMPT_TOKEN_IDS,
    WHITE_WON_ID,
)
from krasnal.trainer import (
    apply_optimizer_lr,
    build_optimizer,
    cosine_warmup_lr,
    run_supervised_training,
)
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

    def forward(self, x, y, ignore_index=-100, **_kwargs):
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


def test_muon_lr_schedule_uses_separate_base_lr():
    """Muon must keep its own base LR; applying AdamW's LR makes updates ~40x too small."""
    cfg = GPTConfig(
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        use_time_conditioning=False,
        time_conditioning_hidden=32,
        vocab_size=100,
    )
    model = GPT(cfg)
    train_config = TrainConfig(
        learning_rate=5e-4,
        min_lr=5e-5,
        epochs=1.0,
        warmup_iters=100,
        batch_size=8,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        log_interval=10,
        eval_interval=100,
        optimizer="muon",
        muon_lr=0.02,
        max_iters=1000,
    )
    optimizer = build_optimizer(model, train_config, "cpu")
    assert hasattr(optimizer, "muon_opt")

    lr = cosine_warmup_lr(100, train_config)
    apply_optimizer_lr(optimizer, lr, train_config)

    muon_lr = optimizer.muon_opt.param_groups[0]["lr"]
    adam_lr = optimizer.adam_opt.param_groups[0]["lr"]
    assert muon_lr == pytest.approx(train_config.muon_lr)
    assert adam_lr == pytest.approx(train_config.learning_rate)
    assert muon_lr > adam_lr * 10


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
    x, active_x, opponent_x, y = collate(
        [torch.tensor([0, TC_RAPID_INC_ID, WHITE_WON_ID, ELO_2000_2099_ID, 500], dtype=torch.long)]
    )

    assert x.tolist() == [[0, TC_RAPID_INC_ID, WHITE_WON_ID, ELO_2000_2099_ID]]
    assert active_x.shape == x.shape
    assert opponent_x.shape == x.shape
    assert y.tolist() == [[-100, -100, -100, 500]]


def test_collate_masks_is_check_targets():
    collate = make_collate_fn()
    x, _active_x, _opponent_x, y = collate([torch.tensor([10, IS_CHECK_ID, 500], dtype=torch.long)])

    assert x.tolist() == [[10, IS_CHECK_ID]]
    assert y.tolist() == [[-100, 500]]


def test_collate_masks_whats_on_prompt_tokens():
    collate = make_collate_fn()
    whats_on_e4 = MOVE_TO_ID["<whats_on_e4>"]
    answer = MOVE_TO_ID["<w:pawn>"]

    assert whats_on_e4 in WHATS_ON_PROMPT_TOKEN_IDS
    assert answer not in WHATS_ON_PROMPT_TOKEN_IDS

    x, _active_x, _opponent_x, y = collate(
        [torch.tensor([10, whats_on_e4, answer], dtype=torch.long)]
    )

    assert x.tolist() == [[10, whats_on_e4]]
    assert y.tolist() == [[-100, answer]]


def test_collate_shifts_clock_features_to_prediction_targets():
    collate = make_collate_fn()
    tokens = torch.tensor([500, 501, 502], dtype=torch.long)
    active_clocks = torch.tensor([CLOCK_IGNORE_ID, 30, 20], dtype=torch.long)
    opponent_clocks = torch.tensor([CLOCK_IGNORE_ID, 40, 35], dtype=torch.long)

    x, active_x, opponent_x, y = collate([(tokens, active_clocks, opponent_clocks)])

    assert x.tolist() == [[500, 501]]
    assert active_x.tolist() == [[30, 20]]
    assert opponent_x.tolist() == [[40, 35]]
    assert y.tolist() == [[501, 502]]


def test_format_eval_metric_key_groups_game_metrics():
    assert format_eval_metric_key("illegal_mass") == "eval/game/illegal_mass"
    assert format_eval_metric_key("top1_legal") == "eval/game/top1_legal"
    assert format_eval_metric_key("acc_when_low_time") == "eval/game/acc_when_low_time"
    key = "qa/what_is_on/acc_matrix"
    assert format_eval_metric_key(key) == "eval/qa/what_is_on/acc_matrix"
    assert format_eval_metric_key("val_loss") == "eval/val_loss"
