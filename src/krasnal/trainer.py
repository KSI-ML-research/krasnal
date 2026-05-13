import math
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from krasnal.config import ARTIFACTS_DIR, GPTConfig, TrainConfig
from krasnal.dataset import LOSS_IGNORE_INDEX, ChessDataset, make_collate_fn
from krasnal.model import GPT
from krasnal.tokens import get_vocab_size, save_to_json


def build_model(model_config: GPTConfig) -> GPT:
    if model_config.vocab_size is None:
        model_config.vocab_size = get_vocab_size()
    return GPT(model_config)


def resolve_pretrained_checkpoint(model_path: str | None, latest: bool) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if not latest:
        raise ValueError("Either --model or --latest-pretrain must be specified")

    pretrain_dirs = sorted(
        (ARTIFACTS_DIR / "pretrain").iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for d in pretrain_dirs:
        model_file = d / "model.pt"
        if model_file.exists():
            return model_file

    raise FileNotFoundError(
        f"No pretrained checkpoint found in {ARTIFACTS_DIR / 'pretrain'}. "
        "Run pretrain first or specify --model."
    )


def evaluate_loss(
    model,
    dataset_path: Path | list[Path],
    batch_size: int,
    num_workers: int,
    device: str,
) -> float:
    eval_dataset = ChessDataset(dataset_path)
    collate = make_collate_fn()
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        pin_memory=(device == "cuda"),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, loss = model(x, y, ignore_index=LOSS_IGNORE_INDEX)
            valid_tokens = (y != LOSS_IGNORE_INDEX).sum().item()
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
    model.train()
    if total_tokens == 0:
        raise ValueError("Eval dataset has no valid tokens")
    return total_loss / total_tokens


def setup_runtime() -> tuple[
    torch.device, torch.dtype, AbstractContextManager, torch.amp.GradScaler
]:
    """Build common runtime objects used by training scripts."""
    if torch.cuda.is_available():
        selected_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        selected_device = torch.device("mps")
    else:
        selected_device = torch.device("cpu")

    if selected_device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif selected_device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    ctx = (
        torch.amp.autocast(device_type=selected_device.type, dtype=dtype)
        if selected_device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(selected_device.type, enabled=(dtype == torch.float16))
    return selected_device, dtype, ctx, scaler


def cosine_warmup_lr(iter_num: int, train_config: TrainConfig) -> float:
    """Cosine annealing learning rate schedule with warmup.

    Args:
        iter_num: Current iteration number.
        train_config: Config object with learning_rate, min_lr, warmup_iters, max_iters.

    Returns:
        The learning rate for the current iteration.
    """
    if train_config.warmup_iters <= 0:
        raise ValueError(f"warmup_iters must be positive, got {train_config.warmup_iters}")
    if iter_num < train_config.warmup_iters:
        return train_config.learning_rate * iter_num / train_config.warmup_iters

    if iter_num > train_config.max_iters:
        return train_config.min_lr

    if train_config.max_iters <= train_config.warmup_iters:
        raise ValueError(
            f"max_iters ({train_config.max_iters}) must be greater than "
            f"warmup_iters ({train_config.warmup_iters})"
        )
    decay_ratio = (iter_num - train_config.warmup_iters) / (
        train_config.max_iters - train_config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap compiled model to get original for saving weights."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_model_state(model: torch.nn.Module, out_path: Path) -> None:
    """Save model state_dict and vocabulary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    torch.save(raw_model.state_dict(), out_path)
    save_to_json(out_path.parent / "vocab.json")


def run_supervised_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable,
    train_config: TrainConfig,
    *,
    device: str,
    ctx: AbstractContextManager,
    scaler: torch.amp.GradScaler,
    lr_fn: Callable[[int], float],
    desc: str,
    log_fn: Callable[[int, float, float], None] | None = None,
    eval_fn: Callable[[torch.nn.Module, int], dict[str, Any]],
    eval_log_fn: Callable[[int, dict[str, Any]], None],
    val_loader: Iterable,
):
    """Run a standard autoregressive supervised training loop.

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer.
        train_loader: DataLoader for training data.
        train_config: Training configuration (max_iters, grad_clip, log_interval, eval_interval).
        device: Device to train on.
        ctx: Autocast context.
        scaler: Gradient scaler.
        lr_fn: Learning rate schedule function (iter_num -> lr).
        desc: Description for progress bar.
        log_fn: Optional callback (iter_num, last_loss_value, epoch_float) for custom logging.
        eval_fn: Callback (model, iter_num) -> dict of eval metrics.
        eval_log_fn: Callback to log eval metrics.
        val_loader: DataLoader for validation data.

    Returns:
        The last loss value observed during training.
    """
    max_iters = train_config.max_iters
    steps_per_epoch = train_config.steps_per_epoch
    grad_clip = train_config.grad_clip
    log_interval = train_config.log_interval
    eval_interval = train_config.eval_interval
    iter_num = 0
    last_loss_value = None
    est_epochs = max_iters / max(steps_per_epoch, 1)
    pbar = tqdm(
        total=max_iters,
        desc=f"{desc} (~{est_epochs:.2f} ep)",
        unit="iter",
        dynamic_ncols=True,
    )

    model.train()
    while iter_num < max_iters:
        for x, y in train_loader:
            lr = lr_fn(iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                _, loss = model(x, y, ignore_index=LOSS_IGNORE_INDEX)
            last_loss_value = float(loss.item())

            scaler.scale(loss).backward()

            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # logging training metrics
            if iter_num % log_interval == 0:
                epoch_float = iter_num / max(steps_per_epoch, 1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr:.2e}",
                    epoch=f"{epoch_float:.2f}",
                )
                if log_fn is not None:
                    log_fn(iter_num, last_loss_value, epoch_float)

            # running evaluation and logging eval metrics
            if iter_num % eval_interval == 0:
                raw_model = unwrap_model(model)
                raw_model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.inference_mode():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(device, non_blocking=True)
                        y_val = y_val.to(device, non_blocking=True)
                        _, loss = raw_model(x_val, y_val, ignore_index=LOSS_IGNORE_INDEX)
                        val_loss_sum += loss.item()
                        val_batches += 1

                eval_metrics = {"val_loss": val_loss_sum / val_batches}
                eval_metrics.update(eval_fn(model, iter_num))
                eval_log_fn(iter_num, eval_metrics)
                raw_model.train()

            pbar.update(1)
            iter_num += 1
            if iter_num >= max_iters:
                break

    pbar.close()

    raw_model = unwrap_model(model)
    raw_model.eval()
    final_metrics = eval_fn(model, iter_num)
    eval_log_fn(iter_num, final_metrics)

    return last_loss_value
