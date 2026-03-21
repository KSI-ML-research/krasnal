import math
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .run_manager import update_run_config
from .tokenizer import Tokenizer, save_tokenizer_for_artifact


def setup_runtime(device: str | None = None):
    """Build common runtime objects used by training scripts."""
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in selected_device else "cpu"

    if device_type == "cuda":
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        dtype = "float32"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    return selected_device, device_type, dtype, ctx, scaler


def cosine_warmup_lr(iter_num, train_config):
    """Cosine annealing learning rate schedule with warmup."""
    if iter_num < train_config.warmup_iters:
        return train_config.learning_rate * iter_num / train_config.warmup_iters

    if iter_num > train_config.max_iters:
        return train_config.min_lr

    assert train_config.max_iters > train_config.warmup_iters
    decay_ratio = (iter_num - train_config.warmup_iters) / (
        train_config.max_iters - train_config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_model_state(model, out_path: Path, tokenizer: Tokenizer | None = None):
    """Save model state_dict exactly at out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = _unwrap_model(model)
    torch.save(raw_model.state_dict(), out_path)
    if tokenizer is not None:
        save_tokenizer_for_artifact(tokenizer, out_path)


def save_model_run(
    model,
    run_folder: Path,
    tokenizer: Tokenizer | None = None,
    final_loss: float | None = None,
    duration_seconds: float | None = None,
):
    """Save model to run folder with tokenizer and update config."""
    model_path = run_folder / "model.pt"
    save_model_state(model, model_path, tokenizer=tokenizer)

    if final_loss is not None or duration_seconds is not None:
        updates = {}
        if final_loss is not None:
            updates["final_loss"] = float(final_loss)
        if duration_seconds is not None:
            updates["duration_seconds"] = float(duration_seconds)
        update_run_config(run_folder, updates)

    return model_path


def run_supervised_training(
    model,
    optimizer,
    train_loader,
    *,
    max_iters: int,
    steps_per_epoch: int,
    device: str,
    ctx,
    scaler,
    grad_clip: float,
    pad_id: int,
    lr_fn: Callable[[int], float],
    desc: str,
    log_interval: int = 10,
):
    """Run a standard autoregressive supervised training loop."""
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
                _, loss = model(x, y, ignore_index=pad_id)
            last_loss_value = float(loss.item())

            scaler.scale(loss).backward()

            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if iter_num % log_interval == 0:
                epoch_float = iter_num / max(steps_per_epoch, 1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr:.2e}",
                    epoch=f"{epoch_float:.2f}",
                )

            pbar.update(1)
            iter_num += 1
            if iter_num >= max_iters:
                break

    pbar.close()
    return last_loss_value
