import math
import os
import shutil
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from krasnal.config import ARTIFACTS_DIR, GPTConfig, TrainConfig
from krasnal.model import GPT
from krasnal.supervised_target_mask import LOSS_IGNORE_INDEX
from krasnal.tokens import get_vocab_size


@dataclass(frozen=True)
class DistributedInfo:
    """Process group info; when ``enabled`` is False, rank is always 0 and world_size is 1."""

    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedInfo(enabled=False, rank=0, world_size=1, local_rank=0)
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA and torchrun.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedInfo(
        enabled=True,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
    )


def teardown_distributed(dinfo: DistributedInfo) -> None:
    if dinfo.enabled:
        dist.destroy_process_group()


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


def setup_runtime(
    *,
    device: torch.device | None = None,
) -> tuple[torch.device, torch.dtype, AbstractContextManager, torch.amp.GradScaler]:
    if device is not None:
        selected_device = device
    elif torch.cuda.is_available():
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
    m: torch.nn.Module = model
    while True:
        if isinstance(m, DDP):
            m = m.module
            continue
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod  # type: ignore[assignment]
            continue
        break
    return m


def save_model_state(
    model: torch.nn.Module,
    out_path: Path,
    *,
    move_vocab_path: Path | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    torch.save(raw_model.state_dict(), out_path)
    if move_vocab_path is not None:
        shutil.copyfile(move_vocab_path, out_path.parent / "move_vocab.json")


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
    dist_info: DistributedInfo | None = None,
    train_sampler: Any | None = None,
):
    dinfo = dist_info or DistributedInfo(False, 0, 1, 0)
    master = dinfo.is_master
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
        disable=not master,
    )

    model.train()
    epoch = 0
    while iter_num < max_iters:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
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

            if master and iter_num % log_interval == 0:
                epoch_float = iter_num / max(steps_per_epoch, 1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr:.2e}",
                    epoch=f"{epoch_float:.2f}",
                )
                if log_fn is not None:
                    log_fn(iter_num, last_loss_value, epoch_float)

            if master and iter_num % eval_interval == 0:
                raw_model = unwrap_model(model)
                raw_model.eval()
                with torch.inference_mode():
                    val_losses = [
                        raw_model(
                            xv.to(device, non_blocking=True),
                            yv.to(device, non_blocking=True),
                            ignore_index=LOSS_IGNORE_INDEX,
                        )[1].item()
                        for xv, yv in val_loader
                    ]
                raw_model.train()
                n_val = len(val_losses)
                eval_metrics = {"val_loss": sum(val_losses) / n_val if n_val else float("nan")}
                eval_metrics.update(eval_fn(model, iter_num))
                eval_log_fn(iter_num, eval_metrics)

            pbar.update(1)
            iter_num += 1
            if iter_num >= max_iters:
                break
        epoch += 1

    pbar.close()
    if master:
        eval_log_fn(iter_num, eval_fn(model, iter_num))
    if dinfo.enabled:
        dist.barrier()
    return last_loss_value
