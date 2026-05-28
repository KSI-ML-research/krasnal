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
from torch.optim import Muon
from tqdm.auto import tqdm

from krasnal.config import GPTConfig, TrainConfig
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


class CombinedOptimizer:
    """Wraps two optimizers (Muon + AdamW) so the training loop can treat them as one."""

    def __init__(self, muon_opt: torch.optim.Optimizer, adam_opt: torch.optim.AdamW) -> None:
        self.muon_opt = muon_opt
        self.adam_opt = adam_opt
        # Tag each group with its initial LR so the cosine schedule can scale
        # proportionally rather than overwriting with a single absolute value.
        for g in adam_opt.param_groups + muon_opt.param_groups:
            g.setdefault("initial_lr", g["lr"])
        # Expose param_groups for LR scheduling (training loop sets LR on these)
        self.param_groups = adam_opt.param_groups + muon_opt.param_groups

    def step(self, closure=None) -> None:
        self.muon_opt.step(closure)
        self.adam_opt.step(closure)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adam_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "muon": self.muon_opt.state_dict(),
            "adam": self.adam_opt.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon_opt.load_state_dict(state_dict["muon"])
        self.adam_opt.load_state_dict(state_dict["adam"])


def build_optimizer(
    model: GPT,
    train_config: TrainConfig,
    device_type: str,
) -> torch.optim.Optimizer | CombinedOptimizer:
    """Build optimizer: pure AdamW or Muon+AdamW hybrid."""
    if train_config.optimizer == "adamw":
        return model.configure_optimizers(
            weight_decay=train_config.weight_decay,
            learning_rate=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            device_type=device_type,
        )

    if train_config.optimizer != "muon":
        raise ValueError(
            f"Unknown optimizer: {train_config.optimizer!r}. Choose 'adamw' or 'muon'."
        )

    # Muon+AdamW hybrid: Muon for 2D weights, AdamW for the rest
    muon_params = []
    adam_decay_params = []
    adam_nodecay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Embedding weights are tied with lm_head — use AdamW (Muon doesn't
        # benefit from embeddings, and weight tying complicates orthogonalization)
        if "wte" in name or "lm_head" in name:
            adam_decay_params.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_nodecay_params.append(p)

    muon_opt = Muon(
        muon_params,
        lr=train_config.muon_lr,
        momentum=train_config.muon_momentum,
        # Keller-style Muon: no weight decay on hidden 2D weights (AdamW still decays embeddings).
        weight_decay=0.0,
    )

    adam_opt = torch.optim.AdamW(
        [
            {"params": adam_decay_params, "weight_decay": train_config.weight_decay},
            {"params": adam_nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
    )

    return CombinedOptimizer(muon_opt, adam_opt)


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


def apply_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
    train_config: TrainConfig,
) -> None:
    """Set per-param-group learning rates for the current training step."""
    for param_group in optimizer.param_groups:
        if "initial_lr" in param_group:
            # CombinedOptimizer: scale each group proportionally. lr is the
            # AdamW-scheduled value; apply the cosine ratio to each group's base.
            ratio = lr / train_config.learning_rate
            param_group["lr"] = param_group["initial_lr"] * ratio
        else:
            param_group["lr"] = lr


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


def _unpack_supervised_batch(batch):
    if len(batch) == 2:
        x, y = batch
        return x, None, None, y, None, None
    if len(batch) == 4:
        x, active_x, opponent_x, y = batch
        return x, active_x, opponent_x, y, None, None
    if len(batch) == 6:
        x, active_x, opponent_x, y, segment_x, position_x = batch
        return x, active_x, opponent_x, y, segment_x, position_x
    raise ValueError(f"Expected a 2-, 4-, or 6-item supervised batch, got {len(batch)} items")


def _require_clock_tensors_if_time_model(
    model: torch.nn.Module,
    active: torch.Tensor | None,
    opponent: torch.Tensor | None,
) -> None:
    cfg = getattr(unwrap_model(model), "config", None)
    if cfg is not None and cfg.use_time_conditioning and (active is None or opponent is None):
        raise ValueError(
            "use_time_conditioning is True but this batch has no clock tensors. "
            "Use a dataset with active_clock_ids/opponent_clock_ids columns "
            "and the standard collate."
        )


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
        for batch in train_loader:
            x, active_x, opponent_x, y, segment_x, position_x = _unpack_supervised_batch(batch)
            _require_clock_tensors_if_time_model(model, active_x, opponent_x)
            lr = lr_fn(iter_num)
            apply_optimizer_lr(optimizer, lr, train_config)

            x = x.to(device, non_blocking=True)
            active_x = active_x.to(device, non_blocking=True) if active_x is not None else None
            opponent_x = (
                opponent_x.to(device, non_blocking=True) if opponent_x is not None else None
            )
            y = y.to(device, non_blocking=True)
            segment_x = segment_x.to(device, non_blocking=True) if segment_x is not None else None
            position_x = (
                position_x.to(device, non_blocking=True) if position_x is not None else None
            )

            with ctx:
                _, loss = model(
                    x,
                    y,
                    ignore_index=LOSS_IGNORE_INDEX,
                    active_clock_ids=active_x,
                    opponent_clock_ids=opponent_x,
                    segment_ids=segment_x,
                    position_ids=position_x,
                )
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

            if iter_num > 0 and iter_num % eval_interval == 0:
                if dinfo.enabled:
                    dist.barrier()
                if master:
                    raw_model = unwrap_model(model)
                    raw_model.eval()
                    with torch.inference_mode():
                        val_losses = []
                        for val_batch in val_loader:
                            (
                                xv,
                                active_xv,
                                opponent_xv,
                                yv,
                                segment_xv,
                                position_xv,
                            ) = _unpack_supervised_batch(val_batch)
                            _require_clock_tensors_if_time_model(model, active_xv, opponent_xv)
                            segment_xv = (
                                segment_xv.to(device, non_blocking=True)
                                if segment_xv is not None
                                else None
                            )
                            position_xv = (
                                position_xv.to(device, non_blocking=True)
                                if position_xv is not None
                                else None
                            )
                            val_losses.append(
                                raw_model(
                                    xv.to(device, non_blocking=True),
                                    yv.to(device, non_blocking=True),
                                    ignore_index=LOSS_IGNORE_INDEX,
                                    active_clock_ids=active_xv.to(device, non_blocking=True)
                                    if active_xv is not None
                                    else None,
                                    opponent_clock_ids=opponent_xv.to(device, non_blocking=True)
                                    if opponent_xv is not None
                                    else None,
                                    segment_ids=segment_xv,
                                    position_ids=position_xv,
                                )[1].item()
                            )
                    raw_model.train()
                    n_val = len(val_losses)
                    eval_metrics = {"val_loss": sum(val_losses) / n_val if n_val else float("nan")}
                    eval_metrics.update(eval_fn(model, iter_num))
                    eval_log_fn(iter_num, eval_metrics)
                if dinfo.enabled:
                    dist.barrier()

            pbar.update(1)
            iter_num += 1
            if iter_num >= max_iters:
                break
        epoch += 1

    pbar.close()
    if dinfo.enabled:
        dist.barrier()
    if master:
        eval_log_fn(iter_num, eval_fn(model, iter_num))
    if dinfo.enabled:
        dist.barrier()
    return last_loss_value
