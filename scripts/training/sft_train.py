#!/usr/bin/env python3
"""Offline Supervised CoT training that reads generated shards."""

import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import wandb
from krasnal.config import (
    ARTIFACTS_DIR,
    EVAL_DATASET_PATH,
    MOVE_VOCAB_PATH,
    SFT_COT_SHARDS_DIR,
    GPTConfig,
    TrainConfig,
)
from krasnal.dataset import ChessDataset, make_collate_fn
from krasnal.eval import chess_evaluator_from_config
from krasnal.sft.train import (
    RandomTokenSource,
    compute_batch_sizes,
    resolve_shard_paths,
    split_shard_paths,
)
from krasnal.tokens import get_vocab_size, load_move_vocab
from krasnal.trainer import (
    DistributedInfo,
    build_model,
    cosine_warmup_lr,
    resolve_pretrained_checkpoint,
    run_supervised_training,
    save_model_state,
    setup_distributed,
    setup_runtime,
    teardown_distributed,
    unwrap_model,
)
from krasnal.utils import (
    init_wandb,
    log_eval_metrics_to_wandb,
    print_model_config,
    save_wandb_run,
    set_seed,
)


def build_run_config(
    cfg: DictConfig,
    tconf: TrainConfig,
    mconf: GPTConfig,
    *,
    cot_train_paths: list[Path],
    cot_eval_paths: list[Path],
    normal_dataset_path: Path,
    vocab_size: int,
    total_iters: int,
    dist_info: DistributedInfo,
) -> dict[str, Any]:
    piece_aware_moves = bool(cfg.get("piece_aware_moves", False))
    side_prefixed_moves = bool(cfg.get("side_prefixed_moves", True))
    return {
        "stage": "sft_cot_train",
        "seed": cfg.seed,
        "cot_shards_dir": str(SFT_COT_SHARDS_DIR),
        "cot_train_shards": len(cot_train_paths),
        "cot_eval_shards": len(cot_eval_paths),
        "normal_dataset": str(normal_dataset_path),
        "cot_ratio": cfg.cot_ratio,
        "batch_size": tconf.batch_size,
        "learning_rate": tconf.learning_rate,
        "max_iters": total_iters,
        "epochs": tconf.epochs,
        "gpt_model_name": cfg.model.get("name", "custom"),
        "vocab_size": vocab_size,
        "block_size": mconf.block_size,
        "n_layer": mconf.n_layer,
        "n_head": mconf.n_head,
        "n_embd": mconf.n_embd,
        "dropout": mconf.dropout,
        "bias": mconf.bias,
        "piece_aware_moves": piece_aware_moves,
        "side_prefixed_moves": side_prefixed_moves,
        "move_vocab_path": str(MOVE_VOCAB_PATH),
        "world_size": dist_info.world_size,
        "ddp": dist_info.enabled,
    }


def mixed_batch_generator(
    cot_source: RandomTokenSource,
    normal_source: RandomTokenSource,
    cot_batch_size: int,
    normal_batch_size: int,
    collate: Callable,
):
    while True:
        cot_batch = cot_source.sample_sequences(cot_batch_size)
        normal_batch = normal_source.sample_sequences(normal_batch_size)
        batch = cot_batch + normal_batch
        if not batch:
            raise ValueError("Both CoT and normal datasets are empty")
        permutation = torch.randperm(len(batch))
        batch = [batch[idx] for idx in permutation.tolist()]
        x, y = collate(batch)
        yield x, y


@hydra.main(version_base=None, config_path="../../config", config_name="sft_train")
def main(cfg: DictConfig) -> None:
    dist_info = setup_distributed()
    try:
        _main(cfg, dist_info)
    finally:
        teardown_distributed(dist_info)


def _main(cfg: DictConfig, dist_info: DistributedInfo) -> None:
    piece_aware_moves = bool(cfg.get("piece_aware_moves", False))
    side_prefixed_moves = bool(cfg.get("side_prefixed_moves", True))
    load_move_vocab(
        MOVE_VOCAB_PATH,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )
    set_seed(cfg.seed + dist_info.rank)

    shard_paths = resolve_shard_paths(SFT_COT_SHARDS_DIR)
    cot_train_paths, cot_eval_paths = split_shard_paths(
        shard_paths,
        eval_fraction=float(cfg.cot_eval_shard_fraction),
    )
    normal_dataset_path = Path(cfg.normal_dataset)
    if not normal_dataset_path.exists():
        raise FileNotFoundError(f"Normal dataset not found at {normal_dataset_path}")

    checkpoint_path = resolve_pretrained_checkpoint(cfg.model_path, cfg.latest_pretrain)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    mconf = GPTConfig(vocab_size=get_vocab_size(), **model_cfg)
    model = build_model(model_config=mconf)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    tconf = TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    collate = make_collate_fn(tconf.padding_bucket_sizes)
    if dist_info.enabled:
        scale = float(dist_info.world_size)
        tconf.learning_rate *= scale
        tconf.min_lr *= scale

    cot_batch_size, normal_batch_size = compute_batch_sizes(tconf.batch_size, cfg.cot_ratio)

    cot_source = RandomTokenSource(
        cot_train_paths,
        seed=cfg.seed + 1 + dist_info.rank,
        include_elo=cfg.get("include_elo", True),
    )
    normal_source = RandomTokenSource(
        normal_dataset_path,
        seed=cfg.seed + dist_info.rank,
        include_elo=cfg.get("include_elo", True),
    )

    cot_len = len(cot_source.dataset)
    steps_per_epoch = max(1, math.ceil(cot_len / tconf.batch_size))

    if cfg.max_iters is not None:
        total_iters = cfg.max_iters
    elif tconf.epochs > 0:
        total_iters = max(1, math.ceil(tconf.epochs * steps_per_epoch))
    else:
        raise ValueError("Either --max-iters > 0 or TrainConfig.epochs must be > 0")

    tconf.max_iters = total_iters

    train_device = torch.device("cuda", dist_info.local_rank) if dist_info.enabled else None
    device, dtype, ctx, scaler = setup_runtime(device=train_device)
    model.to(device)

    optimizer = model.configure_optimizers(
        weight_decay=tconf.weight_decay,
        learning_rate=tconf.learning_rate,
        betas=(tconf.beta1, tconf.beta2),
        device_type=device.type,
    )

    if tconf.compile and device.type == "cuda":
        model = torch.compile(model, fullgraph=True, dynamic=False)

    if dist_info.enabled:
        model = DDP(model, device_ids=[dist_info.local_rank])

    artifact_dir = None
    wandb_run_url = ""
    run_config: dict[str, Any] = {}
    vocab_size = get_vocab_size()

    if dist_info.is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = ARTIFACTS_DIR / "sft_cot" / timestamp
        artifact_dir.mkdir(parents=True, exist_ok=True)
        run_config = build_run_config(
            cfg,
            tconf,
            mconf,
            cot_train_paths=cot_train_paths,
            cot_eval_paths=cot_eval_paths,
            normal_dataset_path=normal_dataset_path,
            vocab_size=vocab_size,
            total_iters=total_iters,
            dist_info=dist_info,
        )

        run_id, entity, project = init_wandb(
            project=cfg.wandb_project,
            config=run_config,
            stage="sft_cot_train",
        )
        wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

        print_model_config(
            stage="SFT CoT",
            params_m=unwrap_model(model).get_num_params() / 1_000_000,
            dataset_size=tconf.batch_size,
            dataset_label="batch",
            config=mconf,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
            compile_enabled=tconf.compile,
            artifact_dir=artifact_dir,
        )

    train_loader = mixed_batch_generator(
        cot_source=cot_source,
        normal_source=normal_source,
        cot_batch_size=cot_batch_size,
        normal_batch_size=normal_batch_size,
        collate=collate,
    )

    eval_dataset = ChessDataset(EVAL_DATASET_PATH, include_elo=cfg.get("include_elo", True))
    cot_eval_dataset = ChessDataset(cot_eval_paths, include_elo=cfg.get("include_elo", True))
    val_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        pin_memory=tconf.pin_memory,
        batch_size=tconf.batch_size,
        num_workers=tconf.num_workers,
        collate_fn=collate,
    )

    classical_evaluator = (
        chess_evaluator_from_config(cfg, metrics=list(cfg.eval.metrics))
        if dist_info.is_master
        else None
    )
    cot_evaluator = (
        chess_evaluator_from_config(cfg, metrics=list(cfg.eval.cot_metrics), cot=True)
        if dist_info.is_master
        else None
    )
    eval_device = torch.device(device)

    def log_fn(iter_num: int, last_loss_value: float, epoch_float: float) -> None:
        wandb.log({"train_loss": last_loss_value, "epoch": epoch_float}, step=iter_num)

    def eval_fn(model: torch.nn.Module, _iter_num: int) -> dict[str, Any]:
        if classical_evaluator is None or cot_evaluator is None:
            return {}
        raw_model = unwrap_model(model)
        metrics = classical_evaluator.evaluate(raw_model, eval_dataset, 100, eval_device)
        cot_num_games = min(100, len(cot_eval_dataset))
        if cot_num_games > 0:
            metrics.update(
                cot_evaluator.evaluate(
                    raw_model,
                    cot_eval_dataset,
                    cot_num_games,
                    eval_device,
                )
            )
        return metrics

    def eval_log_fn(_iter_num: int, metrics: dict[str, Any]) -> None:
        log_eval_metrics_to_wandb(metrics)

    tconf.steps_per_epoch = steps_per_epoch

    run_supervised_training(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_config=tconf,
        device=device,
        ctx=ctx,
        scaler=scaler,
        lr_fn=lambda i: cosine_warmup_lr(i, tconf),
        desc="sft-train",
        log_fn=log_fn,
        eval_fn=eval_fn,
        eval_log_fn=eval_log_fn,
        val_loader=val_loader,
        dist_info=dist_info,
    )

    if dist_info.is_master:
        assert artifact_dir is not None
        model_path = artifact_dir / "model.pt"
        save_model_state(unwrap_model(model), model_path, move_vocab_path=MOVE_VOCAB_PATH)

        save_wandb_run(
            artifact_dir=artifact_dir,
            run_config=run_config,
            wandb_run_url=wandb_run_url,
            artifact_name="sft_cot",
            artifact_type="model",
        )
        wandb.finish()

        print(f"Saved SFT CoT model to {model_path}")


if __name__ == "__main__":
    main()
