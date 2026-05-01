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
from torch.utils.data import DataLoader

import wandb
from krasnal.config import (
    ARTIFACTS_DIR,
    EVAL_DATASET_PATH,
    SFT_COT_SHARDS_DIR,
    GPTConfig,
    TrainConfig,
)
from krasnal.dataset import ChessDataset, make_collate_fn
from krasnal.eval import ChessEvaluator, get_stockfish_client
from krasnal.sft.train import (
    RandomTokenSource,
    compute_batch_sizes,
    resolve_shard_paths,
    split_shard_paths,
)
from krasnal.tokens import get_vocab_size, set_side_prefixed_moves
from krasnal.trainer import (
    build_model,
    cosine_warmup_lr,
    resolve_pretrained_checkpoint,
    run_supervised_training,
    save_model_state,
    setup_runtime,
    unwrap_model,
)
from krasnal.utils import (
    format_eval_metric_key,
    init_wandb,
    print_model_config,
    save_wandb_run,
    set_seed,
)


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
    set_side_prefixed_moves(bool(cfg.get("side_prefixed_moves", True)))
    set_seed(cfg.seed)

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

    cot_batch_size, normal_batch_size = compute_batch_sizes(tconf.batch_size, cfg.cot_ratio)

    cot_source = RandomTokenSource(cot_train_paths, seed=cfg.seed + 1)
    normal_source = RandomTokenSource(normal_dataset_path, seed=cfg.seed)

    cot_len = len(cot_source.dataset)
    steps_per_epoch = max(1, math.ceil(cot_len / tconf.batch_size))

    if cfg.max_iters is not None:
        total_iters = cfg.max_iters
    elif tconf.epochs > 0:
        total_iters = max(1, math.ceil(tconf.epochs * steps_per_epoch))
    else:
        raise ValueError("Either --max-iters > 0 or TrainConfig.epochs must be > 0")

    tconf.max_iters = total_iters

    device, dtype, ctx, scaler = setup_runtime()
    model.to(device)

    optimizer = model.configure_optimizers(
        weight_decay=tconf.weight_decay,
        learning_rate=tconf.learning_rate,
        betas=(tconf.beta1, tconf.beta2),
        device_type=device.type,
    )

    if tconf.compile and device.type == "cuda":
        model = torch.compile(model, fullgraph=True, dynamic=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_DIR / "sft_cot" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = get_vocab_size()

    run_config = {
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
    }

    run_id, entity, project = init_wandb(
        project=cfg.wandb_project,
        config=run_config,
        stage="sft_cot_train",
    )
    wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

    print_model_config(
        stage="SFT CoT",
        params_m=model.get_num_params() / 1_000_000,
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

    eval_dataset = ChessDataset(EVAL_DATASET_PATH)
    cot_eval_dataset = ChessDataset(cot_eval_paths)
    val_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        pin_memory=tconf.pin_memory,
        batch_size=tconf.batch_size,
        num_workers=tconf.num_workers,
        collate_fn=collate,
    )

    stockfish = get_stockfish_client(depth=cfg.eval.stockfish.depth)
    classical_evaluator = ChessEvaluator(
        metrics=list(cfg.eval.metrics),
        stockfish=stockfish,
        seed=cfg.seed,
        acpl_sample_size=cfg.eval.stockfish.acpl_sample_size,
        qa_config=OmegaConf.to_container(cfg.eval.qa, resolve=True),
    )
    cot_evaluator = ChessEvaluator(
        metrics=list(cfg.eval.cot_metrics),
        cot=True,
        stockfish=stockfish,
        seed=cfg.seed,
        qa_config=OmegaConf.to_container(cfg.eval.qa, resolve=True),
    )
    eval_device = torch.device(device)

    def log_fn(iter_num: int, last_loss_value: float, epoch_float: float) -> None:
        wandb.log({"train_loss": last_loss_value, "epoch": epoch_float}, step=iter_num)

    def eval_fn(model: torch.nn.Module, _iter_num: int) -> dict[str, Any]:
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
        payload = {}
        for k, v in metrics.items():
            if k.startswith("qa/what_is_on/f1_per_square/"):
                continue
            payload[format_eval_metric_key(k)] = v
        if "qa/what_is_on/f1_matrix" in metrics:
            heatmap = metrics["qa/what_is_on/f1_matrix"]
            payload[format_eval_metric_key("qa/what_is_on/f1_matrix")] = heatmap
            wandb.run.summary["eval/qa/what_is_on/f1_matrix"] = heatmap  # type: ignore[index]
        wandb.log(payload)

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
    )

    model_path = artifact_dir / "model.pt"
    save_model_state(unwrap_model(model), model_path)

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
