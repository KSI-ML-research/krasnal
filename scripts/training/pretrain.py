import math
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from krasnal.config import (
    ARTIFACTS_DIR,
    EVAL_DATASET_PATH,
    MOVE_VOCAB_PATH,
    PRETRAIN_DATASET_PATH,
    GPTConfig,
    TrainConfig,
)
from krasnal.dataset import ChessDataset, PretrainDataset, make_collate_fn, make_packed_collate_fn
from krasnal.eval import chess_evaluator_from_config
from krasnal.tokens import get_vocab_size, load_move_vocab
from krasnal.trainer import (
    DistributedInfo,
    build_model,
    build_optimizer,
    cosine_warmup_lr,
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
    write_artifact_config_json,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../../config", config_name="pretrain")
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

    if not PRETRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Pretraining dataset not found at {PRETRAIN_DATASET_PATH}. "
            "Run scripts/data/preprocess.py first to generate it."
        )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    dataloader_num_workers = int(model_cfg.pop("dataloader_num_workers", 0))
    mconf = GPTConfig(vocab_size=get_vocab_size(), **model_cfg)
    train_dataset = PretrainDataset(PRETRAIN_DATASET_PATH)
    dataset_mtime = int(PRETRAIN_DATASET_PATH.stat().st_mtime)
    model = build_model(model_config=mconf)
    vocab_size = get_vocab_size()
    tconf = TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    train_collate = make_packed_collate_fn()
    eval_collate = make_collate_fn()
    if tconf.epochs <= 0:
        raise ValueError("TrainConfig.epochs must be > 0")
    if dist_info.enabled:
        scale = float(dist_info.world_size)
        tconf.learning_rate *= scale
        tconf.min_lr *= scale
        if tconf.optimizer == "muon":
            tconf.muon_lr *= scale

    train_device = torch.device("cuda", dist_info.local_rank) if dist_info.enabled else None
    device, dtype, ctx, scaler = setup_runtime(device=train_device)

    params_M = model.get_num_params() / 1_000_000
    artifact_dir = None
    wandb_run_url = ""
    wandb_config: dict = {}

    if dist_info.is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = ARTIFACTS_DIR / "pretrain" / timestamp
        artifact_dir.mkdir(parents=True, exist_ok=True)
        wandb_config = {
            "stage": "pretrain",
            "params_M": params_M,
            "vocab_size": vocab_size,
            "block_size": mconf.block_size,
            "n_layer": mconf.n_layer,
            "n_head": mconf.n_head,
            "n_embd": mconf.n_embd,
            "dropout": mconf.dropout,
            "use_time_conditioning": mconf.use_time_conditioning,
            "time_conditioning_hidden": mconf.time_conditioning_hidden,
            "mlp_activation": mconf.mlp_activation,
            "epochs": tconf.epochs,
            "batch_size": tconf.batch_size,
            "learning_rate": tconf.learning_rate,
            "seed": cfg.seed,
            "piece_aware_moves": piece_aware_moves,
            "side_prefixed_moves": side_prefixed_moves,
            "gpt_model_name": cfg.model.get("name", "custom"),
            "move_vocab_path": str(MOVE_VOCAB_PATH),
            "dataset_mtime": dataset_mtime,
            "dataset_size": len(train_dataset),
            "train_window_rows": len(train_dataset),
            "tokens_per_step": tconf.batch_size * mconf.block_size,
            "optimizer": tconf.optimizer,
            "model_repr": repr(model),
            "world_size": dist_info.world_size,
            "ddp": dist_info.enabled,
        }
        write_artifact_config_json(artifact_dir, wandb_config)
        run_id, entity, project = init_wandb(
            project=cfg.wandb_project,
            config=wandb_config,
            stage="pretrain",
        )
        wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

        print_model_config(
            stage="Pretrain",
            params_m=params_M,
            dataset_size=len(train_dataset),
            dataset_label="packed windows",
            config=mconf,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
            compile_enabled=tconf.compile,
            artifact_dir=artifact_dir,
        )

    optimizer = build_optimizer(model, tconf, device.type)

    model.to(device)
    if tconf.compile and device.type == "cuda":
        model = torch.compile(
            model,
            mode=tconf.compile_mode,
            dynamic=tconf.compile_dynamic,
            fullgraph=tconf.compile_fullgraph,
        )
    if dist_info.enabled:
        model = DDP(model, device_ids=[dist_info.local_rank])

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=True,
            seed=cfg.seed,
        )
        if dist_info.enabled
        else None
    )
    train_num_workers = dataloader_num_workers if dataloader_num_workers > 0 else tconf.num_workers
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=tconf.pin_memory,
        batch_size=tconf.batch_size,
        num_workers=train_num_workers,
        persistent_workers=train_num_workers > 0,
        prefetch_factor=4 if train_num_workers > 0 else None,
        in_order=False,
        collate_fn=train_collate,
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Training dataset is empty. Cannot run training.")

    total_iters = max(1, math.ceil(tconf.epochs * steps_per_epoch))
    tconf.max_iters = total_iters
    tconf.steps_per_epoch = steps_per_epoch

    tokens_per_step = tconf.batch_size * mconf.block_size

    def log_fn(_iter_num, last_loss_value, epoch_float):
        wandb.log(
            {
                "train_loss": last_loss_value,
                "epoch": epoch_float,
                "tokens_seen": _iter_num * tokens_per_step,
            }
        )

    eval_dataset = ChessDataset(
        EVAL_DATASET_PATH,
        include_elo=cfg.get("include_elo", True),
    )
    val_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        pin_memory=tconf.pin_memory,
        batch_size=tconf.batch_size,
        num_workers=train_num_workers,
        persistent_workers=train_num_workers > 0,
        prefetch_factor=4 if train_num_workers > 0 else None,
        collate_fn=eval_collate,
    )

    evaluator = (
        chess_evaluator_from_config(cfg, metrics=list(cfg.eval.metrics))
        if dist_info.is_master
        else None
    )
    eval_device = torch.device(device)

    def eval_fn(model, _iter_num):
        raw_model = unwrap_model(model)
        if evaluator is None:
            return {}
        return evaluator.evaluate(raw_model, eval_dataset, tconf.eval_num_games, eval_device)

    run_supervised_training(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_config=tconf,
        device=device,
        ctx=ctx,
        scaler=scaler,
        lr_fn=lambda i: cosine_warmup_lr(i, tconf),
        desc="train",
        log_fn=log_fn,
        eval_fn=eval_fn,
        eval_log_fn=lambda _i, m: log_eval_metrics_to_wandb(m),
        val_loader=val_loader,
        dist_info=dist_info,
        train_sampler=train_sampler,
    )

    if dist_info.is_master:
        print("Training finished.")
        assert artifact_dir is not None
        model_path = artifact_dir / "model.pt"
        save_model_state(unwrap_model(model), model_path, move_vocab_path=MOVE_VOCAB_PATH)
        print(f"Model saved to {model_path}")

        save_wandb_run(
            artifact_dir=artifact_dir,
            run_config=wandb_config,
            wandb_run_url=wandb_run_url,
            artifact_name="pretrain",
            artifact_type="model",
        )

        wandb.finish()


if __name__ == "__main__":
    main()
