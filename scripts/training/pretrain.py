import math
from datetime import datetime
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from krasnal.config import (
    ARTIFACTS_DIR,
    EVAL_DATASET_PATH,
    PRETRAIN_DATASET_PATH,
    GPTConfig,
    TrainConfig,
)
from krasnal.dataset import ChessDataset, make_collate_fn
from krasnal.eval import ChessEvaluator, get_stockfish_client
from krasnal.eval.puzzles import evaluate_model_on_puzzle_file
from krasnal.tokens import get_vocab_size, set_side_prefixed_moves
from krasnal.trainer import (
    build_model,
    cosine_warmup_lr,
    run_supervised_training,
    save_model_state,
    setup_runtime,
    unwrap_model,
)
from krasnal.utils import init_wandb, print_model_config, save_wandb_run, set_seed

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../../config", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    set_side_prefixed_moves(bool(cfg.get("side_prefixed_moves", True)))
    set_seed(cfg.seed)

    if not PRETRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Pretraining dataset not found at {PRETRAIN_DATASET_PATH}. "
            "Run scripts/preprocess.py first to generate it."
        )

    train_dataset = ChessDataset(PRETRAIN_DATASET_PATH)
    dataset_mtime = int(PRETRAIN_DATASET_PATH.stat().st_mtime)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    mconf = GPTConfig(vocab_size=get_vocab_size(), **model_cfg)
    model = build_model(model_config=mconf)
    vocab_size = get_vocab_size()
    tconf = TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    collate = make_collate_fn(tconf.padding_bucket_sizes)
    if tconf.epochs <= 0:
        raise ValueError("TrainConfig.epochs must be > 0")

    device, dtype, ctx, scaler = setup_runtime()

    params_M = model.get_num_params() / 1_000_000
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
        "epochs": tconf.epochs,
        "batch_size": tconf.batch_size,
        "learning_rate": tconf.learning_rate,
        "seed": cfg.seed,
        "gpt_model_name": cfg.model.get("name", "custom"),
        "dataset_mtime": dataset_mtime,
        "dataset_size": len(train_dataset),
        "model_repr": repr(model),
        "puzzle_eval_enabled": bool(cfg.eval.puzzle_eval.enabled),
        "puzzle_eval_path": str(cfg.eval.puzzle_eval.path),
        "puzzle_eval_sample_size": cfg.eval.puzzle_eval.sample_size,
    }

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
        dataset_label="games",
        config=mconf,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
        compile_enabled=tconf.compile,
        artifact_dir=artifact_dir,
    )

    optimizer = model.configure_optimizers(
        weight_decay=tconf.weight_decay,
        learning_rate=tconf.learning_rate,
        betas=(tconf.beta1, tconf.beta2),
        device_type=device.type,
    )

    model.to(device)

    if tconf.compile and device.type == "cuda":
        model = torch.compile(
            model,
            mode=tconf.compile_mode,
            dynamic=tconf.compile_dynamic,
            fullgraph=tconf.compile_fullgraph,
        )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=tconf.pin_memory,
        batch_size=tconf.batch_size,
        num_workers=tconf.num_workers,
        collate_fn=collate,
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Training dataset is empty. Cannot run training.")

    total_iters = max(1, math.ceil(tconf.epochs * steps_per_epoch))
    tconf.max_iters = total_iters
    tconf.steps_per_epoch = steps_per_epoch

    def log_fn(_iter_num, last_loss_value, epoch_float):
        wandb.log({"train_loss": last_loss_value, "epoch": epoch_float})

    eval_dataset = ChessDataset(EVAL_DATASET_PATH)
    val_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        pin_memory=tconf.pin_memory,
        batch_size=tconf.batch_size,
        num_workers=tconf.num_workers,
        collate_fn=collate,
    )

    stockfish = get_stockfish_client(depth=cfg.eval.stockfish_depth)
    evaluator = ChessEvaluator(
        metrics=list(cfg.eval.metrics),
        stockfish=stockfish,
        seed=cfg.seed,
        acpl_sample_size=cfg.eval.acpl_sample_size,
        enable_check_probe_metrics=bool(cfg.eval.enable_check_probe_metrics),
        enable_piece_probe_metrics=bool(cfg.eval.enable_piece_probe_metrics),
        enable_check_confusion_matrix_metrics=bool(cfg.eval.enable_check_confusion_matrix_metrics),
        enable_piece_confusion_matrix_metrics=bool(cfg.eval.enable_piece_confusion_matrix_metrics),
    )
    eval_device = torch.device(device)

    puzzle_eval_enabled = bool(cfg.eval.puzzle_eval.enabled)
    puzzle_eval_path = Path(cfg.eval.puzzle_eval.path)
    puzzle_eval_sample_size = cfg.eval.puzzle_eval.sample_size
    puzzle_eval_seed = int(cfg.eval.puzzle_eval.seed)

    if puzzle_eval_enabled and not puzzle_eval_path.exists():
        print(
            f"[warn] Puzzle eval disabled: file not found at {puzzle_eval_path}. "
            "Run puzzle preparation first or update cfg.eval.puzzle_eval.path."
        )
        puzzle_eval_enabled = False

    def eval_fn(model, _iter_num):
        raw_model = unwrap_model(model)
        metrics = evaluator.evaluate(raw_model, eval_dataset, tconf.eval_num_games, eval_device)
        if puzzle_eval_enabled:
            puzzle_metrics = evaluate_model_on_puzzle_file(
                model=raw_model,
                device=eval_device,
                puzzle_path=puzzle_eval_path,
                sample_size=puzzle_eval_sample_size,
                seed=puzzle_eval_seed,
            )
            metrics.update({f"puzzle/{k}": v for k, v in puzzle_metrics.items()})
        return metrics

    def eval_log_fn(_iter_num, metrics):
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})

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
        eval_log_fn=eval_log_fn,
        val_loader=val_loader,
    )

    print("Training finished.")
    model_path = artifact_dir / "model.pt"
    save_model_state(unwrap_model(model), model_path)
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
