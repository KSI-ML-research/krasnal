import math
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import wandb
from krasnal.config import (
    ARTIFACTS_DIR,
    EVAL_DATASET_PATH,
    MOVE_VOCAB_PATH,
    PRETRAIN_DATASET_PATH,
    GPTConfig,
    TrainConfig,
)
from krasnal.dataset import ChessDataset, make_collate_fn
from krasnal.eval import ChessEvaluator, get_stockfish_client
from krasnal.tokens import get_vocab_size, load_move_vocab
from krasnal.trainer import (
    build_model,
    cosine_warmup_lr,
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

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../../config", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    piece_aware_moves = bool(cfg.get("piece_aware_moves", False))
    side_prefixed_moves = bool(cfg.get("side_prefixed_moves", True))
    load_move_vocab(
        MOVE_VOCAB_PATH,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )
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
        "piece_aware_moves": piece_aware_moves,
        "side_prefixed_moves": side_prefixed_moves,
        "gpt_model_name": cfg.model.get("name", "custom"),
        "move_vocab_path": str(MOVE_VOCAB_PATH),
        "dataset_mtime": dataset_mtime,
        "dataset_size": len(train_dataset),
        "model_repr": repr(model),
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

    stockfish = get_stockfish_client(depth=cfg.eval.stockfish.depth)
    evaluator = ChessEvaluator(
        metrics=list(cfg.eval.metrics),
        stockfish=stockfish,
        seed=cfg.seed,
        acpl_sample_size=cfg.eval.stockfish.acpl_sample_size,
        qa_config=OmegaConf.to_container(cfg.eval.qa, resolve=True),
    )
    eval_device = torch.device(device)

    def eval_fn(model, _iter_num):
        raw_model = unwrap_model(model)
        return evaluator.evaluate(raw_model, eval_dataset, tconf.eval_num_games, eval_device)

    def eval_log_fn(_iter_num, metrics):
        payload = {}
        for k, v in metrics.items():
            payload[format_eval_metric_key(k)] = v
        if "qa/what_is_on/accuracy_matrix" in metrics:
            heatmap = metrics["qa/what_is_on/accuracy_matrix"]
            payload[format_eval_metric_key("qa/what_is_on/accuracy_matrix")] = heatmap
            wandb.run.summary["eval/qa/what_is_on/accuracy_matrix"] = heatmap  # type: ignore[index]
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
        desc="train",
        log_fn=log_fn,
        eval_fn=eval_fn,
        eval_log_fn=eval_log_fn,
        val_loader=val_loader,
    )

    print("Training finished.")
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
