from pathlib import Path

import torch
from omegaconf import OmegaConf

import wandb
from krasnal.config import EVAL_DATASET_PATH, MOVE_VOCAB_PATH
from krasnal.dataset import ChessDataset
from krasnal.eval.evaluator import chess_evaluator_from_config
from krasnal.tokens import load_move_vocab
from krasnal.uci_engine.provider import ModelProvider
from krasnal.utils import log_eval_metrics_to_wandb


def main():
    artifact_dir = Path("artifacts/pretrain/20260518_190601")

    print(f"Loading model from {artifact_dir}")
    provider = ModelProvider.from_artifact_dir(artifact_dir)

    cfg = OmegaConf.create(provider.artifact_config)
    eval_cfg = OmegaConf.load("config/eval.yaml")
    cfg.eval = eval_cfg
    cfg.seed = 42

    load_move_vocab(
        MOVE_VOCAB_PATH,
        piece_aware_moves=cfg.get("piece_aware_moves", False),
        side_prefixed_moves=cfg.get("side_prefixed_moves", True),
    )

    dataset = ChessDataset(EVAL_DATASET_PATH, include_elo=cfg.get("include_elo", True))

    evaluator = chess_evaluator_from_config(cfg, metrics=list(cfg.eval.metrics))

    print("Evaluating...")
    device = torch.device(provider.device)
    results = evaluator.evaluate(
        model=provider.model,
        dataset=dataset,
        num_games=20,
        device=device,
    )

    print("Initializing wandb run...")
    run = wandb.init(project="krasnal", tags=["pretrain"])

    log_eval_metrics_to_wandb(results)

    print("Done. Run URL:", run.get_url())
    run.finish()


if __name__ == "__main__":
    main()
