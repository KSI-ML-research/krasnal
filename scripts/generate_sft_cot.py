import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl
from utils import set_seed

import wandb
from config import ARTIFACTS_DIR, MOVES_FILE, SFT_COT_DATASET_PATH
from sft import load_raw_games, sample_cot_rows
from tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic CoT samples with Stockfish.")
    parser.add_argument("--out", type=Path, default=SFT_COT_DATASET_PATH)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--stockfish-path", type=Path, required=True)
    parser.add_argument("--multipv-min", type=int, default=1)
    parser.add_argument("--multipv-max", type=int, default=3)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--movetime-ms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="krasnal")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.depth is None) == (args.movetime_ms is None):
        raise ValueError("Use exactly one of --depth or --movetime-ms")
    if not args.stockfish_path.exists():
        raise FileNotFoundError(f"Stockfish not found at {args.stockfish_path}")

    set_seed(args.seed)

    tokenizer = Tokenizer(MOVES_FILE)
    games = load_raw_games()
    rows = sample_cot_rows(
        games=games,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        multipv_min=args.multipv_min,
        multipv_max=args.multipv_max,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.out)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_DIR / "sft_cot_data" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "stage": "generate_sft_cot",
        "dataset_path": str(args.out),
        "dataset_size": len(rows),
        "stockfish_path": str(args.stockfish_path),
        "multipv_min": args.multipv_min,
        "multipv_max": args.multipv_max,
        "depth": args.depth,
        "movetime_ms": args.movetime_ms,
        "seed": args.seed,
    }

    wandb.init(project=args.wandb_project, config=run_config)
    run_id = wandb.run.id  # type: ignore[union-attr]
    entity = wandb.run.entity  # type: ignore[union-attr]
    project = wandb.run.project  # type: ignore[union-attr]
    wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

    with open(artifact_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    with open(artifact_dir / "wandb_run_link.txt", "w") as f:
        f.write(f"{wandb_run_url}\n")

    dataset_copy = artifact_dir / args.out.name
    pl.DataFrame(rows).write_parquet(dataset_copy)
    artifact = wandb.Artifact("sft_cot_data", type="dataset")
    artifact.add_dir(str(artifact_dir))
    wandb.log_artifact(artifact)
    wandb.finish()

    print(f"Generated {len(rows)} synthetic CoT samples at {args.out}")


if __name__ == "__main__":
    main()
