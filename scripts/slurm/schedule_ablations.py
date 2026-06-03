#!/usr/bin/env python3

from __future__ import annotations

import re
import shlex
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

TARGET_GAMES = 5_000_000
HALF_TARGET_GAMES = TARGET_GAMES // 2
MODEL = "medium"

TOKENIZED_BASE = "data/2_tokenized_ablations"
OUTPUT_DIR = "output"

PREPROCESS_PARTITION = "student-cpu"
PREPROCESS_TIME = "08:00:00"
PREPROCESS_CPUS = 12

TRAIN_PARTITION = "student-nvidia"
TRAIN_TIME = "08:00:00"
TRAIN_CPUS = 24
TRAIN_GPUS = 1
TRAIN_WORKERS = 4


@dataclass(frozen=True)
class TrainVariant:
    name: str
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class DataVariant:
    name: str
    preprocess_overrides: tuple[str, ...]
    train_variants: tuple[str, ...]
    train_overrides: tuple[str, ...] = ()


BASE_TRAIN = ("model=medium", "train.epochs=1.0", "train.batch_size=128", "seed=42")

TRAIN_VARIANTS = {
    "baseline": TrainVariant("baseline", BASE_TRAIN),
    "no_time_model": TrainVariant(
        "no_time_model",
        (*BASE_TRAIN, "model.use_time_conditioning=false"),
    ),
    "gelu": TrainVariant("gelu", (*BASE_TRAIN, "model.mlp_activation=gelu")),
    "relu2": TrainVariant("relu2", (*BASE_TRAIN, "model.mlp_activation=relu2")),
    "muon": TrainVariant("muon", (*BASE_TRAIN, "train.optimizer=muon")),
}

DATA_VARIANTS = (
    DataVariant(
        name=f"baseline_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}",),
        train_variants=("baseline", "no_time_model", "gelu", "relu2", "muon"),
    ),
    DataVariant(
        name=f"no_tc_token_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "time_control.enabled=false"),
        train_variants=("baseline",),
        train_overrides=("time_control.enabled=false",),
    ),
    DataVariant(
        name=f"no_outcome_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "outcome_conditioning.enabled=false"),
        train_variants=("baseline",),
        train_overrides=("outcome_conditioning.enabled=false",),
    ),
    DataVariant(
        name=f"no_elo_token_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "include_elo=false"),
        train_variants=("baseline",),
        train_overrides=("include_elo=false",),
    ),
    DataVariant(
        name=f"no_piece_prefix_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "piece_aware_moves=false"),
        train_variants=("baseline",),
        train_overrides=("piece_aware_moves=false",),
    ),
    DataVariant(
        name=f"no_color_prefix_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "side_prefixed_moves=false"),
        train_variants=("baseline",),
        train_overrides=("side_prefixed_moves=false",),
    ),
    DataVariant(
        name=f"opp_material_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "opponent_material.enabled=true"),
        train_variants=("baseline",),
        train_overrides=("opponent_material.enabled=true",),
    ),
    DataVariant(
        name=f"opp_material_{HALF_TARGET_GAMES}",
        preprocess_overrides=(
            f"target_games={HALF_TARGET_GAMES}",
            "opponent_material.enabled=true",
        ),
        train_variants=("baseline",),
        train_overrides=("opponent_material.enabled=true",),
    ),
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def export_arg(env: dict[str, str]) -> str:
    return "ALL," + ",".join(f"{key}={value}" for key, value in env.items())


def run_command(command: list[str], *, submit: bool) -> str:
    print(shlex.join(command))
    if not submit:
        return "DRY_RUN_JOB"
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = result.stdout.strip().splitlines()[-1]
    print(f"  -> {job_id}")
    return job_id


def submit_preprocess(data: DataVariant, tokenized_dir: str, *, submit: bool) -> str:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"krasnal-prep-{slug(data.name)}",
        "--output",
        f"{OUTPUT_DIR}/%j_preprocess_{slug(data.name)}.out",
        "--error",
        f"{OUTPUT_DIR}/%j_preprocess_{slug(data.name)}.err",
        "--cpus-per-task",
        str(PREPROCESS_CPUS),
        "--partition",
        PREPROCESS_PARTITION,
        "--time",
        PREPROCESS_TIME,
        "--export",
        export_arg(
            {
                "KRASNAL_TOKENIZED_DIR": tokenized_dir,
                "PREPROCESS_EXTRA_ARGS": " ".join(data.preprocess_overrides),
            }
        ),
        "scripts/slurm/preprocess.sh",
    ]
    return run_command(command, submit=submit)


def submit_train(
    data: DataVariant,
    train: TrainVariant,
    preprocess_job_id: str,
    *,
    submit: bool,
) -> None:
    run_name = f"{data.name}-{train.name}"
    tokenized_dir = f"{TOKENIZED_BASE}/{slug(data.name)}"
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"krasnal-{slug(run_name)}",
        "--output",
        f"{OUTPUT_DIR}/%j_train_{slug(run_name)}.out",
        "--error",
        f"{OUTPUT_DIR}/%j_train_{slug(run_name)}.err",
        "--gres",
        f"gpu:{TRAIN_GPUS}",
        "--cpus-per-task",
        str(TRAIN_CPUS),
        "--partition",
        TRAIN_PARTITION,
        "--time",
        TRAIN_TIME,
        "--dependency",
        f"afterok:{preprocess_job_id}",
        "--export",
        export_arg(
            {
                "KRASNAL_TOKENIZED_DIR": tokenized_dir,
                "RUN_GROUP": f"krasnal-{MODEL}-{data.name}",
                "RUN_NAME": run_name,
                "RUN_OVERRIDES": " ".join((*train.overrides, *data.train_overrides)),
                "RUN_NPROC": str(TRAIN_GPUS),
                "RUN_NUM_WORKERS": str(TRAIN_WORKERS),
            }
        ),
        "scripts/slurm/train_ablation.sh",
    ]
    run_command(command, submit=submit)


def main() -> None:
    parser = ArgumentParser(description="Schedule Krasnal ablation Slurm jobs.")
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch.")
    args = parser.parse_args()

    mode = "SUBMIT" if args.submit else "DRY RUN"
    train_count = sum(len(data.train_variants) for data in DATA_VARIANTS)

    print(f"{mode}: {len(DATA_VARIANTS)} preprocess jobs, {train_count} train jobs")
    print(f"Model: {MODEL}")
    print(f"Target games: {TARGET_GAMES}")

    PROJECT_ROOT.joinpath(OUTPUT_DIR).mkdir(exist_ok=True)

    for data in DATA_VARIANTS:
        print(f"\nData variant: {data.name}")
        tokenized_dir = f"{TOKENIZED_BASE}/{slug(data.name)}"
        preprocess_job_id = submit_preprocess(data, tokenized_dir, submit=args.submit)
        for train_name in data.train_variants:
            submit_train(data, TRAIN_VARIANTS[train_name], preprocess_job_id, submit=args.submit)

    if not args.submit:
        print("\nDry run only. Pass --submit to call sbatch.")


if __name__ == "__main__":
    main()
