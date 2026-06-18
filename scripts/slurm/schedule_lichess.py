#!/usr/bin/env python3

from __future__ import annotations

import os
import shlex
import subprocess
from argparse import ArgumentParser
from itertools import count
from pathlib import Path

GAMES = 16_000_000
MIN_ELO = 1800
MODEL = "large"
SEED = 42
BATCH_SIZE = 64

NAME = "lichess_16M_elo1800"
FILTERED_DIR = f"data/1_filtered_{NAME}"
TOKENIZED_DIR = f"data/2_tokenized_{NAME}"
ARTIFACT_DIR = f"artifacts/pretrain/{NAME}_{MODEL}_seed{SEED}"
OUTPUT_DIR = "output"

DOWNLOAD_PARTITION = "student-cpu"
DOWNLOAD_TIME = "48:00:00"
DOWNLOAD_CPUS = 6

PREPROCESS_PARTITION = "student-cpu"
PREPROCESS_TIME = "48:00:00"
PREPROCESS_CPUS = 12

TRAIN_PARTITION = "student-nvidia"
TRAIN_TIME = "48:00:00"
TRAIN_CPUS = 24
TRAIN_GPUS = 2
TRAIN_WORKERS = 4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRY_RUN_JOB_IDS = count(10_000)


def export_arg(env: dict[str, str]) -> str:
    return "ALL," + ",".join(f"{key}={value}" for key, value in env.items())


def run(command: list[str], *, submit: bool) -> str:
    print(shlex.join(command))
    if not submit:
        return str(next(DRY_RUN_JOB_IDS))
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = result.stdout.strip().splitlines()[-1]
    print(f"  -> {job_id}")
    return job_id


def submit_download(*, submit: bool) -> str:
    return run(
        [
            "sbatch",
            "--parsable",
            "--job-name",
            f"krasnal-dl-{NAME}",
            "--output",
            f"{OUTPUT_DIR}/%j_download_{NAME}.out",
            "--error",
            f"{OUTPUT_DIR}/%j_download_{NAME}.err",
            "--cpus-per-task",
            str(DOWNLOAD_CPUS),
            "--partition",
            DOWNLOAD_PARTITION,
            "--time",
            DOWNLOAD_TIME,
            "--export",
            export_arg(
                {
                    "KRASNAL_FILTERED_DIR": FILTERED_DIR,
                    "KRASNAL_TOKENIZED_DIR": TOKENIZED_DIR,
                    "DOWNLOAD_EXTRA_ARGS": f"target_games={GAMES} min_elo={MIN_ELO}",
                }
            ),
            "scripts/slurm/download_games.sh",
        ],
        submit=submit,
    )


def submit_preprocess(download_job_id: str, *, submit: bool) -> str:
    return run(
        [
            "sbatch",
            "--parsable",
            "--job-name",
            f"krasnal-prep-{NAME}",
            "--output",
            f"{OUTPUT_DIR}/%j_preprocess_{NAME}.out",
            "--error",
            f"{OUTPUT_DIR}/%j_preprocess_{NAME}.err",
            "--cpus-per-task",
            str(PREPROCESS_CPUS),
            "--partition",
            PREPROCESS_PARTITION,
            "--time",
            PREPROCESS_TIME,
            "--dependency",
            f"afterok:{download_job_id}",
            "--export",
            export_arg(
                {
                    "KRASNAL_FILTERED_DIR": FILTERED_DIR,
                    "KRASNAL_TOKENIZED_DIR": TOKENIZED_DIR,
                    "PREPROCESS_EXTRA_ARGS": (
                        f"--config-name preprocess_lichess target_games={GAMES}"
                    ),
                }
            ),
            "scripts/slurm/preprocess.sh",
        ],
        submit=submit,
    )


def submit_train(preprocess_job_id: str, *, submit: bool) -> str:
    run_name = f"{NAME}-{MODEL}-seed{SEED}"
    return run(
        [
            "sbatch",
            "--parsable",
            "--job-name",
            f"krasnal-{run_name}",
            "--output",
            f"{OUTPUT_DIR}/%j_train_{run_name}.out",
            "--error",
            f"{OUTPUT_DIR}/%j_train_{run_name}.err",
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
                    "KRASNAL_TOKENIZED_DIR": TOKENIZED_DIR,
                    "KRASNAL_ARTIFACT_DIR": ARTIFACT_DIR,
                    "RUN_GROUP": NAME,
                    "RUN_NAME": run_name,
                    "RUN_CONFIG_NAME": "pretrain_lichess",
                    "RUN_NPROC": str(TRAIN_GPUS),
                    "RUN_NUM_WORKERS": str(TRAIN_WORKERS),
                    "RUN_OVERRIDES": (
                        f"model={MODEL} train.epochs=1.0 train.batch_size={BATCH_SIZE} seed={SEED}"
                    ),
                }
            ),
            "scripts/slurm/train_ablation.sh",
        ],
        submit=submit,
    )


def main() -> None:
    parser = ArgumentParser(description="Schedule one Lichess pretraining run.")
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch.")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    PROJECT_ROOT.joinpath(OUTPUT_DIR).mkdir(exist_ok=True)

    mode = "SUBMIT" if args.submit else "DRY RUN"
    print(f"{mode}: 1 download job, 1 preprocess job, 1 train job")
    print(f"Games: {GAMES:,}")
    print(f"Minimum Elo: {MIN_ELO}")
    print(f"Model: {MODEL}, seed: {SEED}, batch size: {BATCH_SIZE}")
    print(f"Tokenized dir: {TOKENIZED_DIR}")

    download_job_id = submit_download(submit=args.submit)
    preprocess_job_id = submit_preprocess(download_job_id, submit=args.submit)
    submit_train(preprocess_job_id, submit=args.submit)

    if not args.submit:
        print("\nDry run only. Pass --submit to call sbatch.")


if __name__ == "__main__":
    main()
