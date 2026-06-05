#!/usr/bin/env python3

from __future__ import annotations

import re
import shlex
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

TARGET_GAMES = 10_000_000
DOUBLE_TARGET_GAMES = TARGET_GAMES * 2
MODEL = "medium"
TRAIN_BATCH_SIZE = 64
PREPROCESS_REPORT_OVERRIDE = "report.enabled=false"

TOKENIZED_BASE = "data/2_tokenized_ablations"
ARTIFACT_BASE = "artifacts/pretrain"
DIAGNOSTIC_OUTPUT_DIR = "artifacts/diagnostics"
OUTPUT_DIR = "output"

DOWNLOAD_PARTITION = "student-cpu"
DOWNLOAD_TIME = "04:00:00"
DOWNLOAD_CPUS = 4

PREPROCESS_PARTITION = "student-cpu"
PREPROCESS_TIME = "08:00:00"
PREPROCESS_CPUS = 12

TRAIN_PARTITION = "student-nvidia"
TRAIN_TIME = "08:00:00"
TRAIN_CPUS = 24
TRAIN_GPUS = 1
TRAIN_WORKERS = 4

DIAGNOSTIC_PARTITION = "student-nvidia"
DIAGNOSTIC_TIME = "04:00:00"
DIAGNOSTIC_CPUS = 12
DIAGNOSTIC_GPUS = 1
DIAGNOSTIC_WANDB = True


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


BASE_TRAIN = (
    f"model={MODEL}",
    "train.epochs=1.0",
    f"train.batch_size={TRAIN_BATCH_SIZE}",
    "seed=42",
)

TRAIN_VARIANTS = {
    "baseline": TrainVariant("baseline", BASE_TRAIN),
    "no_clock_conditioning": TrainVariant(
        "no_clock_conditioning",
        (*BASE_TRAIN, "model.use_time_conditioning=false"),
    ),
    "gelu": TrainVariant("gelu", (*BASE_TRAIN, "model.mlp_activation=gelu")),
    "relu2": TrainVariant("relu2", (*BASE_TRAIN, "model.mlp_activation=relu2")),
    # "muon": TrainVariant("muon", (*BASE_TRAIN, "train.optimizer=muon")),
}

DATA_VARIANTS = (
    DataVariant(
        name=f"baseline_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}",),
        train_variants=("baseline", "no_clock_conditioning", "gelu", "relu2"),
    ),
    DataVariant(
        name=f"no_time_control_token_{TARGET_GAMES}",
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
        name=f"no_is_check_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "qa.check.enabled=false"),
        train_variants=("baseline",),
    ),
    DataVariant(
        name=f"no_what_is_on_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "qa.what_is_on.enabled=false"),
        train_variants=("baseline",),
    ),
    DataVariant(
        name=f"no_opp_material_{TARGET_GAMES}",
        preprocess_overrides=(f"target_games={TARGET_GAMES}", "opponent_material.enabled=false"),
        train_variants=("baseline",),
        train_overrides=("opponent_material.enabled=false",),
    ),
    DataVariant(
        name=f"no_opp_material_{DOUBLE_TARGET_GAMES}",
        preprocess_overrides=(
            f"target_games={DOUBLE_TARGET_GAMES}",
            "opponent_material.enabled=false",
        ),
        train_variants=("baseline",),
        train_overrides=("opponent_material.enabled=false",),
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


def download_overrides(data: DataVariant) -> tuple[str, ...]:
    vocab_relevant_prefixes = (
        "target_games=",
        "piece_aware_moves=",
        "side_prefixed_moves=",
    )
    return (
        *(
            override
            for override in data.preprocess_overrides
            if override.startswith(vocab_relevant_prefixes)
        ),
        "require_evals=false",
    )


def submit_download(
    data: DataVariant,
    tokenized_dir: str,
    *,
    dependency_job_id: str | None,
    submit: bool,
) -> str:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"krasnal-dl-{slug(data.name)}",
        "--output",
        f"{OUTPUT_DIR}/%j_download_{slug(data.name)}.out",
        "--error",
        f"{OUTPUT_DIR}/%j_download_{slug(data.name)}.err",
        "--cpus-per-task",
        str(DOWNLOAD_CPUS),
        "--partition",
        DOWNLOAD_PARTITION,
        "--time",
        DOWNLOAD_TIME,
    ]
    if dependency_job_id is not None:
        command.extend(["--dependency", f"afterok:{dependency_job_id}"])
    command.extend(
        [
            "--export",
            export_arg(
                {
                    "KRASNAL_TOKENIZED_DIR": tokenized_dir,
                    "DOWNLOAD_EXTRA_ARGS": " ".join(download_overrides(data)),
                }
            ),
            "scripts/slurm/download_games.sh",
        ]
    )
    return run_command(command, submit=submit)


def submit_preprocess(
    data: DataVariant,
    tokenized_dir: str,
    download_job_id: str,
    *,
    submit: bool,
) -> str:
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
        "--dependency",
        f"afterok:{download_job_id}",
        "--export",
        export_arg(
            {
                "KRASNAL_TOKENIZED_DIR": tokenized_dir,
                "PREPROCESS_EXTRA_ARGS": " ".join(
                    (*data.preprocess_overrides, PREPROCESS_REPORT_OVERRIDE)
                ),
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
) -> str:
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
                "KRASNAL_ARTIFACT_DIR": f"{ARTIFACT_BASE}/{slug(run_name)}",
            }
        ),
        "scripts/slurm/train_ablation.sh",
    ]
    return run_command(command, submit=submit)


def submit_diagnostic(
    *,
    name: str,
    script: str,
    artifact_run_names: tuple[str, ...],
    eval_data_variant: str,
    dependency_job_ids: tuple[str, ...],
    submit: bool,
) -> None:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"krasnal-probe-{slug(name)}",
        "--output",
        f"{OUTPUT_DIR}/%j_probe_{slug(name)}.out",
        "--error",
        f"{OUTPUT_DIR}/%j_probe_{slug(name)}.err",
        "--gres",
        f"gpu:{DIAGNOSTIC_GPUS}",
        "--cpus-per-task",
        str(DIAGNOSTIC_CPUS),
        "--partition",
        DIAGNOSTIC_PARTITION,
        "--time",
        DIAGNOSTIC_TIME,
        "--dependency",
        "afterok:" + ":".join(dependency_job_ids),
        "--export",
        export_arg(
            {
                "PROBE_SCRIPT": script,
                "PROBE_ARTIFACT_DIRS": " ".join(
                    f"{ARTIFACT_BASE}/{slug(run_name)}" for run_name in artifact_run_names
                ),
                "PROBE_EVAL_PARQUET": f"{TOKENIZED_BASE}/{slug(eval_data_variant)}/eval.parquet",
                "PROBE_JSON_OUT": f"{DIAGNOSTIC_OUTPUT_DIR}/{slug(name)}.json",
                "PROBE_WANDB": str(DIAGNOSTIC_WANDB).lower(),
                "PROBE_WANDB_NAME": name,
                "PROBE_WANDB_GROUP": f"krasnal-{MODEL}-diagnostics",
            }
        ),
        "scripts/slurm/diagnostic_probe.sh",
    ]
    run_command(command, submit=submit)


def main() -> None:
    parser = ArgumentParser(description="Schedule Krasnal ablation Slurm jobs.")
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch.")
    args = parser.parse_args()

    mode = "SUBMIT" if args.submit else "DRY RUN"
    download_count = len(DATA_VARIANTS)
    train_count = sum(len(data.train_variants) for data in DATA_VARIANTS)
    diagnostic_count = 2

    print(
        f"{mode}: {download_count} download/vocab jobs, "
        f"{len(DATA_VARIANTS)} preprocess jobs, {train_count} train jobs, "
        f"{diagnostic_count} diagnostic jobs"
    )
    print(f"Model: {MODEL}")
    print(f"Target games: {TARGET_GAMES}")
    print(f"Train batch size: {TRAIN_BATCH_SIZE}")

    PROJECT_ROOT.joinpath(OUTPUT_DIR).mkdir(exist_ok=True)

    previous_download_job_id: str | None = None
    train_job_ids: dict[str, str] = {}
    for data in DATA_VARIANTS:
        print(f"\nData variant: {data.name}")
        tokenized_dir = f"{TOKENIZED_BASE}/{slug(data.name)}"
        download_job_id = submit_download(
            data,
            tokenized_dir,
            dependency_job_id=previous_download_job_id,
            submit=args.submit,
        )
        previous_download_job_id = download_job_id
        preprocess_job_id = submit_preprocess(
            data,
            tokenized_dir,
            download_job_id,
            submit=args.submit,
        )
        for train_name in data.train_variants:
            run_name = f"{data.name}-{train_name}"
            train_job_ids[run_name] = submit_train(
                data,
                TRAIN_VARIANTS[train_name],
                preprocess_job_id,
                submit=args.submit,
            )

    baseline_run = f"baseline_{TARGET_GAMES}-baseline"
    no_is_check_run = f"no_is_check_{TARGET_GAMES}-baseline"
    no_what_is_on_run = f"no_what_is_on_{TARGET_GAMES}-baseline"

    print("\nDiagnostic probes")
    submit_diagnostic(
        name=f"check_state_no_is_check_vs_baseline_{TARGET_GAMES}",
        script="scripts/diagnostics/check_state_probe.py",
        artifact_run_names=(no_is_check_run, baseline_run),
        eval_data_variant=f"no_is_check_{TARGET_GAMES}",
        dependency_job_ids=(train_job_ids[no_is_check_run], train_job_ids[baseline_run]),
        submit=args.submit,
    )
    submit_diagnostic(
        name=f"board_state_no_what_is_on_vs_baseline_{TARGET_GAMES}",
        script="scripts/diagnostics/board_state_probe.py",
        artifact_run_names=(no_what_is_on_run, baseline_run),
        eval_data_variant=f"no_what_is_on_{TARGET_GAMES}",
        dependency_job_ids=(train_job_ids[no_what_is_on_run], train_job_ids[baseline_run]),
        submit=args.submit,
    )

    if not args.submit:
        print("\nDry run only. Pass --submit to call sbatch.")


if __name__ == "__main__":
    main()
