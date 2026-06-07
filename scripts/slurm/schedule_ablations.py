#!/usr/bin/env python3

from __future__ import annotations

import re
import shlex
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import count
from pathlib import Path

BASE_GAMES = 4_000_000
DOUBLE_BASE_GAMES = BASE_GAMES * 2
DATA_SCALE_GAMES = BASE_GAMES * 8
SEEDS = (42, 43, 44)
TRAIN_BATCH_SIZE = 64

FILTERED_BASE = "data/1_filtered_paper"
TOKENIZED_BASE = "data/2_tokenized_paper"
ARTIFACT_BASE = "artifacts/pretrain"
DIAGNOSTIC_OUTPUT_DIR = "artifacts/diagnostics"
OUTPUT_DIR = "output"
DRY_RUN_JOB_IDS = count(10_000)

DOWNLOAD_PARTITION = "student-cpu"
DOWNLOAD_TIME = "04:00:00"
DOWNLOAD_CPUS = 4

PREPROCESS_PARTITION = "student-cpu"
PREPROCESS_TIME = "16:00:00"
PREPROCESS_CPUS = 12

TRAIN_PARTITION = "student-nvidia"
TRAIN_TIME = "24:00:00"
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
    run_name: str | None = None


@dataclass(frozen=True)
class DataVariant:
    name: str
    preprocess_overrides: tuple[str, ...]
    train_variants: tuple[str, ...]
    train_overrides: tuple[str, ...] = ()
    report: bool = False


def games_label(games: int) -> str:
    return f"{games // 1_000_000}M"


def epoch_fraction(*, budget_games: int, corpus_games: int) -> str:
    value = budget_games / corpus_games
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def train_variant(
    *,
    name: str,
    model: str,
    seed: int,
    budget_games: int,
    corpus_games: int,
    run_name: str | None = None,
    extra_overrides: tuple[str, ...] = (),
) -> TrainVariant:
    return TrainVariant(
        name=name,
        run_name=run_name,
        overrides=(
            f"model={model}",
            f"train.epochs={epoch_fraction(budget_games=budget_games, corpus_games=corpus_games)}",
            f"train.batch_size={TRAIN_BATCH_SIZE}",
            f"seed={seed}",
            *extra_overrides,
        ),
    )


TRAIN_VARIANTS: dict[str, TrainVariant] = {}

for seed in SEEDS:
    for model in ("small", "medium", "large"):
        key = f"{model}_{games_label(BASE_GAMES)}_seed{seed}"
        TRAIN_VARIANTS[key] = train_variant(
            name=key,
            model=model,
            seed=seed,
            budget_games=BASE_GAMES,
            corpus_games=DATA_SCALE_GAMES,
            run_name=f"baseline_{games_label(BASE_GAMES)}-{model}-seed{seed}",
        )

    key = f"medium_no_clock_{games_label(BASE_GAMES)}_seed{seed}"
    TRAIN_VARIANTS[key] = train_variant(
        name=key,
        model="medium",
        seed=seed,
        budget_games=BASE_GAMES,
        corpus_games=DATA_SCALE_GAMES,
        run_name=f"no_clock_encodings_{games_label(BASE_GAMES)}-medium-seed{seed}",
        extra_overrides=("model.use_clock_encodings=false",),
    )

for budget_games in (DOUBLE_BASE_GAMES, BASE_GAMES * 4, DATA_SCALE_GAMES):
    key = f"large_{games_label(budget_games)}_seed42"
    TRAIN_VARIANTS[key] = train_variant(
        name=key,
        model="large",
        seed=42,
        budget_games=budget_games,
        corpus_games=DATA_SCALE_GAMES,
        run_name=f"baseline_{games_label(budget_games)}-large-seed42",
    )

for seed in SEEDS:
    key = f"medium_seed{seed}"
    TRAIN_VARIANTS[key] = train_variant(
        name=key,
        model="medium",
        seed=seed,
        budget_games=BASE_GAMES,
        corpus_games=BASE_GAMES,
    )

TRAIN_VARIANTS["medium_8M_seed42"] = train_variant(
    name="medium_8M_seed42",
    model="medium",
    seed=42,
    budget_games=DOUBLE_BASE_GAMES,
    corpus_games=DOUBLE_BASE_GAMES,
)

DATA_VARIANTS = (
    DataVariant(
        name=f"baseline_{games_label(DATA_SCALE_GAMES)}",
        preprocess_overrides=(f"target_games={DATA_SCALE_GAMES}",),
        train_variants=(
            *(
                f"{model}_{games_label(BASE_GAMES)}_seed{seed}"
                for model in ("small", "medium", "large")
                for seed in SEEDS
            ),
            *(f"medium_no_clock_{games_label(BASE_GAMES)}_seed{seed}" for seed in SEEDS),
            f"large_{games_label(DOUBLE_BASE_GAMES)}_seed42",
            f"large_{games_label(BASE_GAMES * 4)}_seed42",
            f"large_{games_label(DATA_SCALE_GAMES)}_seed42",
        ),
        report=True,
    ),
    DataVariant(
        name=f"no_time_control_token_{games_label(BASE_GAMES)}",
        preprocess_overrides=(f"target_games={BASE_GAMES}", "time_control_token.enabled=false"),
        train_variants=tuple(f"medium_seed{seed}" for seed in SEEDS),
        train_overrides=("time_control_token.enabled=false",),
    ),
    DataVariant(
        name=f"no_elo_token_{games_label(BASE_GAMES)}",
        preprocess_overrides=(f"target_games={BASE_GAMES}", "include_elo=false"),
        train_variants=tuple(f"medium_seed{seed}" for seed in SEEDS),
        train_overrides=("include_elo=false",),
    ),
    DataVariant(
        name=f"no_piece_prefix_{games_label(BASE_GAMES)}",
        preprocess_overrides=(f"target_games={BASE_GAMES}", "piece_aware_moves=false"),
        train_variants=tuple(f"medium_seed{seed}" for seed in SEEDS),
        train_overrides=("piece_aware_moves=false",),
    ),
    DataVariant(
        name=f"no_color_prefix_{games_label(BASE_GAMES)}",
        preprocess_overrides=(f"target_games={BASE_GAMES}", "side_prefixed_moves=false"),
        train_variants=tuple(f"medium_seed{seed}" for seed in SEEDS),
        train_overrides=("side_prefixed_moves=false",),
    ),
    DataVariant(
        name=f"no_is_check_{games_label(BASE_GAMES)}",
        preprocess_overrides=(f"target_games={BASE_GAMES}", "qa.check.enabled=false"),
        train_variants=tuple(f"medium_seed{seed}" for seed in SEEDS),
    ),
    DataVariant(
        name=f"no_what_is_on_{games_label(BASE_GAMES)}",
        preprocess_overrides=(f"target_games={BASE_GAMES}", "qa.what_is_on.enabled=false"),
        train_variants=tuple(f"medium_seed{seed}" for seed in SEEDS),
    ),
    DataVariant(
        name=f"no_opp_material_{games_label(BASE_GAMES)}",
        preprocess_overrides=(f"target_games={BASE_GAMES}", "opponent_material.enabled=false"),
        train_variants=tuple(f"medium_seed{seed}" for seed in SEEDS),
        train_overrides=("opponent_material.enabled=false",),
    ),
    DataVariant(
        name=f"no_opp_material_{games_label(DOUBLE_BASE_GAMES)}",
        preprocess_overrides=(
            f"target_games={DOUBLE_BASE_GAMES}",
            "opponent_material.enabled=false",
        ),
        train_variants=("medium_8M_seed42",),
        train_overrides=("opponent_material.enabled=false",),
    ),
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def filtered_dir_for(data_name: str) -> str:
    return f"{FILTERED_BASE}/{slug(data_name)}"


def tokenized_dir_for(data_name: str) -> str:
    return f"{TOKENIZED_BASE}/{slug(data_name)}"


def export_arg(env: dict[str, str]) -> str:
    return "ALL," + ",".join(f"{key}={value}" for key, value in env.items())


def run_command(command: list[str], *, submit: bool) -> str:
    print(shlex.join(command))
    if not submit:
        return str(next(DRY_RUN_JOB_IDS))
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
    return tuple(
        override
        for override in data.preprocess_overrides
        if override.startswith(vocab_relevant_prefixes)
    )


def submit_download(
    data: DataVariant,
    filtered_dir: str,
    tokenized_dir: str,
    *,
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
    command.extend(
        [
            "--export",
            export_arg(
                {
                    "KRASNAL_FILTERED_DIR": filtered_dir,
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
    filtered_dir: str,
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
                "KRASNAL_FILTERED_DIR": filtered_dir,
                "KRASNAL_TOKENIZED_DIR": tokenized_dir,
                "PREPROCESS_EXTRA_ARGS": " ".join(
                    (
                        *data.preprocess_overrides,
                        f"report.enabled={str(data.report).lower()}",
                    )
                ),
            }
        ),
        "scripts/slurm/preprocess.sh",
    ]
    return run_command(command, submit=submit)


def run_name_for(data: DataVariant, train: TrainVariant) -> str:
    return train.run_name or f"{data.name}-{train.name}"


def submit_train(
    data: DataVariant,
    train: TrainVariant,
    preprocess_job_id: str,
    *,
    submit: bool,
) -> str:
    run_name = run_name_for(data, train)
    tokenized_dir = tokenized_dir_for(data.name)
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
                "RUN_GROUP": f"krasnal-paper-{data.name}",
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
                "PROBE_EVAL_PARQUET": f"{tokenized_dir_for(eval_data_variant)}/eval.parquet",
                "PROBE_JSON_OUT": f"{DIAGNOSTIC_OUTPUT_DIR}/{slug(name)}.json",
                "PROBE_WANDB": str(DIAGNOSTIC_WANDB).lower(),
                "PROBE_WANDB_NAME": name,
                "PROBE_WANDB_GROUP": "krasnal-paper-diagnostics",
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
    print(f"Base games: {games_label(BASE_GAMES)}")
    print(f"Data scale corpus: {games_label(DATA_SCALE_GAMES)}")
    print(f"Seeds: {', '.join(str(seed) for seed in SEEDS)}")
    print(f"Train batch size: {TRAIN_BATCH_SIZE}")

    PROJECT_ROOT.joinpath(OUTPUT_DIR).mkdir(exist_ok=True)

    train_job_ids: dict[str, str] = {}
    for data in DATA_VARIANTS:
        print(f"\nData variant: {data.name}")
        filtered_dir = filtered_dir_for(data.name)
        tokenized_dir = tokenized_dir_for(data.name)
        download_job_id = submit_download(
            data,
            filtered_dir,
            tokenized_dir,
            submit=args.submit,
        )
        preprocess_job_id = submit_preprocess(
            data,
            filtered_dir,
            tokenized_dir,
            download_job_id,
            submit=args.submit,
        )
        for train_name in data.train_variants:
            train = TRAIN_VARIANTS[train_name]
            run_name = run_name_for(data, train)
            train_job_ids[run_name] = submit_train(
                data,
                train,
                preprocess_job_id,
                submit=args.submit,
            )

    baseline_run = f"baseline_{games_label(BASE_GAMES)}-medium-seed42"
    no_is_check_run = f"no_is_check_{games_label(BASE_GAMES)}-medium_seed42"
    no_what_is_on_run = f"no_what_is_on_{games_label(BASE_GAMES)}-medium_seed42"

    print("\nDiagnostic probes")
    submit_diagnostic(
        name=f"check_state_no_is_check_vs_baseline_{games_label(BASE_GAMES)}",
        script="scripts/diagnostics/check_state_probe.py",
        artifact_run_names=(no_is_check_run, baseline_run),
        eval_data_variant=f"no_is_check_{games_label(BASE_GAMES)}",
        dependency_job_ids=(train_job_ids[no_is_check_run], train_job_ids[baseline_run]),
        submit=args.submit,
    )
    submit_diagnostic(
        name=f"board_state_no_what_is_on_vs_baseline_{games_label(BASE_GAMES)}",
        script="scripts/diagnostics/board_state_probe.py",
        artifact_run_names=(no_what_is_on_run, baseline_run),
        eval_data_variant=f"no_what_is_on_{games_label(BASE_GAMES)}",
        dependency_job_ids=(train_job_ids[no_what_is_on_run], train_job_ids[baseline_run]),
        submit=args.submit,
    )

    if not args.submit:
        print("\nDry run only. Pass --submit to call sbatch.")


if __name__ == "__main__":
    main()
