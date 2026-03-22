#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def ensure_docker_available() -> None:
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker CLI not found. Install Docker Desktop (Windows/macOS) or Docker Engine (Linux)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Docker daemon is not reachable. Start Docker Desktop and wait until status is 'Running'. "
            "On Windows verify Linux containers mode is enabled."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run training container.")
    parser.add_argument("--image", default="krasnal-train:latest")
    parser.add_argument("--target", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--run-preprocess", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()

    ensure_docker_available()

    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve()
    models_dir = (root / args.models_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_name = "Dockerfile.train.gpu" if args.target == "gpu" else "Dockerfile.train"

    if not args.no_build:
        run(
            [
                "docker",
                "build",
                "-f",
                str(root / "docker" / dockerfile_name),
                "-t",
                args.image,
                str(root),
            ]
        )

    command_parts = []
    if args.run_preprocess:
        command_parts.append("python scripts/preprocess.py")
    command_parts.append("python scripts/pretrain.py")
    if args.run_eval:
        command_parts.append("python scripts/evaluate.py")

    container_cmd = " && ".join(command_parts)
    run_cmd = ["docker", "run", "--rm"]
    if args.target == "gpu":
        run_cmd.extend(["--gpus", "all"])

    run_cmd.extend(
        [
            "-e",
            f"SEED={args.seed}",
            "-v",
            f"{data_dir}:/app/data",
            "-v",
            f"{models_dir}:/app/models",
            args.image,
            "sh",
            "-lc",
            container_cmd,
        ]
    )
    run(run_cmd)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
