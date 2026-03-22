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
    parser = argparse.ArgumentParser(description="Build and run inference container.")
    parser.add_argument("--image", default="krasnal-inference:latest")
    parser.add_argument("--target", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--provider", choices=["pytorch", "mock"], default="pytorch")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    ensure_docker_available()

    root = Path(__file__).resolve().parents[1]
    models_dir = (root / args.models_dir).resolve()
    data_dir = (root / args.data_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_name = "Dockerfile.inference.gpu" if args.target == "gpu" else "Dockerfile.inference"

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

    run_cmd = ["docker", "run", "--rm"]
    if args.target == "gpu":
        run_cmd.extend(["--gpus", "all"])

    run_cmd.extend(
        [
            "-p",
            f"{args.port}:8000",
            "-e",
            f"ENGINE_PROVIDER={args.provider}",
            "-e",
            f"ENGINE_TEMPERATURE={args.temperature}",
            "-e",
            f"ENGINE_TOP_P={args.top_p}",
            "-v",
            f"{models_dir}:/app/models",
            "-v",
            f"{data_dir}:/app/data:ro",
            args.image,
        ]
    )
    run(run_cmd)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
