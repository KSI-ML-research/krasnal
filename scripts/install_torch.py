#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

TORCH_VERSION = "2.10.0"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTORCH_CU124_INDEX = "https://download.pytorch.org/whl/cu124"


def run(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Install PyTorch variant into current uv environment. "
            "Use CPU for universal compatibility or CUDA for NVIDIA GPUs."
        )
    )
    parser.add_argument(
        "--target",
        choices=["cpu", "cu124"],
        default="cpu",
        help="PyTorch wheel target: cpu or cu124.",
    )
    parser.add_argument(
        "--torch-version",
        default=TORCH_VERSION,
        help=f"PyTorch version (default: {TORCH_VERSION}).",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Force reinstall even if torch is already present.",
    )
    args = parser.parse_args()

    index_url = PYTORCH_CPU_INDEX if args.target == "cpu" else PYTORCH_CU124_INDEX

    cmd = [
        "uv",
        "pip",
        "install",
        f"torch=={args.torch_version}",
        "--index-url",
        index_url,
    ]

    if args.reinstall:
        cmd.append("--reinstall")

    run(cmd)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
