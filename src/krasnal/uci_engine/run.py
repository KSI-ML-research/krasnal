#!/usr/bin/env -S uv run python

import os
import sys
from pathlib import Path

from loguru import logger

from krasnal.uci_engine.provider import ModelProvider, RandomMockProvider
from krasnal.uci_engine.uci_parser import UCIParser


def build_provider():
    requested_provider = os.environ.get("KRASNAL_ENGINE_PROVIDER", "auto").strip().lower()

    if requested_provider == "mock":
        provider = RandomMockProvider()
        return provider, "Krasnal Mock"

    artifact_dir_env = os.environ.get("KRASNAL_MODEL_ARTIFACT_DIR")
    if not artifact_dir_env:
        raise ValueError("Either set KRASNAL_ENGINE_PROVIDER=mock or KRASNAL_MODEL_ARTIFACT_DIR")
    artifact_dir = Path(artifact_dir_env)
    if not artifact_dir.exists():
        raise ValueError(f"Model artifact directory not found: {artifact_dir}")
    provider = ModelProvider.from_artifact_dir(artifact_dir)
    return provider, provider.engine_name


def main():
    if os.environ.get("KRASNAL_ENGINE_PROVIDER") == "mock":
        print("\n" + "=" * 60, file=sys.stderr)
        print("⚠️  WARNING: Running in MOCK mode (random moves)", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

    logger.info("Starting Krasnal UCI Engine")

    provider, engine_name = build_provider()
    uci = UCIParser(provider, engine_name=engine_name)

    uci.run()


if __name__ == "__main__":
    main()
