#!/usr/bin/env -S uv run python

import os
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
        raise ValueError(
            "KRASNAL_MODEL_ARTIFACT_DIR must be set when using model provider, "
            "or set KRASNAL_ENGINE_PROVIDER=mock"
        )
    artifact_dir = Path(artifact_dir_env)
    provider = ModelProvider.from_artifact_dir(artifact_dir)
    return provider, provider.engine_name


def main():
    """
    Entrypoint tailored for Lichess-bot.
    Connects a specific engine implementation with the UCI loop.
    """
    logger.info("Starting Krasnal UCI Engine")

    provider, engine_name = build_provider()
    uci = UCIParser(provider, engine_name=engine_name)

    uci.run()


if __name__ == "__main__":
    main()
