#!/usr/bin/env python3

import logging
import os
import sys

from engine.mock_provider import RandomMockProvider
from engine.pytorch_provider import PyTorchModelProvider
from engine.uci_parser import UCIParser


def main():
    """
    Entrypoint tailored for Lichess-bot.
    Connects a specific engine implementation with the UCI loop.
    """
    # Configure standard Python logging to output to stderr
    # This prevents polluting stdout which is strictly for UCI communication
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Krasnal UCI Engine")

    provider_name = os.environ.get("ENGINE_PROVIDER", "mock").strip().lower()
    logger.info("ENGINE_PROVIDER=%s", provider_name)

    if provider_name == "pytorch":
        temperature = float(os.environ.get("ENGINE_TEMPERATURE", "0.0"))
        top_p = float(os.environ.get("ENGINE_TOP_P", "1.0"))
        provider = PyTorchModelProvider(temperature=temperature, top_p=top_p)
    elif provider_name == "mock":
        provider = RandomMockProvider()
    else:
        raise ValueError(
            f"Unknown ENGINE_PROVIDER='{provider_name}'. Supported values: mock, pytorch"
        )

    uci = UCIParser(provider)

    uci.run()


if __name__ == "__main__":
    main()
