#!/usr/bin/env python3

import logging
import sys

from engine.mock_provider import RandomMockProvider
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

    # TODO: Environment variable `ENGINE_ENV` to determine which
    # Provider to inject (PyTorch Model or MockProvider).
    # For now, we hardcode the Mock use.

    provider = RandomMockProvider()

    uci = UCIParser(provider)

    uci.run()


if __name__ == "__main__":
    main()
