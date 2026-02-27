#!/usr/bin/env python3

from engine.mock_provider import RandomMockProvider
from engine.uci_parser import UCIParser

def main():
    """
    Entrypoint tailored for Lichess-bot. Connects a specific engine implementation with the UCI loop.
    """
    # TODO: Environment variable `ENGINE_ENV` to determine which 
    # Provider to inject (PyTorch Model or MockProvider).
    # For now, we hardcode the Mock use.

    provider = RandomMockProvider()

    uci = UCIParser(provider)

    uci.run()

if __name__ == "__main__":
    main()
