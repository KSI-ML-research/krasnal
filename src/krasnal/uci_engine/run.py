#!/usr/bin/env -S uv run python
"""UCI entrypoint. When lichess-bot runs ``../.venv/bin/python ../src/.../run.py``, ``src`` may
not be on ``sys.path``; bootstrap before importing ``krasnal``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ``run.py`` lives at ``src/krasnal/uci_engine/run.py`` — add ``src`` for ``import krasnal``.
_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from loguru import logger  # noqa: E402

from krasnal.uci_engine.uci_parser import UCIParser  # noqa: E402

_UCI_LOG_FORMAT = "{time:HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"


def configure_uci_stderr_logging() -> None:
    """Loguru to stderr: ERROR by default, DEBUG when ``KRASNAL_UCI_VERBOSE`` is set."""
    logger.remove()
    level = "DEBUG" if os.environ.get("KRASNAL_UCI_VERBOSE") else "ERROR"
    logger.add(sys.stderr, level=level, format=_UCI_LOG_FORMAT)


def main() -> None:
    configure_uci_stderr_logging()

    if os.environ.get("KRASNAL_ENGINE_PROVIDER") == "mock" and os.environ.get(
        "KRASNAL_UCI_VERBOSE"
    ):
        print("\n" + "=" * 60, file=sys.stderr)
        print("WARNING: Running in MOCK mode (random moves)", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

    # Defer ``build_provider()`` until ``isready`` / first use so python-chess gets ``uciok``
    # quickly while torch / weights load.
    uci = UCIParser(lazy_start=True)
    uci.run()


if __name__ == "__main__":
    main()
