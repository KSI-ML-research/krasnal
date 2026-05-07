"""Table-driven tests for UCI inbound line parsing."""

import pytest

from krasnal.uci_engine.uci_parse import (
    CmdGo,
    CmdIsReady,
    CmdPosition,
    CmdQuit,
    CmdUci,
    CmdUciNewGame,
    Unrecognized,
    parse_uci_line,
)


@pytest.mark.parametrize(
    ("line", "expected_type", "payload"),
    [
        ("uci", CmdUci, None),
        ("isready", CmdIsReady, None),
        ("ucinewgame", CmdUciNewGame, None),
        ("quit", CmdQuit, None),
        ("position startpos", CmdPosition, ""),
        (
            "position startpos moves e2e4 e7e5 g1f3",
            CmdPosition,
            "e2e4 e7e5 g1f3",
        ),
        ("position fen 8/8/8/8/8/8/8/8 w - - 0 1 moves e2e4", CmdPosition, "e2e4"),
        ("go", CmdGo, ""),
        ("go ponder", CmdGo, "ponder"),
        ("go infinite", CmdGo, "infinite"),
    ],
)
def test_parse_known_commands(line: str, expected_type: type, payload: str | None):
    msg = parse_uci_line(line)
    assert isinstance(msg, expected_type)
    if isinstance(msg, CmdPosition):
        assert msg.moves_uci == payload
    elif isinstance(msg, CmdGo):
        assert msg.rest == payload


def test_parse_unrecognized_preserves_original_line():
    msg = parse_uci_line("setoption name Threads value 1")
    assert isinstance(msg, Unrecognized)
    assert msg.line == "setoption name Threads value 1"
