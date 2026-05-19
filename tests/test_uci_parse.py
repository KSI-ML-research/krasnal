"""Table-driven tests for UCI inbound line parsing."""

import pytest

from krasnal.uci_engine.go_params import GoParams
from krasnal.uci_engine.uci_parser import (
    CmdGo,
    CmdIsReady,
    CmdPosition,
    CmdQuit,
    CmdSetOption,
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
    ],
)
def test_parse_known_commands(line: str, expected_type: type, payload: str | None):
    msg = parse_uci_line(line)
    assert isinstance(msg, expected_type)
    if isinstance(msg, CmdPosition):
        assert msg.moves_uci == payload


def test_parse_go_empty():
    msg = parse_uci_line("go")
    assert isinstance(msg, CmdGo)
    assert msg.params == GoParams()


def test_parse_go_wtime_btime():
    msg = parse_uci_line("go wtime 60000 btime 59000 winc 100 binc 100")
    assert isinstance(msg, CmdGo)
    assert msg.params == GoParams(
        wtime_ms=60000,
        btime_ms=59000,
        winc_ms=100,
        binc_ms=100,
    )


def test_parse_go_ponder_ignored_for_clocks():
    msg = parse_uci_line("go ponder wtime 3000 btime 3000")
    assert isinstance(msg, CmdGo)
    assert msg.params.wtime_ms == 3000
    assert msg.params.btime_ms == 3000


def test_parse_setoption_krasnal():
    msg = parse_uci_line("setoption name KrasnalWhiteElo value 1850")
    assert isinstance(msg, CmdSetOption)
    assert msg.name == "KrasnalWhiteElo"
    assert msg.value == "1850"


def test_parse_setoption_case_insensitive_keyword():
    msg = parse_uci_line("SETOPTION name Hash value 16")
    assert isinstance(msg, CmdSetOption)
    assert msg.name == "Hash"
    assert msg.value == "16"


def test_parse_unrecognized_preserves_original_line():
    msg = parse_uci_line("debug on")
    assert isinstance(msg, Unrecognized)
    assert msg.line == "debug on"
