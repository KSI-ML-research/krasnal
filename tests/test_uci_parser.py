"""Tests for the small UCI command loop."""

import pytest

from krasnal.uci_engine.uci_parser import UCIParser, moves_from_position_command


@pytest.mark.parametrize(
    ("line", "moves"),
    [
        ("position startpos", ""),
        ("position startpos moves e2e4 e7e5 g1f3", "e2e4 e7e5 g1f3"),
        ("position fen 8/8/8/8/8/8/8/8 w - - 0 1 moves e2e4", "e2e4"),
    ],
)
def test_moves_from_position_command(line: str, moves: str):
    assert moves_from_position_command(line) == moves


def test_uci_handshake(capsys):
    parser = UCIParser()

    parser._process_command("uci")

    assert capsys.readouterr().out.splitlines() == [
        "id name Krasnal",
        "id author KSI UWr",
        "uciok",
    ]


def test_position_updates_current_moves():
    parser = UCIParser()

    parser._process_command("position startpos moves e2e4 e7e5")

    assert parser.current_moves == "e2e4 e7e5"


def test_ignored_commands_do_nothing(capsys):
    parser = UCIParser()

    parser._process_command("setoption name Hash value 16")
    parser._process_command("debug on")

    assert capsys.readouterr().out == ""
