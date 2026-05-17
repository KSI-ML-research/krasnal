from krasnal.lichess_clocks import extract_uci_moves_and_clocks


def test_extract_uci_moves_and_clocks_parses_clock_comments() -> None:
    pgn = """[Event "?"]
[Site "?"]
[Date "2024.01.01"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "*"]

1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:04:58] }
2. Nf3 { [%clk 0:04:57.5] } 2... Nc6 { [%clk 0:04:56] }
*"""

    uci_moves, move_clocks_seconds = extract_uci_moves_and_clocks(pgn)

    assert uci_moves == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert move_clocks_seconds == [300.0, 298.0, 297.5, 296.0]