from krasnal.uci_engine.go_params import GoParams, parse_go_rest, uci_ms_to_clock_seconds


def test_uci_ms_to_clock_seconds():
    assert uci_ms_to_clock_seconds(0) == 0
    assert uci_ms_to_clock_seconds(1500) == 1
    assert uci_ms_to_clock_seconds(59999) == 59


def test_parse_go_rest_collects_int_fields():
    p = parse_go_rest("depth 12 wtime 100 btime 90 nodes 3 movetime 500")
    assert p == GoParams(wtime_ms=100, btime_ms=90, movetime_ms=500)
