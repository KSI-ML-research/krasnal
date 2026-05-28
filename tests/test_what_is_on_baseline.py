from krasnal.eval.what_is_on_baseline import WhatIsOnBaselineCounts
from krasnal.tokens import COLORED_PIECE_TOKENS, EMPTY_ID


def test_baseline_falls_back_to_square_marginal():
    w_pawn = COLORED_PIECE_TOKENS["<w:pawn>"]
    counts = WhatIsOnBaselineCounts(
        {("e4", 0): {w_pawn: 1}},
        {"e4": {EMPTY_ID: 2, w_pawn: 1}},
    )
    assert counts.predict("e4", 0) == w_pawn
    assert counts.predict("e4", 999) == EMPTY_ID
