from krasnal.preprocess.stats import (
    elo_game_counts_by_white,
    elo_rating_counts_for_players,
)


def test_elo_rating_counts_for_players_counts_both_sides():
    counts = elo_rating_counts_for_players([1850], [2250])
    assert counts["<elo_1800_1899>"] == 1
    assert counts["<elo_above_2200>"] == 1


def test_elo_game_counts_by_white_counts_one_per_game():
    counts = elo_game_counts_by_white([1850, 2250, 2250])
    assert counts["<elo_1800_1899>"] == 1
    assert counts["<elo_above_2200>"] == 2
