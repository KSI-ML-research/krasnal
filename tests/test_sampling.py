from krasnal.sampling import whats_on_square_index


def test_whats_on_square_index_deterministic():
    kwargs = dict(
        game_key="e2e4 e7e5",
        ply=0,
        seed=123,
    )
    assert whats_on_square_index(**kwargs) == whats_on_square_index(**kwargs)


def test_whats_on_square_index_game_key_changes_square():
    a = whats_on_square_index(
        game_key="e2e4 e7e5 g1f3",
        ply=1,
        seed=99,
    )
    b = whats_on_square_index(
        game_key="d2d4 d7d5 c2c4",
        ply=1,
        seed=99,
    )
    assert a != b
