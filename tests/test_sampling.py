from krasnal.sampling import sample_bool, whats_on_square_index


def test_whats_on_square_index_deterministic():
    kwargs = dict(
        post_move_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        game_key="e2e4 e7e5",
        ply=0,
        seed=123,
    )
    assert whats_on_square_index(**kwargs) == whats_on_square_index(**kwargs)


def test_whats_on_square_index_game_key_changes_square():
    a = whats_on_square_index(
        post_move_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        game_key="e2e4 e7e5 g1f3",
        ply=1,
        seed=99,
    )
    b = whats_on_square_index(
        post_move_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        game_key="d2d4 d7d5 c2c4",
        ply=1,
        seed=99,
    )
    assert a != b


def test_sample_bool_deterministic():
    kwargs = dict(seed=42, game_key="e2e4 e7e5", ply=5, probability=0.5)
    assert sample_bool(**kwargs) == sample_bool(**kwargs)


def test_sample_bool_always_false_at_prob_zero():
    assert sample_bool(seed=0, game_key="x", ply=0, probability=0.0) is False


def test_sample_bool_always_true_at_prob_one():
    assert sample_bool(seed=0, game_key="x", ply=0, probability=1.0) is True


def test_sample_bool_different_seeds_differ():
    a = sample_bool(seed=1, game_key="e2e4", ply=0, probability=0.5)
    b = sample_bool(seed=2, game_key="e2e4", ply=0, probability=0.5)
    assert a != b
