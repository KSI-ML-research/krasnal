import duckdb
import polars as pl

from krasnal.preprocess.move_vocab_duckdb import build_move_vocab_from_filtered_parquet


def test_build_move_vocab_from_filtered_parquet(tmp_path):
    corpus_path = tmp_path / "games.parquet"
    pl.DataFrame(
        {
            "uci_moves": ["g1f3 e7e5", "e2e4 e7e8q"],
            "piece_moved": [["n", "p"], ["p", "q"]],
        }
    ).write_parquet(corpus_path)

    piece_aware = True
    side_prefixed = True
    vocab_path = tmp_path / "move_vocab.json"

    con = duckdb.connect()
    artifact = build_move_vocab_from_filtered_parquet(
        con,
        str(corpus_path),
        vocab_path,
        piece_aware_moves=piece_aware,
        side_prefixed_moves=side_prefixed,
    )
    con.close()

    move_vocab = {
        token: token_id
        for token, token_id in artifact["vocab"].items()
        if not token.startswith("<")
    }
    assert artifact["manifest"]["piece_aware_moves"] is True
    assert artifact["manifest"]["side_prefixed_moves"] is True
    assert sorted(move_vocab) == [
        "b:pawn:e7e5",
        "b:queen:e7e8q",
        "w:knight:g1f3",
        "w:pawn:e2e4",
    ]
