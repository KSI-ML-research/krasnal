"""Autouse fixture installs a minimal move vocab so tests share consistent token globals."""

import pytest

from krasnal.tokens import install_move_vocab_artifact, make_move_vocab_artifact

_TEST_MOVE_KEYS = [
    "b:c7c5",
    "b:e7e5",
    "b:g1f3",
    "w:c7c5",
    "w:e2e4",
    "w:e7e5",
]


def _install_test_vocab() -> None:
    install_move_vocab_artifact(
        make_move_vocab_artifact(
            _TEST_MOVE_KEYS,
            piece_aware_moves=False,
            side_prefixed_moves=True,
            generation_timestamp="test",
        )
    )


@pytest.fixture(autouse=True)
def _test_move_vocab() -> None:
    _install_test_vocab()
