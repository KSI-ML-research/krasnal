from pathlib import Path

import pytest

from src.tokenizer import (
    DRAW_ID,
    EOS_ID,
    PAD_ID,
    SOS_ID,
    SPECIAL_TOKENS,
    STEP_BACK_ID,
    THINK_END_ID,
    THINK_START_ID,
    WIN_BLACK_ID,
    WIN_WHITE_ID,
    Tokenizer,
)


@pytest.fixture
def tokenizer():
    return Tokenizer(Path("data/all_uci_moves.txt"))


class TestSpecialTokenIds:
    def test_think_start_id_value(self):
        assert THINK_START_ID == 6

    def test_think_end_id_value(self):
        assert THINK_END_ID == 7

    def test_step_back_id_value(self):
        assert STEP_BACK_ID == 8

    def test_special_tokens_list(self):
        assert len(SPECIAL_TOKENS) == 9
        assert SOS_ID in SPECIAL_TOKENS
        assert EOS_ID in SPECIAL_TOKENS
        assert PAD_ID in SPECIAL_TOKENS
        assert WIN_WHITE_ID in SPECIAL_TOKENS
        assert WIN_BLACK_ID in SPECIAL_TOKENS
        assert DRAW_ID in SPECIAL_TOKENS
        assert THINK_START_ID in SPECIAL_TOKENS
        assert THINK_END_ID in SPECIAL_TOKENS
        assert STEP_BACK_ID in SPECIAL_TOKENS


class TestTokenizerCoTTokens:
    def test_think_start_token_in_vocab(self, tokenizer):
        assert "<think>" in tokenizer.move_to_id
        assert tokenizer.move_to_id["<think>"] == THINK_START_ID

    def test_think_end_token_in_vocab(self, tokenizer):
        assert "</think>" in tokenizer.move_to_id
        assert tokenizer.move_to_id["</think>"] == THINK_END_ID

    def test_branch_token_in_vocab(self, tokenizer):
        assert "<branch>" in tokenizer.move_to_id
        assert tokenizer.move_to_id["<branch>"] == STEP_BACK_ID

    def test_think_start_id_to_move(self, tokenizer):
        assert tokenizer.id_to_move[THINK_START_ID] == "<think>"

    def test_think_end_id_to_move(self, tokenizer):
        assert tokenizer.id_to_move[THINK_END_ID] == "</think>"

    def test_branch_id_to_move(self, tokenizer):
        assert tokenizer.id_to_move[STEP_BACK_ID] == "<branch>"

    def test_tokenizer_attributes(self, tokenizer):
        assert tokenizer.think_start_id == THINK_START_ID
        assert tokenizer.think_end_id == THINK_END_ID
        assert tokenizer.step_back_id == STEP_BACK_ID


class TestTokenizerEncodeDecode:
    def test_encode_think_tokens(self, tokenizer):
        encoded = tokenizer.encode("<think> e2e4 </think>")
        assert encoded == [THINK_START_ID, tokenizer.move_to_id["e2e4"], THINK_END_ID]

    def test_encode_with_branch(self, tokenizer):
        encoded = tokenizer.encode("<branch>")
        assert encoded == [STEP_BACK_ID]

    def test_decode_think_tokens(self, tokenizer):
        decoded = tokenizer.decode([THINK_START_ID, tokenizer.move_to_id["e2e4"], THINK_END_ID])
        assert "<think>" in decoded
        assert "</think>" in decoded


class TestTokenizerVocabSize:
    def test_vocab_size_with_cot_tokens(self, tokenizer):
        num_moves = len(tokenizer.move_to_id)
        assert num_moves == len(SPECIAL_TOKENS) + 1968
