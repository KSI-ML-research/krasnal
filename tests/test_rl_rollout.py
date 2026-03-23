from pathlib import Path

import pytest
import torch

from src.rl.rollout import Phase1RolloutGenerator
from src.tokenizer import SOS_ID, SPECIAL_TOKENS, THINK_END_ID, THINK_START_ID, Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer(Path("data/all_uci_moves.txt"))


@pytest.fixture
def mock_model(tokenizer):
    class MockModel:
        class config:
            block_size = 128

        def __init__(self):
            self.config = MockModel.config

        def __call__(self, x):
            bsz, seq_len = x.shape
            vocab_size = tokenizer.get_vocab_size()
            logits = torch.zeros(bsz, seq_len, vocab_size)
            logits[:, -1, tokenizer.move_to_id["e2e4"]] = 10.0
            logits[:, -1, tokenizer.move_to_id["e7e5"]] = 9.0
            logits[:, -1, tokenizer.move_to_id["g1f3"]] = 8.0
            return logits, None

    return MockModel()


def test_rollout_injects_think_tokens_and_marks_generated_moves(tokenizer, mock_model):
    prompts = torch.tensor([[SOS_ID]])
    prompt_lengths = torch.tensor([1])
    generator = Phase1RolloutGenerator(mock_model, tokenizer, device="cpu", temperature=0.0)

    batch = generator.generate(
        prompts,
        prompt_lengths,
        group_size=2,
        think_min_tokens=2,
        think_max_tokens=2,
    )

    assert batch.token_ids.shape[0] == 2
    assert torch.all(batch.think_lengths == 2)
    for row, mask in zip(batch.token_ids.tolist(), batch.completion_mask.tolist(), strict=True):
        assert row[1] == THINK_START_ID
        assert row[4] == THINK_END_ID
        assert row[5] not in SPECIAL_TOKENS
        assert mask[0] == 0.0
        assert mask[1] == 0.0
        assert mask[2] == 1.0
        assert mask[3] == 1.0
        assert mask[4] == 0.0
        assert mask[5] == 1.0


def test_rollout_samples_think_length_in_requested_range(tokenizer, mock_model):
    prompts = torch.tensor([[SOS_ID]])
    prompt_lengths = torch.tensor([1])
    generator = Phase1RolloutGenerator(mock_model, tokenizer, device="cpu", temperature=0.0)

    batch = generator.generate(
        prompts,
        prompt_lengths,
        group_size=8,
        think_min_tokens=2,
        think_max_tokens=8,
    )

    assert int(batch.think_lengths.min().item()) >= 2
    assert int(batch.think_lengths.max().item()) <= 8
    for row, think_len in zip(batch.token_ids.tolist(), batch.think_lengths.tolist(), strict=True):
        assert row[1] == THINK_START_ID
        assert row[1 + 1 + think_len] == THINK_END_ID
