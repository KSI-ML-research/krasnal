#!/usr/bin/env python3
"""Test script for inference module."""

import bulletchess
import pytest
import torch

from config import MOVES_FILE, GPTConfig
from inference import BatchInferenceSession, DefaultSampler, InferenceSession, MoveGenerator
from model import GPT
from tokenizer import Tokenizer


def test_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(MOVES_FILE)

    config = GPTConfig(
        block_size=128,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    model = GPT(config).to(device)

    session = InferenceSession(model, device)
    sampler = DefaultSampler()
    generator = MoveGenerator()

    board = bulletchess.Board()
    move = generator.generate_move(session, board, tokenizer, sampler)

    legal_moves = [m.uci() for m in board.legal_moves()]
    assert move in legal_moves, f"Invalid move: {move}"

    board.apply(bulletchess.Move.from_uci(move))
    move2 = generator.generate_move(session, board, tokenizer, sampler)
    legal_moves = [m.uci() for m in board.legal_moves()]
    assert move2 in legal_moves, f"Invalid move 2: {move2}"


def test_batch_inference_matches_sequential_for_mixed_prefix_lengths():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(MOVES_FILE)

    config = GPTConfig(
        block_size=128,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    model = GPT(config).to(device)
    model.eval()

    session = InferenceSession(model, device)
    batch_session = BatchInferenceSession(model, device)

    e2e4 = tokenizer.move_to_id["e2e4"]
    e7e5 = tokenizer.move_to_id["e7e5"]
    g1f3 = tokenizer.move_to_id["g1f3"]

    sequences = [
        [tokenizer.sos_id],
        [tokenizer.sos_id, e2e4],
        [tokenizer.sos_id, e2e4, e7e5, g1f3],
    ]

    batch_probs = batch_session.get_probs_batch(sequences, batch_size=len(sequences))

    for i, seq in enumerate(sequences):
        session.reset()
        for token_id in seq[1:]:
            session.feed(token_id)
        seq_probs = session.get_probs()
        assert torch.allclose(batch_probs[i], seq_probs, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_batch_inference_handles_single_token_context(batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(MOVES_FILE)

    config = GPTConfig(
        block_size=32,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=1,
        n_head=1,
        n_embd=32,
    )
    model = GPT(config).to(device)
    model.eval()

    batch_session = BatchInferenceSession(model, device)
    probs = batch_session.get_probs_batch([[tokenizer.sos_id]] * batch_size, batch_size=batch_size)

    assert probs.shape == (batch_size, tokenizer.get_vocab_size())
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(batch_size, device=device, dtype=probs.dtype), atol=1e-6
    )
