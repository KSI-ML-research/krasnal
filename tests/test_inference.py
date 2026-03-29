#!/usr/bin/env python3
"""Test script for inference module."""

import bulletchess
import torch

from config import MOVES_FILE
from inference import DefaultSampler, InferenceSession, MoveGenerator
from model import GPT, GPTConfig
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


def test_kv_single_token():
    torch.manual_seed(7)
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

    session_no_cache = InferenceSession(model, device, use_kv_cache=False)
    session_cache = InferenceSession(model, device, use_kv_cache=True)

    # deterministic synthetic token stream within model vocab
    token_stream = [1, 17, 42, 5, 73, 19]

    for token_id in token_stream:
        probs_no_cache = session_no_cache.get_probs()
        probs_cache = session_cache.get_probs()
        assert torch.allclose(probs_no_cache, probs_cache)

        session_no_cache.feed(token_id)
        session_cache.feed(token_id)

def test_kv_multi_token():
    torch.manual_seed(7)
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

    session_no_cache = InferenceSession(model, device, use_kv_cache=False)
    session_cache = InferenceSession(model, device, use_kv_cache=True)

    # deterministic synthetic token stream within model vocab
    token_stream = [[1,3], [17, 23], [42, 2], [5, 37], [73, 19]]

    for i in range(len(token_stream)):
        probs_no_cache = session_no_cache.get_probs()
        probs_cache = session_cache.get_probs()
        assert torch.allclose(probs_no_cache, probs_cache)

        session_no_cache.feed(token_stream[i])
        session_cache.feed(token_stream[i])
