#!/usr/bin/env python3
"""Test script for inference module."""

import chess
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

    board = chess.Board()
    move = generator.generate_move(session, board, tokenizer, sampler)

    legal_moves = [m.uci() for m in board.legal_moves]
    assert move in legal_moves, f"Invalid move: {move}"

    # Apply the generated move to keep the board in sync with the session.
    board.push_uci(move)
    session.feed(tokenizer.move_to_id[move])
    move2 = generator.generate_move(session, board, tokenizer, sampler)

    # Recompute legal moves from the updated board position for the second move.
    legal_moves_after = [m.uci() for m in board.legal_moves]
    assert move2 in legal_moves_after, f"Invalid move 2: {move2}"
