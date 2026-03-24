from __future__ import annotations

import logging

import bulletchess
import torch

from config import MOVES_FILE, ChessGPTConfig
from model import GPT, GPTConfig
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> tuple[GPT, Tokenizer]:
    """Load a trained chess model and its tokenizer from a checkpoint."""
    mconf = ChessGPTConfig()
    tokenizer = Tokenizer(MOVES_FILE)
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        block_size=mconf.block_size,
        vocab_size=vocab_size,
        n_layer=mconf.n_layer,
        n_head=mconf.n_head,
        n_embd=mconf.n_embd,
        dropout=0.0,
        bias=mconf.bias,
    )
    model = GPT(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, tokenizer


def get_legal_token_ids(board: bulletchess.Board, tokenizer: Tokenizer) -> list[int]:
    """Map legal UCI moves on the board to token IDs."""
    return [
        tokenizer.move_to_id[uci]
        for m in board.legal_moves()
        if (uci := m.uci()) in tokenizer.move_to_id
    ]
