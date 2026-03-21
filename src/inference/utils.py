from __future__ import annotations

import logging
from pathlib import Path

import chess
import torch

from ..config import MOVES_FILE, ChessGPTConfig
from ..model import GPT, GPTConfig
from ..tokenizer import Tokenizer, load_tokenizer_from_sidecar, tokenizer_sidecar_path_for_artifact

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> tuple[GPT, Tokenizer]:
    """Load a trained chess model and its tokenizer from a checkpoint."""
    mconf = ChessGPTConfig()
    artifact_path = Path(model_path)
    sidecar_path = tokenizer_sidecar_path_for_artifact(artifact_path)
    if sidecar_path.exists():
        tokenizer = load_tokenizer_from_sidecar(sidecar_path)
    else:
        logger.warning(
            "Tokenizer sidecar not found for %s, falling back to %s. "
            "This may cause token-id drift against preprocessed datasets.",
            artifact_path,
            MOVES_FILE,
        )
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


def get_legal_token_ids(board: chess.Board, tokenizer: Tokenizer) -> list[int]:
    """Map legal UCI moves on the board to token IDs."""
    return [tokenizer.move_to_id[m.uci()] for m in board.legal_moves]
