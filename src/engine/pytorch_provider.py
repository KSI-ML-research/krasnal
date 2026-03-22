from __future__ import annotations

import logging

import chess
import torch

from config import MODEL_PATH, MOVES_FILE, ChessGPTConfig
from engine.provider import ChessModelProvider
from inference.generator import MoveGenerator
from inference.sampler import DefaultSampler
from inference.session import InferenceSession
from model import GPT, GPTConfig
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class PyTorchModelProvider(ChessModelProvider):
    def __init__(self, model_path=MODEL_PATH, temperature: float = 0.0, top_p: float = 1.0):
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = Tokenizer(MOVES_FILE)
        vocab_size = self.tokenizer.get_vocab_size()

        mconf = ChessGPTConfig()
        model_config = GPTConfig(
            block_size=mconf.block_size,
            vocab_size=vocab_size,
            n_layer=mconf.n_layer,
            n_head=mconf.n_head,
            n_embd=mconf.n_embd,
            dropout=mconf.dropout,
            bias=mconf.bias,
        )
        self.model = GPT(model_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Train a model first or mount a valid checkpoint."
            )

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.generator = MoveGenerator()
        self.sampler = DefaultSampler()

        logger.info("Loaded PyTorch model provider on device=%s", self.device)

    def get_best_move(self, uci_moves: str) -> str:
        board = chess.Board()
        session = InferenceSession(self.model, self.device)

        if uci_moves.strip():
            for move_str in uci_moves.strip().split():
                try:
                    move = chess.Move.from_uci(move_str)
                    if move not in board.legal_moves:
                        logger.warning("Illegal move in history: %s", move_str)
                        return "0000"
                    board.push(move)

                    token_id = self.tokenizer.move_to_id.get(move_str)
                    if token_id is None:
                        logger.warning("Move not found in tokenizer vocab: %s", move_str)
                    else:
                        session.feed(token_id)
                except ValueError:
                    logger.warning("Invalid UCI move in history: %s", move_str)
                    return "0000"

        return self.generator.generate_move(
            session,
            board,
            self.tokenizer,
            self.sampler,
            temperature=self.temperature,
            top_p=self.top_p,
        )
