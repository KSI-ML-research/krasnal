"""Model loading and inference utilities."""

from contextlib import nullcontext

import chess
import torch
import torch.nn.functional as F

from config import MOVES_FILE, SOS_ID, ChessGPTConfig
from model import GPT, GPTConfig
from tokenizer import Tokenizer


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


def get_legal_token_ids(board: chess.Board, tokenizer: Tokenizer) -> list[int]:
    """Map legal UCI moves on the board to token IDs."""
    return [tokenizer.move_to_id[m.uci()] for m in board.legal_moves]


class InferenceSession:
    """Stateful move-by-move inference session.

    Maintains token context and provides next-move probability distributions.
    Currently uses full forward pass; KV cache can be added later as an
    internal optimization without changing the public API.

    Usage:
        session = InferenceSession(model, device)
        session.feed(e2e4_token_id)
        probs = session.get_probs()     # P(next_token | SOS, e2e4)
        session.feed(e7e5_token_id)
        probs = session.get_probs()     # P(next_token | SOS, e2e4, e7e5)
    """

    def __init__(self, model: GPT, device: torch.device, outcome_token: int = SOS_ID):
        self.model = model
        self.device = device
        self._amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        self.reset(outcome_token)

    def reset(self, outcome_token: int = SOS_ID) -> None:
        """Clear context and start a new game."""
        self.context: list[int] = [outcome_token]

    def feed(self, token_id: int) -> None:
        """Append a move token to the context."""
        self.context.append(token_id)

    def get_probs(self) -> torch.Tensor:
        """Return probability distribution over the next token (vocab_size,).

        Uses the full context accumulated via feed(). Without KV cache this
        runs a full forward pass each time — O(n) in context length.
        """
        x = torch.tensor([self.context], dtype=torch.long, device=self.device)
        with torch.inference_mode(), self._amp_ctx:
            logits, _ = self.model(x)  # (1, 1, vocab_size) — last position only
        return F.softmax(logits[0, -1], dim=-1)
