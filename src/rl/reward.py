from __future__ import annotations

import chess
import torch

from tokenizer import SPECIAL_TOKENS, THINK_END_ID, THINK_START_ID, Tokenizer


def score_phase1_rollouts(
    token_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    think_lengths: torch.Tensor,
    tokenizer: Tokenizer,
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards: list[float] = []
    thinking_ratios: list[float] = []
    played_legality: list[float] = []

    for row, prompt_len, think_len in zip(
        token_ids.tolist(), prompt_lengths.tolist(), think_lengths.tolist(), strict=True
    ):
        prompt_tokens = row[:prompt_len]
        board = _build_board(prompt_tokens, tokenizer)
        thought_tokens = row[prompt_len + 1 : prompt_len + 1 + think_len]
        think_end_idx = prompt_len + 1 + think_len
        played_move_idx = think_end_idx + 1
        played_token = row[played_move_idx] if played_move_idx < len(row) else None

        legal_thinking_moves = 0
        think_board = board.copy()
        for token_id in thought_tokens:
            move = _token_to_move(token_id, tokenizer)
            if move is None or move not in think_board.legal_moves:
                break
            think_board.push(move)
            legal_thinking_moves += 1

        played_legal = 0.0
        played_move = _token_to_move(played_token, tokenizer)
        if played_move is not None and played_move in board.legal_moves:
            played_legal = 1.0

        think_ratio = legal_thinking_moves / max(think_len, 1)
        reward = think_ratio + (0.5 if played_legal else 0.0)
        rewards.append(reward)
        thinking_ratios.append(think_ratio)
        played_legality.append(played_legal)

    metrics = {
        "reward_mean": _mean(rewards),
        "legal_thinking_ratio": _mean(thinking_ratios),
        "played_move_legal_rate": _mean(played_legality),
    }
    return torch.tensor(rewards, dtype=torch.float), metrics


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _build_board(token_ids: list[int], tokenizer: Tokenizer) -> chess.Board:
    board = chess.Board()
    special_ids = set(SPECIAL_TOKENS)
    for token_id in token_ids:
        if token_id in special_ids:
            continue
        move = _token_to_move(token_id, tokenizer)
        if move is None or move not in board.legal_moves:
            break
        board.push(move)
    return board


def _token_to_move(token_id: int | None, tokenizer: Tokenizer) -> chess.Move | None:
    if token_id is None or token_id in {THINK_START_ID, THINK_END_ID}:
        return None
    uci = tokenizer.id_to_move.get(int(token_id), "")
    if not uci or uci.startswith("<"):
        return None
    try:
        return chess.Move.from_uci(uci)
    except ValueError:
        return None
