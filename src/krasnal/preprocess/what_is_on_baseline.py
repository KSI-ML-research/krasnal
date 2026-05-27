"""Build the empirical training-frequency baseline for ``what_is_on`` probes."""

from __future__ import annotations

import random
from pathlib import Path

import bulletchess
from loguru import logger
from tqdm.auto import tqdm

from krasnal.dataset import PretrainDataset
from krasnal.eval.parsers import GameTokens, parse_game_tokens
from krasnal.eval.what_is_on_baseline import WhatIsOnBaselineAccumulator
from krasnal.tokens import GAME_START_ID, PAD_ID, get_moves_only, to_uci


def build_what_is_on_baseline_counts(
    *,
    train_dataset_path: Path,
    output_path: Path,
    block_size: int,
    max_games: int,
    seed: int,
) -> None:
    ds = PretrainDataset(train_dataset_path)
    window_indices = list(range(len(ds)))
    if max_games > 0:
        random.Random(seed).shuffle(window_indices)

    accumulator = WhatIsOnBaselineAccumulator()
    games_seen = 0
    positions_seen = 0
    for idx in tqdm(window_indices, desc="what_is_on baseline", unit="window"):
        tokens, _active, _opponent, _segment, _position = ds[idx]
        token_list = tokens.tolist()
        for start, end in _game_spans(token_list):
            if max_games > 0 and games_seen >= max_games:
                break
            games_seen += 1
            game_token_ids = token_list[start:end]
            game_tokens = parse_game_tokens(game_token_ids)
            if game_tokens is None:
                continue
            game_tokens.move_tokens = get_moves_only(game_token_ids)
            positions_seen += _accumulate_game(
                accumulator,
                game_tokens,
                block_size=block_size,
            )
        if max_games > 0 and games_seen >= max_games:
            break

    accumulator.to_counts().dump(output_path)
    logger.info(
        "Wrote what_is_on baseline counts to {} from {} games ({} positions)",
        output_path,
        games_seen,
        positions_seen,
    )


def _accumulate_game(
    accumulator: WhatIsOnBaselineAccumulator,
    game_tokens: GameTokens,
    *,
    block_size: int,
) -> int:
    moves = game_tokens.move_tokens
    if not moves or len(moves) + len(game_tokens.initial_context) > block_size:
        return 0

    board = bulletchess.Board()
    positions = 0
    for move_idx, move_token in enumerate(moves):
        uci_move = to_uci(move_token)
        if not uci_move:
            break
        try:
            move = bulletchess.Move.from_uci(uci_move)
            board.apply(move)
        except ValueError:
            break
        accumulator.update_board(board, move_idx)
        positions += 1
    return positions


def _game_spans(token_list: list[int]) -> list[tuple[int, int]]:
    spans = []
    i = 0
    while i < len(token_list):
        if token_list[i] == PAD_ID:
            break
        if token_list[i] != GAME_START_ID:
            i += 1
            continue
        j = i + 1
        while j < len(token_list) and token_list[j] not in (GAME_START_ID, PAD_ID):
            j += 1
        spans.append((i, j))
        i = j
    return spans
