from __future__ import annotations

import torch

from krasnal.config import CLOCK_IGNORE_ID


def ignore_clock_pair() -> tuple[int, int]:
    return CLOCK_IGNORE_ID, CLOCK_IGNORE_ID


def uniform_clock_pair(seconds: int) -> tuple[int, int]:
    return seconds, seconds


def new_clock_tracks(
    token_count: int,
    *,
    enabled: bool,
    initial_seconds: int | None = None,
) -> tuple[list[int], list[int], int, int]:
    if not enabled:
        return [], [], CLOCK_IGNORE_ID, CLOCK_IGNORE_ID
    if initial_seconds is None:
        raise ValueError("initial_seconds is required when clock tracks are enabled")
    return (
        [initial_seconds] * token_count,
        [initial_seconds] * token_count,
        CLOCK_IGNORE_ID,
        CLOCK_IGNORE_ID,
    )


def sync_prefix_clock_tracks(
    active_clock_ids: list[int],
    opponent_clock_ids: list[int],
    *,
    prefix_len: int,
    total_len: int,
    prefix_clock_seconds: int | None = None,
) -> tuple[list[int], list[int]]:
    tail_a = active_clock_ids[prefix_len:] if len(active_clock_ids) > prefix_len else []
    tail_o = opponent_clock_ids[prefix_len:] if len(opponent_clock_ids) > prefix_len else []
    if prefix_clock_seconds is not None:
        active = [prefix_clock_seconds] * prefix_len + tail_a
        opponent = [prefix_clock_seconds] * prefix_len + tail_o
    else:
        active = [CLOCK_IGNORE_ID] * prefix_len + tail_a
        opponent = [CLOCK_IGNORE_ID] * prefix_len + tail_o
    while len(active) < total_len:
        active.append(CLOCK_IGNORE_ID)
        opponent.append(CLOCK_IGNORE_ID)
    del active[total_len:]
    del opponent[total_len:]
    return active, opponent


def shift_clock_rows_for_training(
    active_padded: torch.Tensor,
    opponent_padded: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the training view where input token g uses the clock row for g + 1."""
    return active_padded[:, 1:], opponent_padded[:, 1:]


def clock_pair_for_input_index(
    global_input_index: int,
    *,
    context_len: int,
    per_token_active: list[int],
    per_token_opp: list[int],
    go_active_sec: int,
    go_opp_sec: int,
    enabled: bool,
) -> tuple[int, int]:
    if not enabled:
        return ignore_clock_pair()

    next_index = global_input_index + 1
    if next_index < context_len:
        return per_token_active[next_index], per_token_opp[next_index]
    if next_index == context_len:
        return go_active_sec, go_opp_sec
    return ignore_clock_pair()


def clock_pairs_for_window(
    first_global_index: int,
    count: int,
    *,
    context_len: int,
    per_token_active: list[int],
    per_token_opp: list[int],
    go_active_sec: int,
    go_opp_sec: int,
    enabled: bool,
) -> tuple[list[int], list[int]]:
    pairs = [
        clock_pair_for_input_index(
            first_global_index + i,
            context_len=context_len,
            per_token_active=per_token_active,
            per_token_opp=per_token_opp,
            go_active_sec=go_active_sec,
            go_opp_sec=go_opp_sec,
            enabled=enabled,
        )
        for i in range(count)
    ]
    return [a for a, _ in pairs], [o for _, o in pairs]
