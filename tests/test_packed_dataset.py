import polars as pl
import torch

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.dataset import PretrainDataset, make_collate_fn, make_packed_collate_fn
from krasnal.preprocess import PackedWindowBuilder, pack_games_into_windows
from krasnal.supervised_target_mask import LOSS_IGNORE_INDEX
from krasnal.tokens import GAME_END_ID, GAME_START_ID, PAD_ID, WHITE_WON_ID


def test_pack_restarts_split_game_from_start_in_next_window():
    block_size = 8
    window_size = block_size + 1
    games = pl.DataFrame(
        {
            "token_ids": [
                [GAME_START_ID, WHITE_WON_ID, 100, GAME_END_ID],
                [GAME_START_ID, WHITE_WON_ID, 200, 201, 202, 203, 204, GAME_END_ID],
            ],
            "active_clock_ids": [
                [CLOCK_IGNORE_ID] * 4,
                [CLOCK_IGNORE_ID] * 8,
            ],
            "opponent_clock_ids": [
                [CLOCK_IGNORE_ID] * 4,
                [CLOCK_IGNORE_ID] * 8,
            ],
        }
    )

    packed = pack_games_into_windows(games, block_size=block_size, seed=0)
    assert len(packed) == 2

    row0 = packed.row(0, named=True)
    row1 = packed.row(1, named=True)
    assert len(row0["token_ids"]) == window_size
    assert len(row1["token_ids"]) == window_size

    # Window 1 ends with a prefix of game B; window 2 restarts game B from <game_start>.
    assert row0["token_ids"][0] == GAME_START_ID
    assert row1["token_ids"][0] == GAME_START_ID
    assert row1["token_ids"][-1] == PAD_ID


def test_pack_games_emits_fixed_window_size():
    block_size = 8
    games = pl.DataFrame(
        {
            "token_ids": [
                [GAME_START_ID, WHITE_WON_ID, 100, GAME_END_ID],
                [GAME_START_ID, WHITE_WON_ID, 200, 201, GAME_END_ID],
            ],
            "active_clock_ids": [
                [CLOCK_IGNORE_ID] * 4,
                [CLOCK_IGNORE_ID] * 5,
            ],
            "opponent_clock_ids": [
                [CLOCK_IGNORE_ID] * 4,
                [CLOCK_IGNORE_ID] * 5,
            ],
        }
    )

    packed = pack_games_into_windows(games, block_size=block_size, seed=0)
    window_size = block_size + 1

    assert len(packed) == 1
    row = packed.row(0, named=True)
    assert len(row["token_ids"]) == window_size

    non_pad = [tok for tok in row["token_ids"] if tok != PAD_ID]
    assert len(non_pad) == 9
    assert non_pad.count(GAME_START_ID) == 2


def test_packed_collate_masks_boundary_and_pad_targets():
    block_size = 4
    games = pl.DataFrame(
        {
            "token_ids": [
                [GAME_START_ID, 10, GAME_END_ID],
                [GAME_START_ID, 20, GAME_END_ID],
            ],
            "active_clock_ids": [[CLOCK_IGNORE_ID] * 3, [CLOCK_IGNORE_ID] * 3],
            "opponent_clock_ids": [[CLOCK_IGNORE_ID] * 3, [CLOCK_IGNORE_ID] * 3],
        }
    )
    packed = pack_games_into_windows(games, block_size=block_size, seed=1)
    row = packed.row(0, named=True)
    batch = [
        (
            torch.tensor(row["token_ids"], dtype=torch.long),
            torch.tensor(row["active_clock_ids"], dtype=torch.long),
            torch.tensor(row["opponent_clock_ids"], dtype=torch.long),
        )
    ]
    collate = make_packed_collate_fn()
    x, _active_x, _opponent_x, y = collate(batch)

    assert x.shape == (1, block_size)
    assert y[0, 2].item() == LOSS_IGNORE_INDEX
    assert y[0, -1].item() == LOSS_IGNORE_INDEX


def test_single_game_packed_matches_unpacked_collate_on_supervised_positions():
    tokens = torch.tensor(
        [GAME_START_ID, WHITE_WON_ID, 500, 501, GAME_END_ID],
        dtype=torch.long,
    )
    clocks = torch.full_like(tokens, CLOCK_IGNORE_ID)
    block_size = 8

    games = pl.DataFrame(
        {
            "token_ids": [tokens.tolist()],
            "active_clock_ids": [clocks.tolist()],
            "opponent_clock_ids": [clocks.tolist()],
        }
    )
    packed = pack_games_into_windows(games, block_size=block_size, seed=0)
    row = packed.row(0, named=True)

    packed_collate = make_packed_collate_fn()
    px, pa, po, py = packed_collate(
        [
            (
                torch.tensor(row["token_ids"], dtype=torch.long),
                torch.tensor(row["active_clock_ids"], dtype=torch.long),
                torch.tensor(row["opponent_clock_ids"], dtype=torch.long),
            )
        ]
    )

    plain_collate = make_collate_fn()
    x, a, o, y = plain_collate([(tokens, clocks, clocks)])

    game_len = tokens.size(0)
    for i in range(game_len - 1):
        assert px[0, i].item() == x[0, i].item()
        assert py[0, i].item() == y[0, i].item()
        assert pa[0, i].item() == a[0, i].item()
        assert po[0, i].item() == o[0, i].item()


def test_packed_builder_writes_mmap_dataset(tmp_path):
    block_size = 4
    games = pl.DataFrame(
        {
            "token_ids": [
                [GAME_START_ID, 10, GAME_END_ID],
                [GAME_START_ID, 20, GAME_END_ID],
            ],
            "active_clock_ids": [[CLOCK_IGNORE_ID] * 3, [CLOCK_IGNORE_ID] * 3],
            "opponent_clock_ids": [[CLOCK_IGNORE_ID] * 3, [CLOCK_IGNORE_ID] * 3],
        }
    )
    builder = PackedWindowBuilder(block_size, flush_every=1)
    builder.feed_from_columns(
        games["token_ids"].to_list(),
        games["active_clock_ids"].to_list(),
        games["opponent_clock_ids"].to_list(),
    )
    builder.drain()
    builder.maybe_flush(tmp_path / "parts")
    builder.finish(tmp_path / "pretrain", part_dir=tmp_path / "parts")

    ds = PretrainDataset(tmp_path / "pretrain")
    assert len(ds) == 2
    tokens, active, opponent = ds[0]
    assert tokens.dtype == torch.long
    assert active.dtype == torch.long
    assert opponent.dtype == torch.long
    assert tokens.shape == (block_size + 1,)
    assert tokens[0].item() == GAME_START_ID
