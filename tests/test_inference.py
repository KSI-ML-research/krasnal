import pytest
import torch

from krasnal.config import CLOCK_IGNORE_ID, GPTConfig
from krasnal.inference import (
    Game,
    InferenceSession,
    StatelessBatchInferenceSession,
    sample_token,
)
from krasnal.inference.kv_cache import KVCache
from krasnal.model import GPT
from krasnal.time_conditioning import (
    clock_pair_for_input_index,
    new_clock_tracks,
    sync_prefix_clock_tracks,
)
from krasnal.tokens import (
    BLACK_PREFIX,
    DRAW_ID,
    MOVE_TO_ID,
    WHITE_PREFIX,
    WHITE_WON_ID,
    get_vocab_size,
)
from krasnal.uci_engine.go_params import GoParams


def test_inference_session_feed_and_get_probs():
    device = torch.device("cpu")
    config = GPTConfig(
        block_size=128,
        vocab_size=get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_time_conditioning=False,
        time_conditioning_hidden=32,
    )
    model = GPT(config).to(device)
    session = InferenceSession(model, device, outcome_token=WHITE_WON_ID)

    e2e4 = MOVE_TO_ID[WHITE_PREFIX + "e2e4"]
    session.feed_token(e2e4)
    probs = session.get_raw_probs()

    assert probs.shape[0] == get_vocab_size()
    assert probs.sum().item() - 1.0 < 0.01


def test_sample_token_greedy():
    probs = torch.tensor([0.1, 0.5, 0.4])
    assert sample_token(probs, temperature=0.0) == 1


def test_inference_session_single_step():
    device = torch.device("cpu")
    config = GPTConfig(
        block_size=128,
        vocab_size=get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_time_conditioning=False,
        time_conditioning_hidden=32,
    )
    model = GPT(config).to(device)
    session = InferenceSession(model, device, outcome_token=WHITE_WON_ID)

    e2e4 = MOVE_TO_ID[WHITE_PREFIX + "e2e4"]
    session.feed_token(e2e4)
    probs = session.get_raw_probs()

    assert probs.shape == (get_vocab_size(),)
    assert not torch.isnan(probs).any()


def test_sample_token_temperature():
    probs = torch.tensor([0.1, 0.5, 0.4])
    result = sample_token(probs, temperature=1.0, top_p=1.0)
    assert result in [0, 1, 2]


def test_game_feed_uci_and_feed_token_keep_board_and_tokens_synchronized():
    game = Game(target_outcome_token=WHITE_WON_ID)

    game.feed_uci("e2e4")
    game.feed_token(MOVE_TO_ID[BLACK_PREFIX + "e7e5"])

    assert game.moves_uci == ["e2e4", "e7e5"]
    assert game.tokens == [
        MOVE_TO_ID[WHITE_PREFIX + "e2e4"],
        MOVE_TO_ID[BLACK_PREFIX + "e7e5"],
    ]
    assert len(game.legal_moves()) > 0


def test_stateless_batch_inference_session_returns_probs():
    device = torch.device("cpu")
    config = GPTConfig(
        block_size=128,
        vocab_size=get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_time_conditioning=False,
        time_conditioning_hidden=32,
    )
    model = GPT(config).to(device)
    batch = StatelessBatchInferenceSession(model, device)
    sequences = [
        [0, WHITE_WON_ID],
        [0, DRAW_ID, 14],
    ]

    probs = batch.get_raw_probs_batch(sequences)

    assert probs.shape == (2, get_vocab_size())
    assert not torch.isnan(probs).any()


def test_model_time_conditioning_forward_accepts_clock_tensors():
    device = torch.device("cpu")
    config = GPTConfig(
        block_size=8,
        vocab_size=get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_time_conditioning=True,
        time_conditioning_hidden=32,
    )
    model = GPT(config).to(device)
    idx = torch.tensor([[0, WHITE_WON_ID, MOVE_TO_ID[WHITE_PREFIX + "e2e4"]]], dtype=torch.long)
    clocks = torch.tensor([[CLOCK_IGNORE_ID, CLOCK_IGNORE_ID, 120]], dtype=torch.long)

    logits, loss = model(idx, idx, active_clock_ids=clocks, opponent_clock_ids=clocks)

    assert logits.shape == (1, 3, get_vocab_size())
    assert loss is not None


def test_time_conditioning_prefix_sync_preserves_tail():
    active, opponent, go_active, go_opponent = new_clock_tracks(6, enabled=True, initial_seconds=99)
    active[4] = 41
    opponent[4] = 52

    synced_active, synced_opponent = sync_prefix_clock_tracks(
        active,
        opponent,
        prefix_len=4,
        total_len=6,
        prefix_clock_seconds=99,
    )

    assert synced_active[:4] == [99] * 4
    assert synced_opponent[:4] == [99] * 4
    assert synced_active[4:] == [41, 99]
    assert synced_opponent[4:] == [52, 99]
    assert go_active == CLOCK_IGNORE_ID
    assert go_opponent == CLOCK_IGNORE_ID


def test_clock_pair_for_input_index_uses_go_clock_at_leaf():
    active = [CLOCK_IGNORE_ID, CLOCK_IGNORE_ID, 30]
    opponent = [CLOCK_IGNORE_ID, CLOCK_IGNORE_ID, 40]

    assert clock_pair_for_input_index(
        2,
        context_len=3,
        per_token_active=active,
        per_token_opp=opponent,
        go_active_sec=11,
        go_opp_sec=22,
        enabled=True,
    ) == (11, 22)


def test_prepare_go_clocks_clears_kv_cache():
    device = torch.device("cpu")
    config = GPTConfig(
        block_size=128,
        vocab_size=get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_time_conditioning=True,
        time_conditioning_hidden=32,
    )
    model = GPT(config).to(device).eval()
    session = InferenceSession(
        model,
        device,
        outcome_token=WHITE_WON_ID,
        clock_initial_seconds=180,
    )
    session.feed_uci("e2e4", clock_active=180, clock_opponent=180)
    session.get_raw_logits()
    assert session.kv_cache is not None
    session.prepare_go_clocks(GoParams(wtime_ms=60_000, btime_ms=60_000))
    assert session.kv_cache is None


def test_prepare_go_clocks_requires_wtime_and_btime():
    device = torch.device("cpu")
    config = GPTConfig(
        block_size=128,
        vocab_size=get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_time_conditioning=True,
        time_conditioning_hidden=32,
    )
    model = GPT(config).to(device).eval()
    session = InferenceSession(
        model,
        device,
        outcome_token=WHITE_WON_ID,
        clock_initial_seconds=180,
    )
    with pytest.raises(ValueError, match="wtime and btime"):
        session.prepare_go_clocks(GoParams())


def _build_test_model(device: torch.device) -> GPT:
    config = GPTConfig(
        block_size=128,
        vocab_size=get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_time_conditioning=False,
        time_conditioning_hidden=32,
    )
    model = GPT(config).to(device)
    model.eval()
    return model


def _build_kv_cache_for_model(model: GPT, device: torch.device, batch_size: int = 1) -> KVCache:
    return KVCache(
        batch_size=batch_size,
        num_layers=model.config.n_layer,
        num_heads=model.config.n_head,
        head_dim=model.config.n_embd // model.config.n_head,
        max_seq_len=model.config.block_size,
        device=device,
        dtype=torch.float32,
    )


def test_kv_cache_single_token_matches_full_prefix_logits():
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = _build_test_model(device)

    sequence = torch.tensor(
        [[0, WHITE_WON_ID, 17, 42, 5, 73, 23]],
        dtype=torch.long,
        device=device,
    )
    kv_cache = _build_kv_cache_for_model(model, device)

    for t in range(1, sequence.size(1) + 1):
        full_logits, _ = model(sequence[:, :t])
        cached_logits, _ = model(sequence[:, t - 1 : t], past_kv=kv_cache)
        assert torch.allclose(cached_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-4)


def test_kv_cache_multi_token_chunks_match_full_prefix_logits():
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = _build_test_model(device)

    sequence = torch.tensor([[0, DRAW_ID, 11, 3, 17, 23, 42, 2]], dtype=torch.long, device=device)
    kv_cache = _build_kv_cache_for_model(model, device)

    chunk_sizes = [2, 3, 3]
    start = 0
    for chunk_size in chunk_sizes:
        end = start + chunk_size
        full_logits, _ = model(sequence[:, :end])
        cached_logits, _ = model(sequence[:, start:end], past_kv=kv_cache)
        assert torch.allclose(cached_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-4)
        start = end


def test_kv_cache_reset_replays_prefix_identically():
    torch.manual_seed(11)
    device = torch.device("cpu")
    model = _build_test_model(device)
    kv_cache = _build_kv_cache_for_model(model, device)

    sequence = torch.tensor(
        [[0, DRAW_ID, 11, 3, 17, 23, 42, 2]],
        dtype=torch.long,
        device=device,
    )
    chunk_sizes = [2, 3, 3]

    def decode_chunks() -> list[torch.Tensor]:
        start = 0
        outputs: list[torch.Tensor] = []
        for chunk_size in chunk_sizes:
            end = start + chunk_size
            logits, _ = model(sequence[:, start:end], past_kv=kv_cache)
            outputs.append(logits[:, -1, :].detach().clone())
            start = end
        return outputs

    first_pass = decode_chunks()
    kv_cache.reset()
    second_pass = decode_chunks()

    assert kv_cache.get_seq_len() == sequence.size(1)
    for first, second in zip(first_pass, second_pass, strict=True):
        assert torch.allclose(first, second, atol=1e-5, rtol=1e-4)


def test_kv_cache_uniform_batch_lengths_match_full_recompute():
    torch.manual_seed(13)
    device = torch.device("cpu")
    model = _build_test_model(device)
    kv_cache = _build_kv_cache_for_model(model, device, batch_size=2)

    sequence = torch.tensor(
        [
            [0, WHITE_WON_ID, 17, 42, 5],
            [0, DRAW_ID, 11, 7, 9],
        ],
        dtype=torch.long,
        device=device,
    )

    full_logits, _ = model(sequence)
    cached_logits, _ = model(sequence, past_kv=kv_cache)

    assert kv_cache.get_seq_len() == sequence.size(1)
    assert torch.allclose(cached_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-4)


def test_inference_session_reuses_kv_cache_for_incremental_moves():
    torch.manual_seed(17)
    device = torch.device("cpu")
    model = _build_test_model(device)
    session = InferenceSession(model, device, outcome_token=WHITE_WON_ID)

    initial_logits = session.get_raw_logits()
    assert session.kv_cache is not None
    initial_seq_len = session.kv_cache.get_seq_len()
    assert initial_seq_len == len(session.context)

    session.feed_uci("e2e4")
    incremented_logits = session.get_raw_logits()

    full_context = torch.tensor([session.context], dtype=torch.long, device=device)
    full_logits, _ = model(full_context)

    assert session.kv_cache is not None
    assert session.kv_cache.get_seq_len() == len(session.context)
    assert session.kv_cache.get_seq_len() == initial_seq_len + 1
    assert torch.allclose(incremented_logits, full_logits[0, -1], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(initial_logits, incremented_logits)
