import torch

from krasnal.config import GPTConfig
from krasnal.inference import (
    Game,
    InferenceSession,
    StatelessBatchInferenceSession,
    sample_token,
)
from krasnal.model import GPT
from krasnal.tokens import (
    BLACK_PREFIX,
    DRAW_ID,
    MOVE_TO_ID,
    THINK_END_ID,
    THINK_START_ID,
    WHITE_PREFIX,
    WHITE_WON_ID,
    get_vocab_size,
)


def test_inference_session_feed_and_get_probs():
    device = torch.device("cpu")
    config = GPTConfig(block_size=128, vocab_size=get_vocab_size(), n_layer=2, n_head=2, n_embd=64)
    model = GPT(config).to(device)
    session = InferenceSession(model, device, outcome_token=WHITE_WON_ID)

    e2e4 = MOVE_TO_ID[WHITE_PREFIX + "e2e4"]
    session.feed_token(e2e4)
    probs = session.get_raw_probs()

    assert probs.shape[0] == get_vocab_size()
    assert probs.sum().item() - 1.0 < 0.01

def test_kv_single_token():
    torch.manual_seed(7)
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
    model.eval()

    session_no_cache = InferenceSession(model, device, use_kv_cache=False)
    session_cache = InferenceSession(model, device, use_kv_cache=True)

    # deterministic synthetic token stream within model vocab
    token_stream = [1, 17, 42, 5, 73, 19]

    for token_id in token_stream:
        probs_no_cache = session_no_cache.get_probs()
        probs_cache = session_cache.get_probs()
        assert torch.allclose(probs_no_cache, probs_cache)

        session_no_cache.feed(token_id)
        session_cache.feed(token_id)

def test_kv_multi_token():
    torch.manual_seed(7)
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
    model.eval()

    session_no_cache = InferenceSession(model, device, use_kv_cache=False)
    session_cache = InferenceSession(model, device, use_kv_cache=True)

    # deterministic synthetic token stream within model vocab
    token_stream = [[1,3], [17, 23], [42, 2], [5, 37], [73, 19]]

    for i in range(len(token_stream)):
        probs_no_cache = session_no_cache.get_probs()
        probs_cache = session_cache.get_probs()
        assert torch.allclose(probs_no_cache, probs_cache)

        session_no_cache.feed(token_stream[i])
        session_cache.feed(token_stream[i])
def test_sample_token_greedy():
    probs = torch.tensor([0.1, 0.5, 0.4])
    assert sample_token(probs, temperature=0.0) == 1


def test_inference_session_single_step():
    device = torch.device("cpu")
    config = GPTConfig(block_size=128, vocab_size=get_vocab_size(), n_layer=2, n_head=2, n_embd=64)
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


def test_inference_session_think_tokens_do_not_mutate_game_state():
    device = torch.device("cpu")
    config = GPTConfig(block_size=128, vocab_size=get_vocab_size(), n_layer=2, n_head=2, n_embd=64)
    model = GPT(config).to(device)
    session = InferenceSession(model, device, outcome_token=WHITE_WON_ID)

    session.feed_uci("e2e4")
    session.feed_token(THINK_START_ID)
    session.feed_token(MOVE_TO_ID[BLACK_PREFIX + "e7e5"])
    session.feed_token(THINK_END_ID)

    assert session.game.moves_uci == ["e2e4"]
    assert session.game.tokens == [MOVE_TO_ID[WHITE_PREFIX + "e2e4"]]
    assert session.context[-1] == THINK_END_ID


def test_stateless_batch_inference_session_returns_probs():
    device = torch.device("cpu")
    config = GPTConfig(block_size=128, vocab_size=get_vocab_size(), n_layer=2, n_head=2, n_embd=64)
    model = GPT(config).to(device)
    batch = StatelessBatchInferenceSession(model, device)
    sequences = [
        [0, WHITE_WON_ID],
        [0, DRAW_ID, 14],
    ]

    probs = batch.get_raw_probs_batch(sequences)

    assert probs.shape == (2, get_vocab_size())
    assert not torch.isnan(probs).any()
