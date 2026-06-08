import os
import sys

import chess
import chess.engine
import pytest

from krasnal.uci_engine.provider import ChessModelProvider, ModelProvider, ModelProviderError
from krasnal.uci_engine.uci_parser import UCIParser
from krasnal.utils import write_artifact_config_json


class FailingProvider(ChessModelProvider):
    def reset_session(self) -> None:
        return None

    def get_best_move(self, _uci_moves: str) -> str:
        raise ModelProviderError("model crashed")


def test_uci_reports_no_move_on_unrecoverable_model_provider_error(capsys):
    parser = UCIParser(FailingProvider())

    parser._process_command("ucinewgame")
    parser._process_command("go")

    output = capsys.readouterr().out.splitlines()
    assert any("krasnal-uci ModelProviderError: model crashed" in line for line in output)
    assert "bestmove (none)" in output


def test_uci_handshake_does_not_load_lazy_provider(monkeypatch, capsys):
    called = False

    def fail_if_called():
        nonlocal called
        called = True
        raise AssertionError("provider loaded during uci handshake")

    monkeypatch.setattr("krasnal.uci_engine.bootstrap.build_provider", fail_if_called)
    parser = UCIParser(lazy_start=True)

    parser._process_command("uci")

    output = capsys.readouterr().out.splitlines()
    assert output[-1] == "uciok"
    assert called is False


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("outcome_conditioning_enabled", True, "outcome_conditioning"),
        ("use_clock_encodings", True, "clock_encodings"),
        ("time_control_token_enabled", True, "time_control_token"),
        ("opponent_material_enabled", True, "opponent_material"),
        ("include_elo", True, "elo_tokens"),
    ],
)
def test_model_provider_rejects_unsupported_inference_artifacts(
    tmp_path,
    flag: str,
    value: bool,
    expected: str,
):
    cfg = {
        "block_size": 128,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 64,
        "vocab_size": 9000,
        "dropout": 0.0,
        "use_clock_encodings": False,
        "clock_encoding_hidden": 1,
        "mlp_activation": "swiglu",
        "outcome_conditioning_enabled": False,
        "time_control_token_enabled": False,
        "opponent_material_enabled": False,
        "include_elo": False,
    }
    cfg[flag] = value
    write_artifact_config_json(
        tmp_path,
        cfg,
    )

    with pytest.raises(ModelProviderError, match=expected):
        ModelProvider.from_artifact_dir(tmp_path)


@pytest.mark.asyncio
async def test_uci_integration():
    # Run the engine as a separate process with mock provider
    env = os.environ.copy()
    env["KRASNAL_ENGINE_PROVIDER"] = "mock"
    _, engine = await chess.engine.popen_uci(
        [sys.executable, "-m", "krasnal.uci_engine.run"], env=env
    )

    try:
        # popen_uci automatically executes 'uci' and 'isready' during initialization

        # Testing move generation from the starting position
        board = chess.Board()
        result = await engine.play(board, chess.engine.Limit(time=0.1))
        assert result.move is not None
        assert result.move in board.legal_moves

        # Testing move generation after a move history
        board.push(result.move)
        board.push(chess.Move.from_uci("e7e5"))  # Example black move
        result = await engine.play(board, chess.engine.Limit(time=0.1))
        assert result.move is not None
        assert result.move in board.legal_moves
    finally:
        # Properly close the engine (send 'quit')
        await engine.quit()
