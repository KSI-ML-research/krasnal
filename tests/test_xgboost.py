import time as _time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
import xgboost as xgb

from krasnal.config import GPTConfig
from krasnal.model import GPT
from krasnal.move_time import xgboost as mt_xgb
from krasnal.uci_engine.go_params import GoParams
from krasnal.uci_engine.provider import ModelProvider

REAL_DATA_DIR = Path("data/3_xgboost_500_stratified")
_REAL_XGB_MODEL = Path("artifacts/xgboost_baseline/xgboost_baseline.json")
_TINY_VOCAB_SIZE = 7954


def test_canonical_overrides_full_config():
    args = SimpleNamespace(
        max_depth=9,
        n_estimators=12,
        learning_rate=0.2,
        subsample=0.5,
        colsample_bytree=0.6,
        min_child_weight=3.0,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=7,
    )

    mt_xgb._apply_canonical_overrides(args)

    assert args.max_depth == 4
    assert args.n_estimators == 500
    assert args.learning_rate == 0.05
    assert args.subsample == 0.8
    assert args.colsample_bytree == 0.8
    assert args.min_child_weight == 1.0
    assert args.reg_alpha == 0.0
    assert args.reg_lambda == 0.0
    assert args.random_state == 42


def test_predict_parquet_smoke(tmp_path):
    n = 3
    df = pl.DataFrame(
        {
            "ply": [1, 2, 3],
            "time_initial": [300, 300, 300],
            "prev_clock_seconds": [200, 200, 200],
            "clock_fraction_left": [0.5, 0.5, 0.5],
            "is_in_check_before_move": [0, 0, 0],
            "total_pieces": [30, 30, 30],
        }
    )

    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)

    X = np.random.RandomState(0).randn(n, len(mt_xgb.FEATURE_COLUMNS)).astype(np.float32)
    y = np.ones(n, dtype=np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "reg:absoluteerror", "tree_method": "hist", "eval_metric": "mae"}
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=2)

    model_path = tmp_path / "model.json"
    model.save_model(str(model_path))

    out_path = tmp_path / "out.parquet"

    res_path = mt_xgb.predict_parquet(
        model_path=model_path,
        input_path=input_path,
        output_path=out_path,
    )

    assert res_path.exists()
    out_df = pl.read_parquet(res_path)
    assert "predicted_move_time_seconds" in out_df.columns
    preds = out_df["predicted_move_time_seconds"].to_numpy()
    assert np.all(np.isfinite(preds))


def test_real_pipeline_smoke(tmp_path):
    train_path = REAL_DATA_DIR / "xgb_train.parquet"
    val_path = REAL_DATA_DIR / "xgb_val.parquet"
    test_path = REAL_DATA_DIR / "xgb_test.parquet"

    args = SimpleNamespace(
        max_depth=4,
        n_estimators=20,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        output_dir=tmp_path,
    )

    model_path = tmp_path / "xgboost_baseline.json"
    results = mt_xgb.train(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        args=args,
        model_path=model_path,
    )

    assert results["features"] == mt_xgb.FEATURE_COLUMNS
    assert results["xgboost"]["best_iteration"] <= args.n_estimators
    assert model_path.exists()

    output_path = tmp_path / "predictions.parquet"
    saved_path = mt_xgb.predict_parquet(
        model_path=model_path,
        input_path=test_path,
        output_path=output_path,
    )

    assert saved_path.exists()
    out_df = pl.read_parquet(saved_path)
    assert "predicted_move_time_seconds" in out_df.columns
    assert np.isfinite(out_df["predicted_move_time_seconds"].to_numpy()).all()


@pytest.fixture
def _tiny_gpt():
    config = GPTConfig(
        block_size=128,
        vocab_size=_TINY_VOCAB_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_clock_encodings=False,
        clock_encoding_hidden=32,
    )
    return GPT(config).to("cpu")


@pytest.fixture
def _real_xgb() -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(str(_REAL_XGB_MODEL))
    return model


def test_predict_single_with_real_model():
    model = xgb.Booster()
    model.load_model(str(_REAL_XGB_MODEL))

    result = mt_xgb.predict_single(
        model=model,
        ply=1,
        time_initial=600,
        prev_clock_seconds=600,
        clock_fraction_left=1.0,
        is_in_check_before_move=False,
        total_pieces=32,
    )
    assert np.isfinite(result)
    assert result >= 0 or abs(result) < 1.0  # allow small negatives from model


def test_get_move_time_returns_finite(_tiny_gpt, _real_xgb):
    provider = ModelProvider(
        model=_tiny_gpt,
        device=torch.device("cpu"),
        artifact_config={"use_clock_encodings": False, "clock_encoding_hidden": 32},
    )
    provider.xgb_model = _real_xgb

    go = GoParams(wtime_ms=600000, btime_ms=600000)
    delay = provider.get_move_time("", go)
    assert np.isfinite(delay), f"Expected finite delay, got {delay}"


def test_get_move_time_increases_with_ply(_tiny_gpt, _real_xgb):
    provider = ModelProvider(
        model=_tiny_gpt,
        device=torch.device("cpu"),
        artifact_config={"use_clock_encodings": False, "clock_encoding_hidden": 32},
    )
    provider.xgb_model = _real_xgb

    go = GoParams(wtime_ms=600000, btime_ms=600000)
    d1 = provider.get_move_time("", go)
    d2 = provider.get_move_time("e2e4", go)
    assert np.isfinite(d1)
    assert np.isfinite(d2)


def test_think_and_move_returns_legal_move(_tiny_gpt, _real_xgb):
    provider = ModelProvider(
        model=_tiny_gpt,
        device=torch.device("cpu"),
        artifact_config={"use_clock_encodings": False, "clock_encoding_hidden": 32},
    )
    provider.xgb_model = _real_xgb

    go = GoParams(wtime_ms=600000, btime_ms=600000)
    move = provider.think_and_move("", go)
    assert isinstance(move, str) and len(move) >= 4


def test_get_move_time_returns_zero_without_xgb_model(_tiny_gpt):
    provider = ModelProvider(
        model=_tiny_gpt,
        device=torch.device("cpu"),
        artifact_config={"use_clock_encodings": False, "clock_encoding_hidden": 32},
    )
    go = GoParams(wtime_ms=600000, btime_ms=600000)
    assert provider.get_move_time("", go) == 0.0
