from types import SimpleNamespace

import numpy as np
import polars as pl
import xgboost as xgb

from krasnal.move_time import xgboost as mt_xgb
from krasnal.uci_engine.go_params import GoParams
from krasnal.uci_engine.provider import ModelProvider


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


def _train_tiny_xgb() -> xgb.Booster:
    rng = np.random.RandomState(42)
    X = rng.randn(20, len(mt_xgb.FEATURE_COLUMNS)).astype(np.float32)
    y = rng.uniform(1, 15, 20).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "reg:absoluteerror", "tree_method": "hist", "eval_metric": "mae"}
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=5)
    return model


def test_predict_single_returns_reasonable_range():
    model = _train_tiny_xgb()

    result = mt_xgb.predict_single(
        model=model,
        ply=10,
        time_initial=600,
        prev_clock_seconds=400,
        clock_fraction_left=0.67,
        is_in_check_before_move=False,
        total_pieces=28,
    )
    assert np.isfinite(result)
    assert 0 <= result <= 300


def test_predict_single_on_various_inputs():
    model = _train_tiny_xgb()

    for ply in (0, 1, 50, 200):
        result = mt_xgb.predict_single(
            model=model,
            ply=ply,
            time_initial=300,
            prev_clock_seconds=120,
            clock_fraction_left=0.4,
            is_in_check_before_move=True,
            total_pieces=16,
        )
        assert np.isfinite(result)
        assert result >= 0


def test_get_move_time_returns_zero_without_xgb_model():
    provider = ModelProvider(
        model=None,
        device=None,
        artifact_config={"use_clock_encodings": False, "clock_encoding_hidden": 32},
    )
    go = GoParams(wtime_ms=600000, btime_ms=600000)
    assert provider.get_move_time("", go) == 0.0


def test_feature_frame_converts_correctly():
    df = pl.DataFrame(
        {
            "ply": [0, 1, 2],
            "time_initial": [300, 300, 300],
            "prev_clock_seconds": [300, 250, 200],
            "clock_fraction_left": [1.0, 0.83, 0.67],
            "is_in_check_before_move": [0, 0, 1],
            "total_pieces": [32, 30, 28],
        }
    )

    arr = mt_xgb._feature_frame(df)
    assert arr.shape == (3, 6)
    assert arr.dtype == np.float32
    assert np.allclose(arr[0], [0, 300, 300, 1.0, 0, 32])
    assert np.allclose(arr[2], [2, 300, 200, 0.67, 1, 28])
