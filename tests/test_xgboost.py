from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from krasnal.move_time import xgboost as mt_xgb
from krasnal.uci_engine.go_params import GoParams
from krasnal.uci_engine.provider import ModelProvider


def test_predict_parquet_smoke(tmp_path):
    """Tests parquet -> model -> prediction -> parquet pipeline"""
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
    """Model for other tests"""
    rng = np.random.RandomState(42)
    X = rng.randn(20, len(mt_xgb.FEATURE_COLUMNS)).astype(np.float32)
    y = rng.uniform(1, 15, 20).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "reg:absoluteerror", "tree_method": "hist", "eval_metric": "mae"}
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=5)
    return model


def test_predict_single_returns_reasonable_range():
    """Tests that predict_single doesn't return NaN or Inf"""
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
    """Similar to test_predict_single_returns_reasonable_range
    but tests that on different game phases"""
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
    """If xgb isn't loaded the predict time should be 0"""
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


def _make_synthetic_parquet(path: Path, n: int = 500, seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    df = pl.DataFrame(
        {
            "ply": rng.randint(1, 121, n),
            "time_initial": rng.choice([180, 300, 600], n),
            "prev_clock_seconds": rng.uniform(10, 600, n).astype(np.float32),
            "clock_fraction_left": rng.uniform(0.01, 1.0, n).astype(np.float32),
            "is_in_check_before_move": rng.randint(0, 2, n),
            "total_pieces": rng.randint(4, 33, n),
        }
    )
    raw = (
        1.0
        + 1.5 * df["is_in_check_before_move"].to_numpy().astype(np.float32)
        + 0.08 * df["total_pieces"].to_numpy().astype(np.float32)
        - 2.0 * df["clock_fraction_left"].to_numpy().astype(np.float32)
        + rng.normal(0, 0.15, n).astype(np.float32)
    )
    target = np.clip(raw, 0.1, 30.0)
    df = df.with_columns(pl.Series(mt_xgb.TARGET_COLUMN, target))
    df.write_parquet(str(path))


def test_train_returns_expected_structure(tmp_path):
    for name in ("train", "val", "test"):
        _make_synthetic_parquet(tmp_path / f"{name}.parquet", n=100)

    args = SimpleNamespace(
        max_depth=3, n_estimators=20, learning_rate=0.2,
        subsample=1.0, colsample_bytree=1.0,
        min_child_weight=1.0, reg_alpha=0.0, reg_lambda=0.0,
        random_state=42,
    )
    result = mt_xgb.train(
        train_path=tmp_path / "train.parquet",
        val_path=tmp_path / "val.parquet",
        test_path=tmp_path / "test.parquet",
        args=args,
    )

    assert set(result) >= {"paths", "rows", "baselines", "xgboost"}
    assert set(result["xgboost"]) >= {"train", "val", "test", "params"}
    assert result["rows"]["train"] == 100
    assert result["rows"]["val"] == 100
    assert result["rows"]["test"] == 100


def test_xgboost_beats_constant_baselines(tmp_path):
    for name in ("train", "val", "test"):
        _make_synthetic_parquet(tmp_path / f"{name}.parquet", n=200, seed=hash(name) % 2**31)

    args = SimpleNamespace(
        max_depth=4, n_estimators=50, learning_rate=0.15,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=1.0, reg_alpha=0.0, reg_lambda=0.0,
        random_state=42,
    )
    result = mt_xgb.train(
        train_path=tmp_path / "train.parquet",
        val_path=tmp_path / "val.parquet",
        test_path=tmp_path / "test.parquet",
        args=args,
    )

    xgb_mae = result["xgboost"]["test"]["mae"]
    mean_mae = result["baselines"]["mean"]["test"]["mae"]
    med_mae = result["baselines"]["median"]["test"]["mae"]

    assert xgb_mae < mean_mae, f"XGBoost MAE {xgb_mae:.3f} ≥ mean MAE {mean_mae:.3f}"
    assert xgb_mae < med_mae, f"XGBoost MAE {xgb_mae:.3f} ≥ median MAE {med_mae:.3f}"


def test_train_saves_model(tmp_path):
    for name in ("train", "val", "test"):
        _make_synthetic_parquet(tmp_path / f"{name}.parquet", n=100)

    model_path = tmp_path / "model.json"
    args = SimpleNamespace(
        max_depth=3, n_estimators=20, learning_rate=0.2,
        subsample=1.0, colsample_bytree=1.0,
        min_child_weight=1.0, reg_alpha=0.0, reg_lambda=0.0,
        random_state=42,
    )
    mt_xgb.train(
        train_path=tmp_path / "train.parquet",
        val_path=tmp_path / "val.parquet",
        test_path=tmp_path / "test.parquet",
        args=args,
        model_path=model_path,
    )

    assert model_path.exists()
    model = xgb.Booster()
    model.load_model(str(model_path))


def test_load_split_raises_on_missing_column(tmp_path):
    df = pl.DataFrame({"ply": [1], "time_initial": [300]})
    path = tmp_path / "bad.parquet"
    df.write_parquet(str(path))
    with pytest.raises(ValueError, match="missing required columns"):
        mt_xgb._load_split(path)


def test_load_split_raises_on_empty(tmp_path):
    cols = mt_xgb.FEATURE_COLUMNS + [mt_xgb.TARGET_COLUMN]
    df = pl.DataFrame({c: [] for c in cols})
    path = tmp_path / "empty.parquet"
    df.write_parquet(str(path))
    with pytest.raises(ValueError, match="non-null rows"):
        mt_xgb._load_split(path)


def test_train_then_predict_on_new_data(tmp_path):
    for name in ("train", "val", "test"):
        _make_synthetic_parquet(tmp_path / f"{name}.parquet", n=200, seed=hash(name) % 2**31)

    model_path = tmp_path / "model.json"
    args = SimpleNamespace(
        max_depth=3, n_estimators=30, learning_rate=0.2,
        subsample=1.0, colsample_bytree=1.0,
        min_child_weight=1.0, reg_alpha=0.0, reg_lambda=0.0,
        random_state=42,
    )
    mt_xgb.train(
        train_path=tmp_path / "train.parquet",
        val_path=tmp_path / "val.parquet",
        test_path=tmp_path / "test.parquet",
        args=args,
        model_path=model_path,
    )

    new_df = pl.DataFrame(
        {
            "ply": [10, 60, 100],
            "time_initial": [300, 600, 180],
            "prev_clock_seconds": [250, 300, 30],
            "clock_fraction_left": [0.83, 0.50, 0.17],
            "is_in_check_before_move": [0, 1, 0],
            "total_pieces": [32, 20, 8],
        }
    )
    new_df.write_parquet(str(tmp_path / "new.parquet"))

    out_path = mt_xgb.predict_parquet(
        model_path=model_path,
        input_path=tmp_path / "new.parquet",
        output_path=tmp_path / "out.parquet",
    )

    result = pl.read_parquet(out_path)
    assert "predicted_move_time_seconds" in result.columns
    assert result["predicted_move_time_seconds"].is_finite().all()
    assert (result["predicted_move_time_seconds"] >= 0).all()
