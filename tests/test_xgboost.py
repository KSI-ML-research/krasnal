import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from krasnal.move_time import xgboost as mt_xgb
from krasnal.inference.move_analysis import ply_scaling, delay_to_seconds


def test_sign_log_roundtrip():
    vals = np.array([-12.0, -1.5, 0.0, 2.0, 50.0], dtype=np.float32)
    t = mt_xgb._sign_log_transform(vals)
    inv = mt_xgb._inverse_sign_log_transform(t)
    assert np.allclose(vals, inv, rtol=1e-6, atol=1e-6)


def test_log1p_roundtrip():
    vals = np.array([0.0, 1e-6, 1.0, 10.0, 1000.0], dtype=np.float32)
    t = mt_xgb._transform_target(vals, "log1p")
    inv = mt_xgb._inverse_transform_target(t, "log1p")
    assert np.allclose(vals, inv, rtol=1e-6, atol=1e-6)


def test_predict_parquet_smoke(tmp_path):
    # prepare small DataFrame with required feature columns
    feature_columns = mt_xgb.BASE_FEATURE_COLUMNS
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

    # write input parquet
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)

    # train a tiny xgboost model on random data matching feature dims
    X = np.random.RandomState(0).randn(n, len(feature_columns)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "reg:absoluteerror", "tree_method": "hist", "eval_metric": "mae"}
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=2)

    model_path = tmp_path / "model.json"
    model.save_model(str(model_path))

    out_path = tmp_path / "out.parquet"

    # run prediction using the module helper
    res_path = mt_xgb.predict_parquet(
        model_path=model_path,
        input_path=input_path,
        output_path=out_path,
        feature_columns=feature_columns,
        target_mode="residual",
        target_transform="log1p",
    )

    assert res_path.exists()
    out_df = pl.read_parquet(res_path)
    assert "predicted_move_time_seconds" in out_df.columns
    preds = out_df["predicted_move_time_seconds"].to_numpy()
    # expected heuristic predictions
    ply = df["ply"].to_numpy()
    heur = np.array([delay_to_seconds(ply_scaling(int(p)) * 1.0) for p in ply], dtype=np.float32)
    # predictions should be finite and close to heuristic (since model residuals are zero)
    assert np.all(np.isfinite(preds))
    assert np.allclose(preds, heur, rtol=1e-5, atol=1e-5)
