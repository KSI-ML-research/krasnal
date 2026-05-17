#!/usr/bin/env python3
"""Train and evaluate a simple XGBoost baseline on move-time prediction.

The baseline uses a small, explicit feature set derived from the move-level
clock parquet files:
  - ply
  - clock_after_seconds
  - side_to_move

It also reports two trivial reference baselines for comparison:
  - mean predictor trained on the train split
  - median predictor trained on the train split

The goal is to establish a clean first learned baseline before feature tuning.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

# heuristic baseline  from move_analysis
from krasnal.inference.move_analysis import ply_scaling, delay_to_seconds


DEFAULT_TRAIN_PATH = Path("data/3_xgboost/xgb_train.parquet")
DEFAULT_VAL_PATH = Path("data/3_xgboost/xgb_val.parquet")
DEFAULT_TEST_PATH = Path("data/3_xgboost/xgb_test.parquet")
DEFAULT_OUTPUT_DIR = Path("artifacts/xgboost_baseline")

FEATURE_COLUMNS = [
    "ply",
    "clock_after_seconds",
    "side_to_move",
    "prev_clock_seconds",
    "clock_diff_seconds",
    "is_in_check_before_move",
]
TARGET_COLUMN = "target_move_time_seconds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple XGBoost baseline for move-time prediction."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet split: {path}")

    df = pl.read_parquet(path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    clean = df.select(FEATURE_COLUMNS + [TARGET_COLUMN]).drop_nulls()
    if clean.is_empty():
        raise ValueError(f"{path} does not contain any non-null rows")

    features = np.column_stack(
        [clean[column].to_numpy().astype(np.float32, copy=False) for column in FEATURE_COLUMNS]
    )
    target = clean[TARGET_COLUMN].to_numpy().astype(np.float32, copy=False)
    return features, target, clean


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)
    mse = float(np.mean(np.square(residuals)))
    return {
        "mae": float(np.mean(abs_residuals)),
        "rmse": float(np.sqrt(mse)),
        "median_ae": float(np.median(abs_residuals)),
        "bias": float(np.mean(residuals)),
    }


def _constant_predictions(y_true: np.ndarray, value: float) -> np.ndarray:
    return np.full(shape=y_true.shape, fill_value=value, dtype=np.float32)


def _predict_regressor(model: xgb.Booster, features: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(xgb.DMatrix(features)), a_min=0.0, a_max=None)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_df = _load_split(args.train)
    x_val, y_val, val_df = _load_split(args.val)
    x_test, y_test, test_df = _load_split(args.test)

    mean_value = float(np.mean(y_train))
    median_value = float(np.median(y_train))

    mean_results = {
        "train": _metrics(y_train, _constant_predictions(y_train, mean_value)),
        "val": _metrics(y_val, _constant_predictions(y_val, mean_value)),
        "test": _metrics(y_test, _constant_predictions(y_test, mean_value)),
    }
    median_results = {
        "train": _metrics(y_train, _constant_predictions(y_train, median_value)),
        "val": _metrics(y_val, _constant_predictions(y_val, median_value)),
        "test": _metrics(y_test, _constant_predictions(y_test, median_value)),
    }

    # Heuristic baseline: use move_analysis-style rule.
    # We don't have model probabilities here, so approximate move entropy=1.0
    def compute_heuristic_preds(df: pl.DataFrame) -> np.ndarray:
        ply_arr = df["ply"].to_numpy()
        preds = [delay_to_seconds(ply_scaling(int(p)) * 1.0) for p in ply_arr]
        return np.array(preds, dtype=np.float32)

    heur_train_pred = compute_heuristic_preds(train_df)
    heur_val_pred = compute_heuristic_preds(val_df)
    heur_test_pred = compute_heuristic_preds(test_df)

    heuristic_results = {
        "train": _metrics(y_train, heur_train_pred),
        "val": _metrics(y_val, heur_val_pred),
        "test": _metrics(y_test, heur_test_pred),
    }

    train_dmatrix = xgb.DMatrix(x_train, label=y_train)
    val_dmatrix = xgb.DMatrix(x_val, label=y_val)
    test_dmatrix = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "seed": args.random_state,
        "nthread": os.cpu_count() or 1,
        "eval_metric": "rmse",
    }
    model = xgb.train(
        params=params,
        dtrain=train_dmatrix,
        num_boost_round=args.n_estimators,
        evals=[(train_dmatrix, "train"), (val_dmatrix, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    train_pred = _predict_regressor(model, x_train)
    val_pred = _predict_regressor(model, x_val)
    test_pred = _predict_regressor(model, x_test)

    results = {
        "paths": {
            "train": str(args.train),
            "val": str(args.val),
            "test": str(args.test),
        },
        "rows": {
            "train": int(train_df.height),
            "val": int(val_df.height),
            "test": int(test_df.height),
        },
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "baselines": {
            "mean": mean_results,
            "median": median_results,
            "heuristic": heuristic_results,
        },
        "xgboost": {
            "best_iteration": int(model.best_iteration if model.best_iteration is not None else args.n_estimators),
            "train": _metrics(y_train, train_pred),
            "val": _metrics(y_val, val_pred),
            "test": _metrics(y_test, test_pred),
            "params": {
                "max_depth": args.max_depth,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_weight": args.min_child_weight,
                "random_state": args.random_state,
                "n_jobs": os.cpu_count() or 1,
            },
        },
    }

    model_path = args.output_dir / "xgboost_baseline.json"
    metrics_path = args.output_dir / "metrics.json"
    model.save_model(model_path)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()