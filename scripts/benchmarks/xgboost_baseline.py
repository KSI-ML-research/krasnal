#!/usr/bin/env python3
"""Train and evaluate XGBoost baselines on move-time prediction.

The script compares two learned variants when entropy features are available:
    - no_entropy: only clock and game-state features
    - with_entropy: clock/state features plus move_entropy and entropy_x_ply_scaling

It can train either directly on the absolute target or on residuals around the
heuristic baseline. In residual mode the final prediction is:
    heuristic + xgboost_residual

It also reports trivial reference baselines for each variant:
    - mean predictor trained on the train split
    - median predictor trained on the train split

The goal is to make the entropy contribution explicit and easy to compare.
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

BASE_FEATURE_COLUMNS = [
    "ply",
    "time_initial",
    "prev_clock_seconds",
    "clock_fraction_left",
    "is_in_check_before_move",
    "total_pieces",
]
ENTROPY_FEATURE_COLUMNS = ["move_entropy", "entropy_x_ply_scaling"]
TARGET_COLUMN = "target_move_time_seconds"
TARGET_TRANSFORMS = ["none", "log1p"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple XGBoost baseline for move-time prediction."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--target-mode",
        choices=["absolute", "residual"],
        default="residual",
        help="Train XGBoost on the absolute target or on residuals around the heuristic.",
    )
    parser.add_argument(
        "--target-transform",
        choices=TARGET_TRANSFORMS,
        default="log1p",
        help="Transform the target before training and invert it after prediction.",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Use/mark canonical config: residual + sign-log + log1p + max_depth=4 + no regularization",
    )
    return parser.parse_args()


def _load_split(
    path: Path,
    feature_columns: list[str],
    keep_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet split: {path}")

    df = pl.read_parquet(path)
    missing = [column for column in feature_columns + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    extra_columns = [column for column in (keep_columns or []) if column in df.columns]
    selected_columns = list(dict.fromkeys(feature_columns + [TARGET_COLUMN] + extra_columns))
    clean = df.select(selected_columns).drop_nulls()
    if clean.is_empty():
        raise ValueError(f"{path} does not contain any non-null rows")

    features = np.column_stack(
        [clean[column].to_numpy().astype(np.float32, copy=False) for column in feature_columns]
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


def _transform_target(values: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None)).astype(np.float32, copy=False)
    return values.astype(np.float32, copy=False)


def _inverse_transform_target(values: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log1p":
        return np.clip(np.expm1(values), a_min=0.0, a_max=None).astype(np.float32, copy=False)
    return np.clip(values, a_min=0.0, a_max=None).astype(np.float32, copy=False)


def _sign_log_transform(values: np.ndarray) -> np.ndarray:
    # sign(x) * log1p(|x|)
    signs = np.sign(values).astype(np.float32)
    mags = np.log1p(np.abs(values)).astype(np.float32)
    return (signs * mags).astype(np.float32, copy=False)


def _inverse_sign_log_transform(values: np.ndarray) -> np.ndarray:
    # sign(v) * expm1(|v|)
    signs = np.sign(values).astype(np.float32)
    mags = np.expm1(np.abs(values)).astype(np.float32)
    return (signs * mags).astype(np.float32, copy=False)


def _predict_regressor(model: xgb.Booster, features: np.ndarray) -> np.ndarray:
    return model.predict(xgb.DMatrix(features))


def _entropy_feature_checks(clean: pl.DataFrame) -> dict[str, float]:
    if "move_entropy" not in clean.columns:
        return {}

    entropy = clean["move_entropy"].to_numpy().astype(np.float32, copy=False)
    entropy_checks = {
        "min": float(np.min(entropy)),
        "max": float(np.max(entropy)),
        "mean": float(np.mean(entropy)),
        "std": float(np.std(entropy)),
        "fraction_close_to_one": float(np.mean(np.isclose(entropy, 1.0, atol=1e-6, rtol=0.0))),
    }
    if np.allclose(entropy, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("move_entropy is 1.0 everywhere; the probability export looks broken")
    if entropy_checks["std"] == 0.0:
        raise ValueError("move_entropy is constant; the probability export looks broken")
    return entropy_checks


def _heuristic_predictions(df: pl.DataFrame) -> np.ndarray:
    ply_arr = df["ply"].to_numpy()
    if "move_entropy" in df.columns:
        entropy_arr = df["move_entropy"].to_numpy().astype(np.float32, copy=False)
    else:
        entropy_arr = np.ones_like(ply_arr, dtype=np.float32)

    preds = [delay_to_seconds(ply_scaling(int(p)) * float(e)) for p, e in zip(ply_arr, entropy_arr)]
    return np.array(preds, dtype=np.float32)


def _train_variant(
    *,
    variant_name: str,
    feature_columns: list[str],
    train_path: Path,
    val_path: Path,
    test_path: Path,
    args: argparse.Namespace,
    model_path: Path | None = None,
) -> dict[str, object]:
    x_train, y_train, train_df = _load_split(train_path, feature_columns, keep_columns=["move_entropy"])
    x_val, y_val, val_df = _load_split(val_path, feature_columns, keep_columns=["move_entropy"])
    x_test, y_test, test_df = _load_split(test_path, feature_columns, keep_columns=["move_entropy"])

    y_train_transformed = _transform_target(y_train, args.target_transform)
    y_val_transformed = _transform_target(y_val, args.target_transform)
    y_test_transformed = _transform_target(y_test, args.target_transform)

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

    heuristic_results = {
        "train": _metrics(y_train, _heuristic_predictions(train_df)),
        "val": _metrics(y_val, _heuristic_predictions(val_df)),
        "test": _metrics(y_test, _heuristic_predictions(test_df)),
    }

    if args.target_mode == "residual":
        heuristic_train = _heuristic_predictions(train_df)
        heuristic_val = _heuristic_predictions(val_df)
        heuristic_test = _heuristic_predictions(test_df)
        if args.target_transform == "log1p":
            # Use sign-log transform on additive residuals: sgn(y-heur)*log1p(|y-heur|)
            y_train_model = _sign_log_transform(y_train - heuristic_train)
            y_val_model = _sign_log_transform(y_val - heuristic_val)
            y_test_model = _sign_log_transform(y_test - heuristic_test)
        else:
            y_train_model = y_train - heuristic_train
            y_val_model = y_val - heuristic_val
            y_test_model = y_test - heuristic_test
    else:
        y_train_model = y_train_transformed
        y_val_model = y_val_transformed
        y_test_model = y_test_transformed

    train_dmatrix = xgb.DMatrix(x_train, label=y_train_model)
    val_dmatrix = xgb.DMatrix(x_val, label=y_val_model)
    test_dmatrix = xgb.DMatrix(x_test, label=y_test_model)

    params = {
        "objective": "reg:absoluteerror",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "alpha": args.reg_alpha,
        "lambda": args.reg_lambda,
        "seed": args.random_state,
        "nthread": os.cpu_count() or 1,
        "eval_metric": "mae",
    }
    model = xgb.train(
        params=params,
        dtrain=train_dmatrix,
        num_boost_round=args.n_estimators,
        evals=[(train_dmatrix, "train"), (val_dmatrix, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    train_pred_raw = _predict_regressor(model, x_train)
    val_pred_raw = _predict_regressor(model, x_val)
    test_pred_raw = _predict_regressor(model, x_test)

    if args.target_mode == "residual":
        heuristic_train = _heuristic_predictions(train_df)
        heuristic_val = _heuristic_predictions(val_df)
        heuristic_test = _heuristic_predictions(test_df)
        if args.target_transform == "log1p":
            # model predicts sign*log1p(|residual|); invert with sign*expm1(|pred|) then add heuristic
            train_residual_pred = _inverse_sign_log_transform(train_pred_raw)
            val_residual_pred = _inverse_sign_log_transform(val_pred_raw)
            test_residual_pred = _inverse_sign_log_transform(test_pred_raw)
            train_pred = np.clip(heuristic_train + train_residual_pred, a_min=0.0, a_max=None)
            val_pred = np.clip(heuristic_val + val_residual_pred, a_min=0.0, a_max=None)
            test_pred = np.clip(heuristic_test + test_residual_pred, a_min=0.0, a_max=None)
        else:
            train_pred = np.clip(heuristic_train + train_pred_raw, a_min=0.0, a_max=None)
            val_pred = np.clip(heuristic_val + val_pred_raw, a_min=0.0, a_max=None)
            test_pred = np.clip(heuristic_test + test_pred_raw, a_min=0.0, a_max=None)
        residual_fit = {
            "train": _metrics(y_train_model, train_pred_raw),
            "val": _metrics(y_val_model, val_pred_raw),
            "test": _metrics(y_test_model, test_pred_raw),
        }
    else:
        train_pred = _inverse_transform_target(train_pred_raw, args.target_transform)
        val_pred = _inverse_transform_target(val_pred_raw, args.target_transform)
        test_pred = _inverse_transform_target(test_pred_raw, args.target_transform)
        residual_fit = None

    variant_results: dict[str, object] = {
        "variant": variant_name,
        "paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "rows": {
            "train": int(train_df.height),
            "val": int(val_df.height),
            "test": int(test_df.height),
        },
        "features": feature_columns,
        "target": TARGET_COLUMN,
        "target_mode": args.target_mode,
            "target_transform": args.target_transform,
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
            **({"residual_fit": residual_fit} if residual_fit is not None else {}),
            "params": {
                "max_depth": args.max_depth,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_weight": args.min_child_weight,
                "reg_alpha": args.reg_alpha,
                "reg_lambda": args.reg_lambda,
                "random_state": args.random_state,
                "n_jobs": os.cpu_count() or 1,
            },
        },
    }

    entropy_checks = _entropy_feature_checks(train_df)
    if entropy_checks:
        variant_results["entropy_checks"] = entropy_checks

    if model_path is not None:
        model.save_model(model_path)

    return variant_results


def main() -> None:
    args = parse_args()
    # Canonical model configuration used in experiments and reporting.
    # This repository's "primary" model is the residual sign-log variant:
    #   - target_mode: residual
    #   - target_transform: log1p
    #   - residual transform: sign * log1p(|y-heur|) (implemented in code)
    #   - inverse: heur + sign * expm1(|pred|)
    #   - recommended hyperparams: max_depth=4, reg_alpha=0, reg_lambda=0
    CANONICAL_SETTINGS = {
        "target_mode": "residual",
        "target_transform": "log1p",
        "max_depth": 4,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    }
    if getattr(args, "canonical", False):
        print("Applying canonical model overrides:", CANONICAL_SETTINGS)
        args.target_mode = CANONICAL_SETTINGS["target_mode"]
        args.target_transform = CANONICAL_SETTINGS["target_transform"]
        args.max_depth = CANONICAL_SETTINGS["max_depth"]
        args.reg_alpha = CANONICAL_SETTINGS["reg_alpha"]
        args.reg_lambda = CANONICAL_SETTINGS["reg_lambda"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    available_entropy_columns = True
    for path in [args.train, args.val, args.test]:
        df = pl.read_parquet(path, n_rows=1)
        if not all(column in df.columns for column in ENTROPY_FEATURE_COLUMNS):
            available_entropy_columns = False
            break

    variants: dict[str, dict[str, object]] = {
        "no_entropy": _train_variant(
            variant_name="no_entropy",
            feature_columns=BASE_FEATURE_COLUMNS,
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            args=args,
            model_path=args.output_dir / f"xgboost_baseline_{args.target_mode}.json",
        )
    }

    if available_entropy_columns:
        variants["with_entropy"] = _train_variant(
            variant_name="with_entropy",
            feature_columns=BASE_FEATURE_COLUMNS + ENTROPY_FEATURE_COLUMNS,
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            args=args,
            model_path=args.output_dir / f"xgboost_baseline_with_entropy_{args.target_mode}.json",
        )
    else:
        print("Entropy columns not found in all splits; skipping with_entropy variant")

    results = {
        "target_mode": args.target_mode,
        "variants": variants,
    }

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Saved model to {args.output_dir / f'xgboost_baseline_{args.target_mode}.json'}")
    if "with_entropy" in variants:
        print(f"Saved model to {args.output_dir / f'xgboost_baseline_with_entropy_{args.target_mode}.json'}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()