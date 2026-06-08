"""XGBoost baseline trainer and predictor for move-time estimation.

Provides utilities to train/evaluate XGBoost variants (with or without
entropy features), predict on parquet files, and produce timestamped
artifact directories with metrics and saved models.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from krasnal.inference.move_analysis import delay_to_seconds, ply_scaling

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
CANONICAL_SETTINGS = {
    "target_mode": "residual",
    "target_transform": "log1p",
    "max_depth": 4,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
}


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--no-entropy-ablation",
        action="store_true",
        help="Also train the no-entropy ablation for comparison.",
    )
    return parser


def _load_split(
    path: Path,
    feature_columns: list[str],
    keep_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet split: {path}")

    df = pl.read_parquet(path)
    missing = [column for column in [*feature_columns, TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    extra_columns = [column for column in (keep_columns or []) if column in df.columns]
    selected_columns = list(dict.fromkeys([*feature_columns, TARGET_COLUMN, *extra_columns]))
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
    signs = np.sign(values).astype(np.float32)
    mags = np.log1p(np.abs(values)).astype(np.float32)
    return (signs * mags).astype(np.float32, copy=False)


def _inverse_sign_log_transform(values: np.ndarray) -> np.ndarray:
    signs = np.sign(values).astype(np.float32)
    mags = np.expm1(np.abs(values)).astype(np.float32)
    return (signs * mags).astype(np.float32, copy=False)


def _apply_canonical_overrides(args: argparse.Namespace) -> None:
    for key, value in CANONICAL_SETTINGS.items():
        setattr(args, key, value)


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

    preds = [
        delay_to_seconds(ply_scaling(int(p)) * float(e))
        for p, e in zip(ply_arr, entropy_arr, strict=False)
    ]
    return np.array(preds, dtype=np.float32)


def train_variant(
    *,
    variant_name: str,
    feature_columns: list[str],
    train_path: Path,
    val_path: Path,
    test_path: Path,
    args: argparse.Namespace,
    model_path: Path | None = None,
) -> dict[str, object]:
    x_train, y_train, train_df = _load_split(
        train_path, feature_columns, keep_columns=["move_entropy"]
    )
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
    xgb.DMatrix(x_test, label=y_test_model)

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
            "best_iteration": int(
                model.best_iteration if model.best_iteration is not None else args.n_estimators
            ),
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


def _feature_frame(df: pl.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return np.column_stack(
        [df[column].to_numpy().astype(np.float32, copy=False) for column in feature_columns]
    )


def predict_frame(
    *,
    model_path: Path,
    frame: pl.DataFrame,
    feature_columns: list[str],
    target_mode: str = "residual",
    target_transform: str = "log1p",
) -> np.ndarray:
    clean = frame.drop_nulls(subset=feature_columns)
    model = xgb.Booster()
    model.load_model(model_path)
    features = _feature_frame(clean, feature_columns)
    raw_pred = _predict_regressor(model, features)

    if target_mode == "absolute":
        return _inverse_transform_target(raw_pred, target_transform)

    heuristic = _heuristic_predictions(clean)
    residual = _inverse_sign_log_transform(raw_pred) if target_transform == "log1p" else raw_pred
    return np.clip(heuristic + residual, a_min=0.0, a_max=None)


def predict_parquet(
    *,
    model_path: Path,
    input_path: Path,
    output_path: Path,
    feature_columns: list[str],
    target_mode: str = "residual",
    target_transform: str = "log1p",
) -> Path:
    frame = pl.read_parquet(input_path)
    predictions = predict_frame(
        model_path=model_path,
        frame=frame,
        feature_columns=feature_columns,
        target_mode=target_mode,
        target_transform=target_transform,
    )
    clean = frame.drop_nulls(subset=feature_columns)
    result = clean.with_columns(pl.Series("predicted_move_time_seconds", predictions))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "canonical", False):
        print("Applying canonical model overrides:", CANONICAL_SETTINGS)
        _apply_canonical_overrides(args)
    # Ensure base output dir exists, then create a timestamped run directory
    base_output_dir = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_output_dir / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    # Use the run-specific dir for all saved artifacts for this invocation
    args.output_dir = run_dir

    canonical_feature_columns = BASE_FEATURE_COLUMNS + ENTROPY_FEATURE_COLUMNS
    canonical_model_path = (
        args.output_dir / f"xgboost_baseline_with_entropy_{args.target_mode}.json"
    )
    variants: dict[str, dict[str, object]] = {
        "with_entropy": train_variant(
            variant_name="with_entropy",
            feature_columns=canonical_feature_columns,
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            args=args,
            model_path=canonical_model_path,
        )
    }

    if args.no_entropy_ablation:
        variants["no_entropy"] = train_variant(
            variant_name="no_entropy",
            feature_columns=BASE_FEATURE_COLUMNS,
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            args=args,
            model_path=args.output_dir / f"xgboost_baseline_no_entropy_{args.target_mode}.json",
        )

    results = {
        "target_mode": args.target_mode,
        "variants": variants,
    }

    metrics_path = args.output_dir / "metrics.json"
    # Add run metadata (timestamp, git sha, cli args) to top-level of metrics
    git_sha = None
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_sha = None

    cli_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    results_with_meta = {
        "run_timestamp": run_ts,
        "git_sha": git_sha,
        "cli_args": cli_args,
        **results,
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results_with_meta, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Saved model to {canonical_model_path}")
    if "no_entropy" in variants:
        print(
            f"Saved model to {args.output_dir / f'xgboost_baseline_no_entropy_{args.target_mode}.json'}"
        )
    print(f"Saved metrics to {metrics_path}")

    # Update (or create) a stable `latest` symlink in the base output dir
    latest_link = base_output_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_symlink() or latest_link.is_file():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)
        # Create symlink pointing to the run directory (absolute path)
        os.symlink(str(run_dir.resolve()), str(latest_link))
    except Exception:
        # If symlink fails (e.g., on filesystems that disallow), fall back to copying metrics.json
        with contextlib.suppress(Exception):
            shutil.copy2(metrics_path, base_output_dir / "latest_metrics.json")


if __name__ == "__main__":
    main()
