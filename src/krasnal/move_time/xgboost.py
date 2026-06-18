"""XGBoost trainer and predictor for move-time estimation.

Trains an XGBoost regressor on absolute move times using clock-derived
features.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from krasnal.inference.move_analysis import delay_to_seconds, ply_scaling

DEFAULT_TRAIN_PATH = Path("data/3_xgboost/xgb_train.parquet")
DEFAULT_VAL_PATH = Path("data/3_xgboost/xgb_val.parquet")
DEFAULT_TEST_PATH = Path("data/3_xgboost/xgb_test.parquet")
DEFAULT_OUTPUT_DIR = Path("artifacts/xgboost_baseline")

FEATURE_COLUMNS = [
    "ply",
    "time_initial",
    "prev_clock_seconds",
    "clock_fraction_left",
    "is_in_check_before_move",
    "total_pieces",
]
TARGET_COLUMN = "target_move_time_seconds"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train XGBoost regressor for absolute move-time prediction."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet split: {path}")

    df = pl.read_parquet(path)
    missing = [col for col in [*FEATURE_COLUMNS, TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    clean = df.select([*FEATURE_COLUMNS, TARGET_COLUMN]).drop_nulls()
    if clean.is_empty():
        raise ValueError(f"{path} does not contain any non-null rows")

    features = np.column_stack(
        [clean[col].to_numpy().astype(np.float32, copy=False) for col in FEATURE_COLUMNS]
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


def _heuristic_predictions(df: pl.DataFrame) -> np.ndarray:
    ply_arr = df["ply"].to_numpy()
    raw = np.array([ply_scaling(int(p)) for p in ply_arr], dtype=np.float32)
    return np.array([delay_to_seconds(float(r)) for r in raw], dtype=np.float32)


def _build_xgb_params(args: argparse.Namespace) -> dict[str, object]:
    return {
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


def train(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    args: argparse.Namespace,
    model_path: Path | None = None,
) -> dict[str, object]:
    x_train, y_train, train_df = _load_split(train_path)
    x_val, y_val, val_df = _load_split(val_path)
    x_test, y_test, test_df = _load_split(test_path)

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

    train_dmatrix = xgb.DMatrix(x_train, label=y_train)
    val_dmatrix = xgb.DMatrix(x_val, label=y_val)

    model = xgb.train(
        params=_build_xgb_params(args),
        dtrain=train_dmatrix,
        num_boost_round=args.n_estimators,
        evals=[(train_dmatrix, "train"), (val_dmatrix, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    train_pred = model.predict(xgb.DMatrix(x_train))
    val_pred = model.predict(xgb.DMatrix(x_val))
    test_pred = model.predict(xgb.DMatrix(x_test))

    results: dict[str, object] = {
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
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
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

    if model_path is not None:
        model.save_model(model_path)

    return results


def predict_single(
    model: xgb.Booster,
    ply: int,
    time_initial: int,
    prev_clock_seconds: int,
    clock_fraction_left: float,
    is_in_check_before_move: bool,
    total_pieces: int,
) -> float:
    """Predict move time in seconds for a single position using a loaded model."""
    features = np.array(
        [
            [
                ply,
                time_initial,
                prev_clock_seconds,
                clock_fraction_left,
                int(is_in_check_before_move),
                total_pieces,
            ]
        ],
        dtype=np.float32,
    )
    return float(model.predict(xgb.DMatrix(features))[0])


def _feature_frame(df: pl.DataFrame) -> np.ndarray:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return np.column_stack(
        [df[col].to_numpy().astype(np.float32, copy=False) for col in FEATURE_COLUMNS]
    )


def predict_frame(
    *,
    model_path: Path,
    frame: pl.DataFrame,
) -> np.ndarray:
    clean = frame.drop_nulls(subset=FEATURE_COLUMNS)
    model = xgb.Booster()
    model.load_model(model_path)
    features = _feature_frame(clean)
    return model.predict(xgb.DMatrix(features))


def predict_parquet(
    *,
    model_path: Path,
    input_path: Path,
    output_path: Path,
) -> Path:
    frame = pl.read_parquet(input_path)
    predictions = predict_frame(model_path=model_path, frame=frame)
    result = frame.with_columns(pl.Series("predicted_move_time_seconds", predictions))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return output_path


def main() -> None:
    args = build_parser().parse_args()

    base_output_dir = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_output_dir / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = run_dir

    model_path = args.output_dir / "xgboost_baseline.json"
    results = train(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        args=args,
        model_path=model_path,
    )

    metrics_path = args.output_dir / "metrics.json"
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
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")

    latest_link = base_output_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_symlink() or latest_link.is_file():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)
        os.symlink(str(run_dir.resolve()), str(latest_link))
    except Exception:
        with contextlib.suppress(Exception):
            shutil.copy2(metrics_path, base_output_dir / "latest_metrics.json")


if __name__ == "__main__":
    main()
