"""CLI wrapper that runs XGBoost prediction on move-lvl parquets"""

from __future__ import annotations

import argparse
from pathlib import Path

from krasnal.move_time.xgboost import BASE_FEATURE_COLUMNS, ENTROPY_FEATURE_COLUMNS, predict_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an XGBoost move-time prediction job.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-entropy", action="store_true")
    parser.add_argument("--target-mode", choices=["absolute", "residual"], default="residual")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="log1p")
    args = parser.parse_args()

    feature_columns = (
        BASE_FEATURE_COLUMNS + ENTROPY_FEATURE_COLUMNS
        if args.with_entropy
        else BASE_FEATURE_COLUMNS
    )
    output_path = predict_parquet(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        feature_columns=feature_columns,
        target_mode=args.target_mode,
        target_transform=args.target_transform,
    )
    print(output_path)


if __name__ == "__main__":
    main()
