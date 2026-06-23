"""CLI wrapper that runs XGBoost prediction on move-lvl parquets"""

from __future__ import annotations

import argparse
from pathlib import Path

from krasnal.move_time.xgboost import predict_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an XGBoost move-time prediction job.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_path = predict_parquet(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
    )
    print(output_path)


if __name__ == "__main__":
    main()
