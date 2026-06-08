"""Compares metrics from quick local XGBoost training with reference artifact and prints out diffs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from krasnal.move_time.xgboost import (
    BASE_FEATURE_COLUMNS,
    ENTROPY_FEATURE_COLUMNS,
    train_variant,
)

DEFAULT_DATA_DIR = Path("data/3_xgboost_300_probs_v4_stratified")
DEFAULT_REFERENCE_METRICS = Path("artifacts/xgboost_baseline_residual_log1p_md4_noreg/metrics.json")


def _load_reference_metrics(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _diff_metric_block(current: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: float(current[key] - reference[key]) for key in sorted(reference)}


def _print_metric_diffs(current: dict[str, object], reference: dict[str, object]) -> None:
    for section in ["baselines", "xgboost", "entropy_checks"]:
        if section not in current or section not in reference:
            continue

        print(f"[{section}]")
        if section == "entropy_checks":
            diffs = _diff_metric_block(current[section], reference[section])
            print(json.dumps(diffs, indent=2, sort_keys=True))
            continue

        for subkey in sorted(reference[section]):
            current_block = current[section][subkey]
            reference_block = reference[section][subkey]
            if isinstance(reference_block, dict) and all(
                isinstance(v, dict) for v in reference_block.values()
            ):
                print(f"  {subkey}")
                for split in ["train", "val", "test"]:
                    if split not in reference_block:
                        continue
                    diffs = _diff_metric_block(current_block[split], reference_block[split])
                    print(f"    {split}: {json.dumps(diffs, sort_keys=True)}")
            elif isinstance(reference_block, dict):
                diffs = _diff_metric_block(current_block, reference_block)
                print(f"  {subkey}: {json.dumps(diffs, sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current XGBoost run with a reference artifact."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reference-metrics", type=Path, default=DEFAULT_REFERENCE_METRICS)
    args = parser.parse_args()

    reference = _load_reference_metrics(args.reference_metrics)

    train_path = args.data_dir / "xgb_train.parquet"
    val_path = args.data_dir / "xgb_val.parquet"
    test_path = args.data_dir / "xgb_test.parquet"

    with tempfile.TemporaryDirectory(prefix="xgb-compare-") as tmp_dir:
        temp_dir = Path(tmp_dir)
        run_args = argparse.Namespace(
            target_mode="residual",
            target_transform="log1p",
            max_depth=4,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            output_dir=temp_dir,
        )

        current = train_variant(
            variant_name="with_entropy",
            feature_columns=BASE_FEATURE_COLUMNS + ENTROPY_FEATURE_COLUMNS,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            args=run_args,
            model_path=temp_dir / "model.json",
        )

    reference_variant = reference["variants"]["with_entropy"]
    print("Comparing current run against reference variant: with_entropy")
    print(f"Reference artifact: {args.reference_metrics}")
    _print_metric_diffs(current, reference_variant)


if __name__ == "__main__":
    main()
