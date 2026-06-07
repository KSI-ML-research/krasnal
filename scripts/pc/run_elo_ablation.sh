#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

TARGET_GAMES=${TARGET_GAMES:-1000000}
MODEL=${MODEL:-medium}

run_variant() {
    local variant=$1 include_elo=$2

    export KRASNAL_FILTERED_DIR="data/$variant/1_filtered"
    export KRASNAL_TOKENIZED_DIR="data/$variant/2_tokenized"
    export KRASNAL_ARTIFACT_DIR="artifacts/pretrain/$variant"
    export WANDB_RUN_GROUP="elo-token-ablation-${TARGET_GAMES}"
    export WANDB_NAME="$variant"

    echo "variant=$variant include_elo=$include_elo target_games=$TARGET_GAMES"
    echo "filtered_dir=$KRASNAL_FILTERED_DIR"
    echo "tokenized_dir=$KRASNAL_TOKENIZED_DIR"
    echo "artifact_dir=$KRASNAL_ARTIFACT_DIR"

    uv run scripts/data/download_games.py \
        target_games="$TARGET_GAMES" \
        require_evals=false

    uv run scripts/data/preprocess.py \
        target_games="$TARGET_GAMES" \
        include_elo="$include_elo" \
        report.enabled=false

    uv run scripts/training/pretrain.py \
        model="$MODEL" \
        train=cuda \
        seed=42 \
        include_elo="$include_elo"
}

uv sync

run_variant baseline true
run_variant no_elo_token false
