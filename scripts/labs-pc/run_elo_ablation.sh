#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO=${SOURCE_REPO:-/Ziob/ijakus/krasnal}
WORKTREE_BASE=${WORKTREE_BASE:-/Ziob/ijakus}
REMOTE_SOURCE_REPO=${REMOTE_SOURCE_REPO:-/ziob/ijakus/krasnal}
REMOTE_WORKTREE_BASE=${REMOTE_WORKTREE_BASE:-/ziob/ijakus}
TARGET_GAMES=${TARGET_GAMES:-1000000}
MODEL=${MODEL:-medium}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-1}
EPOCHS=${EPOCHS:-1.0}
UV_CACHE_DIR=${UV_CACHE_DIR:-/ziob/ijakus/.cache/uv}
HF_HOME=${HF_HOME:-/ziob/ijakus/.cache/huggingface}
TOKENIZED_BASE=${TOKENIZED_BASE:-data/2_tokenized_labs}
ARTIFACT_BASE=${ARTIFACT_BASE:-artifacts/pretrain/labs}

LAB01=192.168.4.102
LAB02=192.168.4.103
LAB03=192.168.4.104
LAB04=192.168.4.105
LAB05=192.168.4.106
LAB06=192.168.4.107
LAB07=192.168.4.108
LAB08=192.168.4.109
LAB09=192.168.4.110
LAB10=192.168.4.111
LAB11=192.168.4.112
LAB12=192.168.4.113
LAB13=192.168.4.114
LAB14=192.168.4.115
LAB15=192.168.4.116
LAB16=192.168.4.117

dispatch() {
    mkdir -p output/labs-pc
    start_variant "$LAB06" krasnal-baseline labs-baseline elo_tokens true
    start_variant "$LAB07" krasnal-no-elo-token labs-no-elo-token no_elo_tokens false
}

start_variant() {
    local host=$1 worktree_name=$2 branch=$3 variant=$4 include_elo=$5
    local worktree="$WORKTREE_BASE/$worktree_name"
    local remote_worktree="$REMOTE_WORKTREE_BASE/$worktree_name"
    ensure_worktree "$worktree" "$branch"

    local log="output/labs-pc/${variant}_$(date +%Y%m%d_%H%M%S).log"
    local remote_log="$remote_worktree/$log"
    local cmd
    printf -v cmd \
        'cd %q && mkdir -p output/labs-pc && nohup env TARGET_GAMES=%q MODEL=%q BATCH_SIZE=%q NUM_WORKERS=%q EPOCHS=%q UV_CACHE_DIR=%q HF_HOME=%q TOKENIZED_BASE=%q ARTIFACT_BASE=%q bash %q worker %q %q > %q 2>&1 < /dev/null & echo $!' \
        "$remote_worktree" "$TARGET_GAMES" "$MODEL" "$BATCH_SIZE" "$NUM_WORKERS" \
        "$EPOCHS" "$UV_CACHE_DIR" "$HF_HOME" "$TOKENIZED_BASE" "$ARTIFACT_BASE" \
        "$REMOTE_SOURCE_REPO/scripts/labs-pc/run_elo_ablation.sh" "$variant" "$include_elo" "$remote_log"
    echo "Starting $variant on $host in $remote_worktree"
    ssh "$host" "$cmd"
    echo "  log: $host:$remote_log"
}

ensure_worktree() {
    local worktree=$1 branch=$2
    if [ -d "$worktree/.git" ] || [ -f "$worktree/.git" ]; then
        return
    fi
    if git -C "$SOURCE_REPO" show-ref --verify --quiet "refs/heads/$branch"; then
        git -C "$SOURCE_REPO" worktree add "$worktree" "$branch"
    else
        git -C "$SOURCE_REPO" worktree add -b "$branch" "$worktree" dev
    fi
}

worker() {
    local variant=$1 include_elo=$2
    local tokenized_dir="$TOKENIZED_BASE/$variant"
    local artifact_dir="$ARTIFACT_BASE/$variant"

    source ~/.bashrc
    mkdir -p "$UV_CACHE_DIR" "$HF_HOME" output

    export UV_CACHE_DIR HF_HOME
    export UV_LINK_MODE=copy
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export XDG_RUNTIME_DIR=/tmp/ijakus/runtime
    export POLARS_MAX_THREADS=2
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export KRASNAL_TOKENIZED_DIR="$tokenized_dir"
    export KRASNAL_ARTIFACT_DIR="$artifact_dir"
    export WANDB_RUN_GROUP="labs-elo-token-ablation-${TARGET_GAMES}"
    export WANDB_NAME="labs-${variant}"
    mkdir -p "$XDG_RUNTIME_DIR"

    echo "host=$(hostname)"
    echo "variant=$variant include_elo=$include_elo target_games=$TARGET_GAMES"
    echo "tokenized_dir=$KRASNAL_TOKENIZED_DIR"
    echo "artifact_dir=$KRASNAL_ARTIFACT_DIR"

    uv sync

    rm -rf data/1_filtered "$tokenized_dir" "$artifact_dir"

    uv run scripts/data/download_games.py \
        target_games="$TARGET_GAMES" \
        require_evals=false

    uv run scripts/data/preprocess.py \
        target_games="$TARGET_GAMES" \
        include_elo="$include_elo" \
        preprocess_workers=1 \
        pack_stream_batch_size=5000 \
        report.enabled=false

    uv run torchrun --standalone --nproc_per_node=1 scripts/training/pretrain.py \
        model="$MODEL" \
        train=cuda \
        train.epochs="$EPOCHS" \
        train.batch_size="$BATCH_SIZE" \
        train.num_workers="$NUM_WORKERS" \
        train.pin_memory=false \
        train.compile=true \
        eval.inference_batch_size="$BATCH_SIZE" \
        seed=42 \
        include_elo="$include_elo"
}

case "${1:-dispatch}" in
    dispatch) dispatch ;;
    worker) shift; worker "$@" ;;
    *)
        echo "Usage: $0 [dispatch|worker VARIANT INCLUDE_ELO]" >&2
        exit 2
        ;;
esac
