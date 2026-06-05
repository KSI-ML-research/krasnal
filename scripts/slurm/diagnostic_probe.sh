#!/bin/bash
set -euo pipefail

CLUSTER_USER=${KRASNAL_CLUSTER_USER:-ijakus}
USER_DIRECTORY=/Ziob/$CLUSTER_USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal

source ~/.bashrc
cd "$PROJECT_ROOT"

mkdir -p output artifacts/diagnostics

export XDG_RUNTIME_DIR=/tmp/$CLUSTER_USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=$USER_DIRECTORY/.cache/huggingface
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$XDG_RUNTIME_DIR" "$UV_CACHE_DIR" "$HF_HOME"

test -d .venv || uv venv .venv --python 3.13
uv sync

if [ -z "${PROBE_SCRIPT:-}" ] || [ -z "${PROBE_ARTIFACT_DIRS:-}" ]; then
    echo "PROBE_SCRIPT and PROBE_ARTIFACT_DIRS must be exported by the scheduler." >&2
    exit 1
fi

artifact_args=()
for artifact_dir in $PROBE_ARTIFACT_DIRS; do
    artifact_args+=(--artifact-dir "$artifact_dir")
done

json_args=()
if [ -n "${PROBE_JSON_OUT:-}" ]; then
    json_args+=(--json-out "$PROBE_JSON_OUT")
fi

eval_args=()
if [ -n "${PROBE_EVAL_PARQUET:-}" ]; then
    eval_args+=(--eval-parquet "$PROBE_EVAL_PARQUET")
fi

wandb_args=()
if [ "${PROBE_WANDB:-false}" = "true" ]; then
    wandb_args+=(--wandb)
    if [ -n "${PROBE_WANDB_NAME:-}" ]; then
        wandb_args+=(--wandb-name "$PROBE_WANDB_NAME")
    fi
    if [ -n "${PROBE_WANDB_GROUP:-}" ]; then
        wandb_args+=(--wandb-group "$PROBE_WANDB_GROUP")
    fi
fi

echo "Running diagnostic probe: $PROBE_SCRIPT"
echo "Artifacts: $PROBE_ARTIFACT_DIRS"
echo "Eval parquet: ${PROBE_EVAL_PARQUET:-default}"
echo "Extra args: ${PROBE_EXTRA_ARGS:-none}"

uv run "$PROBE_SCRIPT" \
    "${artifact_args[@]}" \
    "${json_args[@]}" \
    "${wandb_args[@]}" \
    "${eval_args[@]}" \
    ${PROBE_EXTRA_ARGS:-}
