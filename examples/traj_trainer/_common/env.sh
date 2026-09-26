#!/usr/bin/env bash

_TRAJ_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$_TRAJ_COMMON_DIR/../../.." && pwd)}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required to activate environment: $CONDA_ENV" >&2
        exit 1
    fi
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
fi

: "${MODELS_ROOT:?Please set MODELS_ROOT (directory holding base models and checkpoints), e.g. in .env}"
export MODELS_ROOT
CKPT_ROOT="${CKPT_ROOT:-$MODELS_ROOT/ckpt}"
ALFWORLD_BASE_MODEL="${ALFWORLD_BASE_MODEL:-$MODELS_ROOT/Traj-AlfWorld-3B}"
WEBSHOP_BASE_MODEL="${WEBSHOP_BASE_MODEL:-$MODELS_ROOT/Webshop-7B-RL}"
COLLECTION_DIR="${COLLECTION_DIR:-$PROJECT_ROOT/collection}"
COLLECTION_WEBSHOP_DIR="${COLLECTION_WEBSHOP_DIR:-$PROJECT_ROOT/collection_webshop}"

cd "$PROJECT_ROOT"
