#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

UV_EXTRA="${UV_EXTRA:-gpu}"

ensure_python_env() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi

  if [ ! -d ".venv" ]; then
    uv venv
  fi

  uv sync --extra "${UV_EXTRA}"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export NANOCHAT_ENV_READY=1
}

ensure_python_env

# End-to-end Hansard pipeline:
# 1. Build shuffled Hansard and Powell parquet corpora
# 2. Train Hansard tokenizer
# 3. Pretrain Hansard base model
# 4. Fine-tune on Powell text

RUN_NAME="${RUN_NAME:-hansard_d12}"
MODEL_TAG="${MODEL_TAG:-d12}"
SFT_RUN_NAME="${SFT_RUN_NAME:-${RUN_NAME}_powell_sft}"
TOKENIZER_DIR="data/tokenizer_hansard"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"

echo "==> Step 1/4: Building Hansard datasets"
python -m nanochat.dataset --hansard

if [ -f "${TOKENIZER_DIR}/tokenizer.pkl" ] && [ -f "${TOKENIZER_DIR}/token_bytes.pt" ] && [ -f "${TOKENIZER_DIR}/vocab.txt" ]; then
  echo "==> Step 2/4: Using existing Hansard tokenizer in ${TOKENIZER_DIR}"
else
  echo "==> Step 2/4: Training Hansard tokenizer"
  python -m scripts.tok_train_hansard
fi

echo "==> Step 3/4: Pretraining Hansard base model"
RUN_NAME="${RUN_NAME}" MODEL_TAG="${MODEL_TAG}" DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE}" bash train_hansard.sh

echo "==> Step 4/4: Powell SFT"
python -m scripts.hansard_sft --run="${SFT_RUN_NAME}" --model_tag="${MODEL_TAG}"

echo "==> Complete"
echo "Base checkpoints: checkpoints/${MODEL_TAG}"
echo "SFT checkpoints: hansard_sft_checkpoints/${MODEL_TAG}"
