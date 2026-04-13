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

  if [ "${NANOCHAT_ENV_READY:-0}" != "1" ]; then
    uv sync --extra "${UV_EXTRA}"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export NANOCHAT_ENV_READY=1
  fi
}

ensure_python_env

# Optimized training config for Hansard dataset on 80GB A100
# One-pass pretraining over the shuffled Hansard train split.

RUN_NAME="${RUN_NAME:-hansard_d12}"
MODEL_TAG="${MODEL_TAG:-d12}"
DEPTH="${DEPTH:-12}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"

echo "Counting actual train tokens from shuffled Hansard shards..."
TRAIN_TOKENS=$(python -m scripts.count_tokens --split train | awk '/Total tokens:/ {gsub(/,/, "", $3); print $3}')
if [ -z "${TRAIN_TOKENS}" ]; then
  echo "Failed to determine train token count."
  exit 1
fi

SUPERVISED_TOKENS=$(( TRAIN_TOKENS - 1 ))
if [ "${SUPERVISED_TOKENS}" -lt 1 ]; then
  SUPERVISED_TOKENS=1
fi

NUM_ITERATIONS=$(( SUPERVISED_TOKENS / TOTAL_BATCH_SIZE ))
if [ "${NUM_ITERATIONS}" -lt 1 ]; then
  NUM_ITERATIONS=1
fi

echo "Train tokens: ${TRAIN_TOKENS}"
echo "Supervised next-token targets: ${SUPERVISED_TOKENS}"
echo "Total batch size: ${TOTAL_BATCH_SIZE}"
echo "One-pass num_iterations: ${NUM_ITERATIONS}"

python -m scripts.base_train \
  --run="${RUN_NAME}" \
  --model_tag="${MODEL_TAG}" \
  --tokenizer_name="hansard" \
  --depth="${DEPTH}" \
  --max_seq_len="${MAX_SEQ_LEN}" \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --total_batch_size="${TOTAL_BATCH_SIZE}" \
  --num_iterations="${NUM_ITERATIONS}" \
  --embedding_lr=0.2 \
  --unembedding_lr=0.004 \
  --matrix_lr=0.02 \
  --weight_decay=0.0 \
  --grad_clip=1.0 \
  --warmup_ratio=0.05 \
  --warmdown_ratio=0.2 \
  --final_lr_frac=0.0 \
  --eval_every=250 \
  --eval_tokens=10485760 \
  --core_metric_every=1000 \
  --sample_every=1000 \
  --save_every=2000

# Model architecture breakdown (depth=12):
# - Layers: 12
# - Dimension: 12 * 64 = 768
# - Heads: (768 + 127) // 128 = 6
# - Parameters: ~125M-class
#
# Training horizon:
# - Train tokens are counted from the actual shuffled train split
# - supervised_tokens = train_tokens - 1
# - num_iterations = floor(supervised_tokens / total_batch_size)
# - This gives one pass over the train split with no intentional repetition
