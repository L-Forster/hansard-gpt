#!/bin/bash
set -euo pipefail

# End-to-end Hansard pipeline:
# 1. Build shuffled Hansard and Powell parquet corpora
# 2. Train Hansard tokenizer
# 3. Pretrain Hansard base model
# 4. Fine-tune on Powell text

RUN_NAME="${RUN_NAME:-hansard_d12}"
MODEL_TAG="${MODEL_TAG:-d12}"
SFT_RUN_NAME="${SFT_RUN_NAME:-${RUN_NAME}_powell_sft}"

echo "==> Step 1/4: Building Hansard datasets"
python -m nanochat.dataset --hansard

echo "==> Step 2/4: Training Hansard tokenizer"
python -m scripts.tok_train_hansard

echo "==> Step 3/4: Pretraining Hansard base model"
RUN_NAME="${RUN_NAME}" MODEL_TAG="${MODEL_TAG}" bash train_hansard.sh

echo "==> Step 4/4: Powell SFT"
python -m scripts.hansard_sft --run="${SFT_RUN_NAME}" --model_tag="${MODEL_TAG}"

echo "==> Complete"
echo "Base checkpoints: checkpoints/${MODEL_TAG}"
echo "SFT checkpoints: hansard_sft_checkpoints/${MODEL_TAG}"
