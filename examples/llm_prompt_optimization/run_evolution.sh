#!/bin/bash
set -euo pipefail

# -------------------------
# Basic OpenEvolve / queue config
# -------------------------
export OPENAI_API_KEY="dummy"
export OE_QUEUE_ENABLED=true
export OE_QUEUE_INCLUDE_WORST=false
export OE_QUEUE_PATH="stage1_prompt_queues.json"
export OE_QUEUE_K_BEST=5
export OE_QUEUE_K_WORST=5
export OE_QUEUE_MAX_CHARS_EACH=700
export OE_STAGE2_SAVE_DIR="glm_stage2_traces"

# -------------------------
# Reproducibility / evaluator config
# -------------------------
export PYTHONHASHSEED="${PYTHONHASHSEED:-1234}"
export OPENEVOLVE_SEED="${OPENEVOLVE_SEED:-1234}"
export OPENEVOLVE_DETERMINISTIC="${OPENEVOLVE_DETERMINISTIC:-1}"

# generation mode
export OPENEVOLVE_GREEDY="${OPENEVOLVE_GREEDY:-1}"
export OPENEVOLVE_USE_CHAT_TEMPLATE="${OPENEVOLVE_USE_CHAT_TEMPLATE:-1}"
export OPENEVOLVE_ENABLE_THINKING="${OPENEVOLVE_ENABLE_THINKING:-0}"
export OPENEVOLVE_STRIP_THINKING="${OPENEVOLVE_STRIP_THINKING:-1}"
export OPENEVOLVE_MAX_NEW_TOKENS="${OPENEVOLVE_MAX_NEW_TOKENS:-512}"
export OPENEVOLVE_LOAD_IN_4BIT="${OPENEVOLVE_LOAD_IN_4BIT:-0}"
export OPENEVOLVE_MODEL_DTYPE="${OPENEVOLVE_MODEL_DTYPE:-bfloat16}"

# model loading
export OPENEVOLVE_MODEL_PATH="${OPENEVOLVE_MODEL_PATH:-../../../gsm8k/zai-org/GLM-4-9B-0414-wanda-s065}"


if [ $# -lt 1 ]; then
    echo "Usage: $0 <prompt_file> [additional_args...]"
    echo "Example: $0 gsm8k_prompt.txt --iterations 80"
    exit 1
fi

PROMPT_FILE="$1"
shift

export OPENEVOLVE_PROMPT="$PROMPT_FILE"

echo "[info] prompt file: $OPENEVOLVE_PROMPT"
echo "[info] model path: $OPENEVOLVE_MODEL_PATH"
echo "[info] seed: $OPENEVOLVE_SEED"
echo "[info] deterministic: $OPENEVOLVE_DETERMINISTIC"
echo "[info] greedy: $OPENEVOLVE_GREEDY"
echo "[info] thinking: $OPENEVOLVE_ENABLE_THINKING"
echo "[info] max_new_tokens: $OPENEVOLVE_MAX_NEW_TOKENS"

python ../../openevolve-run.py \
    "$PROMPT_FILE" \
    evaluator.py \
    --config config.yaml \
    --output glm_output \
    "$@"