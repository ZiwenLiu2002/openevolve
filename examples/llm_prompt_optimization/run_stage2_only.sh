#!/bin/bash
# Stage-2 only runner (no evolution), environment-aligned with run_evolution.sh

export OPENAI_API_KEY="dummy"
export OPENEVOLVE_PROMPT="gsm8k_prompt.txt"
export OE_QUEUE_ENABLED=true
export OE_QUEUE_INCLUDE_WORST=false
export OE_QUEUE_PATH="stage1_prompt_queues.json"
export OE_QUEUE_K_BEST=5
export OE_QUEUE_K_WORST=5
export OE_QUEUE_MAX_CHARS_EACH=700
export OE_STAGE2_SAVE_DIR="stage2_traces"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <prompt_file>"
    exit 1
fi

PROMPT_FILE=$1
export OPENEVOLVE_PROMPT=$PROMPT_FILE

echo "============================================================"
echo "Running STAGE-2 ONLY evaluation (no evolution)"
echo "Prompt: $PROMPT_FILE"
echo "Stage2 traces dir: $OE_STAGE2_SAVE_DIR"
echo "============================================================"

python run_stage2_only.py "$PROMPT_FILE"
