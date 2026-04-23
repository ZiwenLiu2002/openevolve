import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from evaluator import evaluate_stage2

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_stage2_only.py <prompt_file>")
        sys.exit(1)

    prompt_path = sys.argv[1]

    stage2_dir = os.environ.get("OE_STAGE2_SAVE_DIR", "")
    if not stage2_dir:
        raise RuntimeError("OE_STAGE2_SAVE_DIR is not set")

    print("============================================================")
    print("STAGE-2 ONLY MODE")
    print(f"Prompt file: {prompt_path}")
    print(f"Saving traces to: {stage2_dir}")
    print("============================================================")

    metrics = evaluate_stage2(prompt_path)

    print("\nFinal Stage-2 Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
