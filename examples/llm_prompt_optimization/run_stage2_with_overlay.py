import os

os.environ["OE_STAGE2_SAVE_DIR"] = "stage2_traces_gpt"
os.environ["OPENEVOLVE_PROMPT"] = "gsm8k_prompt.txt"

os.environ["OPENEVOLVE_MODEL_PATH"] = "openai/gpt-oss-20b"

os.environ["OPENEVOLVE_OVERLAY_PATH"] = "../../../gsm8k/overlays/gpt-oss-20b-wanda-s050.pt"

os.environ["OPENEVOLVE_SEED"] = "42"
os.environ["OPENEVOLVE_DETERMINISTIC"] = "1"

os.environ["OPENEVOLVE_GREEDY"] = "1"
os.environ["OPENEVOLVE_USE_CHAT_TEMPLATE"] = "1"
os.environ["OPENEVOLVE_ENABLE_THINKING"] = "0"
os.environ["OPENEVOLVE_STRIP_THINKING"] = "1"
os.environ["OPENEVOLVE_MAX_NEW_TOKENS"] = "512"

os.environ["OPENEVOLVE_USE_PROCESSOR_MODEL"] = "1"

os.environ["OPENEVOLVE_LOAD_IN_4BIT"] = "0"
os.environ["OPENEVOLVE_MODEL_DTYPE"] = "bfloat16"

from overlay_utils import load_overlay, summarize_overlay_sparsity
import evaluator

overlay_path = os.environ.get("OPENEVOLVE_OVERLAY_PATH", "").strip()
if overlay_path:
    info = load_overlay(
        evaluator.target_model,
        overlay_path=overlay_path,
        strict=True,
    )
    print("[ok] overlay applied")
    print({
        "loaded": info["loaded"],
        "metadata": info["metadata"],
    })
    summarize_overlay_sparsity(evaluator.target_model)
else:
    print("[warn] OPENEVOLVE_OVERLAY_PATH not set; running base model without overlay")

res = evaluator.evaluate_stage2("gsm8k_prompt.txt")
print(res)