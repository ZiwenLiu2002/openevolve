import os

from overlay_utils import load_overlay, summarize_overlay_sparsity
import evaluator as _base

overlay_path = os.environ.get("OPENEVOLVE_OVERLAY_PATH", "").strip()

if overlay_path:
    info = load_overlay(
        _base.target_model,
        overlay_path=overlay_path,
        strict=True,
    )
    print("[ok] overlay applied inside evaluator_with_overlay")
    print({
        "loaded": info["loaded"],
        "metadata": info["metadata"],
    })
    summarize_overlay_sparsity(_base.target_model)
else:
    print("[warn] OPENEVOLVE_OVERLAY_PATH not set; using base model only")

# re-export
evaluate_stage1 = _base.evaluate_stage1
evaluate_stage2 = _base.evaluate_stage2
evaluate = _base.evaluate