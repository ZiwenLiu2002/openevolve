import json
import re
from pathlib import Path

import torch


def _extract_layer_id(name: str) -> int:
    m = re.search(r"model\.layers\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return 10**9


def tensor_sparsity(x: torch.Tensor) -> float:
    return float((x == 0).sum().item()) / float(x.numel())


def find_layerwise_params(model, suffix):
    matches = []
    for name, p in model.named_parameters():
        if name.endswith(suffix):
            matches.append((name, p))

    if not matches:
        print(f"[debug] could not find suffix: {suffix}")
        print("[debug] parameter names containing 'gate'/'down'/'expert':")
        shown = 0
        for name, _ in model.named_parameters():
            lname = name.lower()
            if "gate" in lname or "down" in lname or "expert" in lname:
                print(" ", name)
                shown += 1
                if shown >= 200:
                    break
        raise RuntimeError(f"Could not find any parameter with suffix: {suffix}")

    matches = sorted(matches, key=lambda x: _extract_layer_id(x[0]))
    return matches


@torch.no_grad()
def load_overlay(
    model,
    overlay_path: str,
    strict: bool = True,
):
    overlay_path = str(Path(overlay_path).resolve())
    print(f"[info] loading overlay from: {overlay_path}")
    payload = torch.load(overlay_path, map_location="cpu")

    metadata = payload.get("metadata", {})
    tensors = payload.get("tensors", {})

    if not isinstance(tensors, dict) or len(tensors) == 0:
        raise RuntimeError("Overlay file has no tensors")

    name_to_param = dict(model.named_parameters())

    loaded = 0
    missing = []
    shape_mismatch = []

    for name, saved_tensor in tensors.items():
        if name not in name_to_param:
            missing.append(name)
            continue

        dst = name_to_param[name]
        if tuple(dst.shape) != tuple(saved_tensor.shape):
            shape_mismatch.append((name, tuple(saved_tensor.shape), tuple(dst.shape)))
            continue

        dst.data.copy_(saved_tensor.to(device=dst.device, dtype=dst.dtype))
        loaded += 1

    print(f"[info] overlay tensors loaded: {loaded}")
    if metadata:
        print(f"[info] overlay metadata: {json.dumps(metadata, ensure_ascii=False)}")

    if missing:
        print("[warn] missing params when loading overlay:")
        for x in missing[:20]:
            print(" ", x)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if shape_mismatch:
        print("[warn] shape mismatch params when loading overlay:")
        for name, a, b in shape_mismatch[:20]:
            print(f"  {name}: saved={a}, current={b}")
        if len(shape_mismatch) > 20:
            print(f"  ... and {len(shape_mismatch) - 20} more")

    if strict and (missing or shape_mismatch):
        raise RuntimeError("Overlay load failed under strict=True")

    return {
        "loaded": loaded,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "metadata": metadata,
    }


@torch.no_grad()
def summarize_overlay_sparsity(model):
    gate_params = find_layerwise_params(model, "mlp.experts.gate_up_proj")
    down_params = find_layerwise_params(model, "mlp.experts.down_proj")

    gate_sps = [tensor_sparsity(p.data) for _, p in gate_params]
    down_sps = [tensor_sparsity(p.data) for _, p in down_params]

    out = {
        "num_gate_layers": len(gate_sps),
        "num_down_layers": len(down_sps),
        "gate_avg_sparsity": sum(gate_sps) / len(gate_sps) if gate_sps else None,
        "down_avg_sparsity": sum(down_sps) / len(down_sps) if down_sps else None,
    }
    print(f"[info] overlay sparsity summary: {out}")
    return out