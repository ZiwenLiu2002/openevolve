"""
Evaluator for HuggingFace dataset-based prompt optimization.
"""

import os
import re
import time
import json
import yaml
import uuid
import random
import hashlib
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed as hf_set_seed,
)
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast

MAX_PROMPT_CHARS = 1200

with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

llm_config = GLOBAL_CONFIG.get("llm", {})
evaluator_config = GLOBAL_CONFIG.get("evaluator", {})

MAX_RETRIES = int(evaluator_config.get("max_retries", 3))


def _as_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_env_or_cfg(name: str, cfg_value, cast_fn=None):
    v = os.environ.get(name, None)
    if v is None:
        return cfg_value
    return cast_fn(v) if cast_fn is not None else v


# -------------------------
# Repro / generation config
# -------------------------
GLOBAL_SEED = _get_env_or_cfg("OPENEVOLVE_SEED", llm_config.get("seed", 1234), int)
DETERMINISTIC = _as_bool(
    _get_env_or_cfg("OPENEVOLVE_DETERMINISTIC", llm_config.get("deterministic", True))
)

USE_GREEDY = _as_bool(
    _get_env_or_cfg("OPENEVOLVE_GREEDY", llm_config.get("use_greedy", True))
)

USE_CHAT_TEMPLATE = _as_bool(
    _get_env_or_cfg("OPENEVOLVE_USE_CHAT_TEMPLATE", llm_config.get("use_chat_template", False))
)

ENABLE_THINKING = _as_bool(
    _get_env_or_cfg("OPENEVOLVE_ENABLE_THINKING", llm_config.get("enable_thinking", False))
)

STRIP_THINKING = _as_bool(
    _get_env_or_cfg("OPENEVOLVE_STRIP_THINKING", llm_config.get("strip_thinking", True))
)

MAX_NEW_TOKENS = _get_env_or_cfg(
    "OPENEVOLVE_MAX_NEW_TOKENS", llm_config.get("max_new_tokens", 1024), int
)

TEMPERATURE = float(
    _get_env_or_cfg("OPENEVOLVE_TEMPERATURE", llm_config.get("temperature", 0.7), float)
)
TOP_P = float(
    _get_env_or_cfg("OPENEVOLVE_TOP_P", llm_config.get("top_p", 0.8), float)
)
TOP_K = int(
    _get_env_or_cfg("OPENEVOLVE_TOP_K", llm_config.get("top_k", 20), int)
)

LOAD_IN_4BIT = _as_bool(
    _get_env_or_cfg("OPENEVOLVE_LOAD_IN_4BIT", llm_config.get("load_in_4bit", True))
)

MODEL_DTYPE_STR = str(
    _get_env_or_cfg("OPENEVOLVE_MODEL_DTYPE", llm_config.get("model_dtype", "bfloat16"))
).lower()

TARGET_MODEL_PATH = str(
    _get_env_or_cfg("OPENEVOLVE_MODEL_PATH", llm_config.get("model_path", "Qwen/Qwen3.5-9B"))
)

print(f"Using max_new_tokens: {MAX_NEW_TOKENS}")
print(
    f"[gen cfg] seed={GLOBAL_SEED} deterministic={DETERMINISTIC} "
    f"use_greedy={USE_GREEDY} use_chat_template={USE_CHAT_TEMPLATE} "
    f"enable_thinking={ENABLE_THINKING} strip_thinking={STRIP_THINKING}"
)
if not USE_GREEDY:
    print(f"[sampling cfg] temperature={TEMPERATURE} top_p={TOP_P} top_k={TOP_K}")
print(
    f"[model cfg] model={TARGET_MODEL_PATH} load_in_4bit={LOAD_IN_4BIT} "
    f"model_dtype={MODEL_DTYPE_STR}"
)


def resolve_torch_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


MODEL_DTYPE = resolve_torch_dtype(MODEL_DTYPE_STR)


def set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[warn] could not enable full deterministic algorithms: {e}")
    else:
        torch.backends.cudnn.benchmark = True


set_global_seed(GLOBAL_SEED, deterministic=DETERMINISTIC)


print(f"Loading local target model from: {TARGET_MODEL_PATH}")

model_name_lower = TARGET_MODEL_PATH.lower()
is_olmo3 = "olmo-3" in model_name_lower or "olmo3" in model_name_lower
is_old_olmo = ("allenai/olmo" in model_name_lower or os.path.basename(TARGET_MODEL_PATH).lower().startswith("olmo")) and not is_olmo3
if is_olmo3:
    target_tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
    )

    if target_tokenizer.pad_token_id is None and target_tokenizer.eos_token_id is not None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    if LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_PATH,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=MODEL_DTYPE,
        )
        if torch.cuda.is_available():
            target_model = target_model.to("cuda")

elif is_old_olmo:
    target_tokenizer = OLMoTokenizerFast.from_pretrained(TARGET_MODEL_PATH)

    if target_tokenizer.pad_token_id is None and target_tokenizer.eos_token_id is not None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    target_model = OLMoForCausalLM.from_pretrained(TARGET_MODEL_PATH)

    if torch.cuda.is_available():
        target_model = target_model.to("cuda")

else:
    target_tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
    )

    if target_tokenizer.pad_token_id is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    if LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_PATH,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_PATH,
            torch_dtype=MODEL_DTYPE,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            target_model = target_model.to("cuda")

print(">>> Model is on device:", next(target_model.parameters()).device)

target_model.eval()
print("Local target model loaded")
if getattr(target_model, "generation_config", None) is not None:
    if target_model.generation_config.pad_token_id is None:
        target_model.generation_config.pad_token_id = target_tokenizer.pad_token_id
    if target_model.generation_config.eos_token_id is None:
        target_model.generation_config.eos_token_id = target_tokenizer.eos_token_id

prompt_file = os.environ.get("OPENEVOLVE_PROMPT")
if not prompt_file:
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG_PATH = os.path.join(evaluator_dir, "dataset_settings.yaml")
    print("Warning: OPENEVOLVE_PROMPT not set. Using default dataset_settings.yaml")
else:
    basename = os.path.basename(prompt_file)
    dataset_filename = basename.replace("_prompt.txt", "_prompt_dataset.yaml").replace(
        ".txt", "_dataset.yaml"
    )
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG_PATH = os.path.join(evaluator_dir, dataset_filename)
    print(f"Dataset configuration: {dataset_filename}")


def calculate_prompt_features(prompt):
    prompt_length = len(prompt)
    prompt_lower = prompt.lower()
    sophistication_score = 0.0
    if len(prompt) >= 100:
        sophistication_score += 0.1
    has_example = (
        "example" in prompt_lower
        or prompt.count("####") >= 4
        or bool(re.search(r"problem:.*?solution:", prompt_lower, re.DOTALL))
    )
    has_cot = (
        "step by step" in prompt_lower
        or "step-by-step" in prompt_lower
        or any(phrase in prompt_lower for phrase in ["think through", "reasoning", "explain your"])
        or bool(re.search(r"(first|then|next|finally)", prompt_lower))
    )
    has_directive = "solve" in prompt_lower or "calculate" in prompt_lower
    has_strict = "must" in prompt_lower or "exactly" in prompt_lower
    if has_example:
        sophistication_score += 0.6
        if has_cot:
            sophistication_score += 0.3
        elif len(prompt) > 1500:
            sophistication_score += 0.2
        else:
            sophistication_score += 0.1
    elif has_cot:
        sophistication_score += 0.4
        if has_strict:
            sophistication_score += 0.2
        elif len(prompt) > 500:
            sophistication_score += 0.15
        else:
            sophistication_score += 0.1
    else:
        if has_directive:
            sophistication_score += 0.2
        else:
            sophistication_score += 0.1
    sophistication_score = min(1.0, max(0.0, sophistication_score))
    return prompt_length, sophistication_score


def load_prompt_config(prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()
    if not os.path.exists(DATASET_CONFIG_PATH):
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")
    with open(DATASET_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config, prompt


def load_hf_dataset(config):
    dataset_name = config["dataset_name"]
    dataset_config = config.get("dataset_config", None)
    split = config.get("split", "train")

    print(f"Loading dataset: {dataset_name}")

    try:
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split=split,
            )
    except Exception:
        print(f"Split '{split}' not found, falling back to 'train'")
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split="train",
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split="train",
            )

    print(f"Dataset loaded with {len(dataset)} examples")
    return dataset

@torch.inference_mode()
def local_llm_generate(prompt: str, max_new_tokens: int = 256) -> str:
    text_for_model = prompt

    if USE_CHAT_TEMPLATE and hasattr(target_tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            text_for_model = target_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
        except TypeError:
            text_for_model = target_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    model_device = next(target_model.parameters()).device

    inputs = target_tokenizer(
        text_for_model,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        return_token_type_ids=False,
    )

    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # 官方例子里没有传 token_type_ids；这里保险起见再删一次
    inputs.pop("token_type_ids", None)

    if USE_GREEDY:
        outputs = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=target_tokenizer.pad_token_id,
            eos_token_id=target_tokenizer.eos_token_id,
        )
    else:
        outputs = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            pad_token_id=target_tokenizer.pad_token_id,
            eos_token_id=target_tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]
    gen = target_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if STRIP_THINKING and "</think>" in gen:
        gen = gen.split("</think>", 1)[1].strip()

    return gen

def evaluate_prompt(prompt, dataset, config, num_samples, return_traces: bool = False):
    input_field = config["input_field"]
    target_field = config["target_field"]
    dataset_name = config.get("dataset_name", "").lower()
    is_emotion = "emotion" in dataset_name
    is_gsm8k = "gsm8k" in dataset_name
    is_hotpotqa = config.get("is_hotpotqa", False)
    is_ifeval = config.get("is_ifeval", False)
    is_hover = config.get("is_hover", False)

    if hasattr(dataset, "take"):
        samples = dataset.take(num_samples)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples", total=num_samples)
    else:
        indices = range(min(num_samples, len(dataset)))
        samples = dataset.select(indices)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples")

    correct = 0
    total = 0
    traces = [] if return_traces else None

    for example in sample_iter:
        input_text = example[input_field]
        expected = example[target_field]

        if is_hotpotqa:
            context_items = example.get("context", {})
            context_text = ""
            if "title" in context_items and "sentences" in context_items:
                for i, (title, sentences) in enumerate(
                    zip(context_items["title"], context_items["sentences"])
                ):
                    context_text += f"Paragraph {i+1} ({title}):\n"
                    context_text += " ".join(sentences) + "\n\n"
            formatted_prompt = prompt.format(context=context_text.strip(), question=input_text)
        elif is_ifeval:
            formatted_prompt = prompt.format(instruction=input_text)
        elif is_hover:
            formatted_prompt = prompt.format(claim=input_text)
        else:
            formatted_prompt = prompt.format(input_text=input_text)

        output_text = None
        for attempt in range(MAX_RETRIES):
            try:
                output_text = local_llm_generate(
                    formatted_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get response after {MAX_RETRIES} attempts: {e}")
                    total += 1
                    output_text = None
                else:
                    time.sleep(1)

        if output_text is None:
            continue

        output_text = output_text.strip()

        try:
            is_correct = False
            prediction = None
            output_token_len = len(
                target_tokenizer.encode(output_text, add_special_tokens=False)
            )

            if is_gsm8k:
                expected_answer = expected.split("####")[-1].strip()
                try:
                    expected_number = float(expected_answer.replace(",", ""))
                except Exception:
                    print(f"Warning: Could not parse expected answer: {expected_answer}")
                    total += 1
                    continue

                has_hash_anchor = "####" in output_text
                if has_hash_anchor:
                    predicted_answer = output_text.split("####")[-1].strip()
                    numbers = re.findall(r"-?\$?[\d,]+\.?\d*", predicted_answer)
                    if numbers:
                        try:
                            number_str = numbers[0].replace("$", "").replace(",", "")
                            prediction = float(number_str)
                        except Exception:
                            prediction = None

                if prediction is not None and abs(prediction - expected_number) < 0.001:
                    is_correct = True
                    correct += 1

                total += 1

                if return_traces:
                    traces.append(
                        {
                            "input": input_text,
                            "target": expected_answer,
                            "prediction": None if prediction is None else str(prediction),
                            "model_output": output_text,
                            "correct": bool(is_correct),
                            "has_hash_anchor": has_hash_anchor,
                            "missing_final_answer": prediction is None,
                            "output_token_len": output_token_len,
                            "formatted_prompt_preview": formatted_prompt[:400],
                        }
                    )
                continue

            elif is_hotpotqa:
                output_lower = output_text.lower().strip()
                expected_lower = str(expected).lower().strip()
                output_lower = output_lower.rstrip(".,!?;:")
                expected_lower = expected_lower.rstrip(".,!?;:")

                if output_lower == expected_lower or expected_lower in output_lower:
                    is_correct = True
                    correct += 1

                total += 1

                if return_traces:
                    traces.append(
                        {
                            "input": input_text,
                            "target": str(expected),
                            "prediction": output_text,
                            "model_output": output_text,
                            "correct": bool(is_correct),
                            "output_token_len": output_token_len,
                            "formatted_prompt_preview": formatted_prompt[:400],
                        }
                    )
                continue

            elif is_ifeval:
                if len(output_text.strip()) > 10:
                    is_correct = True
                    correct += 1

                total += 1

                if return_traces:
                    traces.append(
                        {
                            "input": input_text,
                            "target": str(expected),
                            "prediction": output_text,
                            "model_output": output_text,
                            "correct": bool(is_correct),
                            "output_token_len": output_token_len,
                            "formatted_prompt_preview": formatted_prompt[:400],
                        }
                    )
                continue

            elif is_hover:
                output_upper = output_text.upper()
                expected_upper = str(expected).upper()

                if "SUPPORTED" in output_upper and "NOT" not in output_upper.replace("NOT SUPPORTED", ""):
                    prediction = "SUPPORTED"
                elif "NOT SUPPORTED" in output_upper or "NOT_SUPPORTED" in output_upper:
                    prediction = "NOT_SUPPORTED"
                else:
                    prediction = None

                if prediction == expected_upper:
                    is_correct = True
                    correct += 1

                total += 1

                if return_traces:
                    traces.append(
                        {
                            "input": input_text,
                            "target": expected_upper,
                            "prediction": prediction,
                            "model_output": output_text,
                            "correct": bool(is_correct),
                            "output_token_len": output_token_len,
                            "formatted_prompt_preview": formatted_prompt[:400],
                        }
                    )
                continue

            elif is_emotion:
                numbers = re.findall(r"\b[0-5]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])
                else:
                    output_lower = output_text.lower()
                    emotion_map = {
                        "sadness": 0, "sad": 0,
                        "joy": 1, "happy": 1, "happiness": 1,
                        "love": 2,
                        "anger": 3, "angry": 3,
                        "fear": 4, "afraid": 4, "scared": 4,
                        "surprise": 5, "surprised": 5,
                    }
                    prediction = -1
                    for emotion, label in emotion_map.items():
                        if emotion in output_lower:
                            prediction = label
                            break

                if prediction == expected:
                    is_correct = True
                    correct += 1

                total += 1

                if return_traces:
                    traces.append(
                        {
                            "input": input_text,
                            "target": expected,
                            "prediction": prediction,
                            "model_output": output_text,
                            "correct": bool(is_correct),
                            "output_token_len": output_token_len,
                            "formatted_prompt_preview": formatted_prompt[:400],
                        }
                    )
                continue

            else:
                numbers = re.findall(r"\b[01]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])
                else:
                    output_lower = output_text.lower()
                    if "positive" in output_lower:
                        prediction = 1
                    elif "negative" in output_lower:
                        prediction = 0
                    else:
                        prediction = -1

                if prediction == expected:
                    is_correct = True
                    correct += 1

                total += 1

                if return_traces:
                    traces.append(
                        {
                            "input": input_text,
                            "target": expected,
                            "prediction": prediction,
                            "model_output": output_text,
                            "correct": bool(is_correct),
                            "output_token_len": output_token_len,
                            "formatted_prompt_preview": formatted_prompt[:400],
                        }
                    )
                continue

        except Exception as e:
            print(f"Error parsing response '{output_text}': {e}")
            total += 1
            if return_traces:
                traces.append(
                    {
                        "input": input_text,
                        "target": expected,
                        "prediction": None,
                        "model_output": output_text,
                        "correct": False,
                        "error": str(e),
                        "output_token_len": len(
                            target_tokenizer.encode(output_text, add_special_tokens=False)
                        ) if output_text else 0,
                        "formatted_prompt_preview": formatted_prompt[:400],
                    }
                )

    accuracy = correct / total if total > 0 else 0.0
    if return_traces:
        return accuracy, correct, total, traces
    return accuracy, correct, total

def _hash_prompt(p: str) -> str:
    return hashlib.sha256(p.encode("utf-8")).hexdigest()


def _load_stage1_queues(path: str, k_best: int, k_worst: int):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        data.setdefault("best", [])
        data.setdefault("worst", [])
        data.setdefault("k_best", k_best)
        data.setdefault("k_worst", k_worst)
        return data
    return {"k_best": k_best, "k_worst": k_worst, "best": [], "worst": []}


def _dedup_keep_better(items, prefer_higher: bool):
    by_hash = {}
    for it in items:
        h = it.get("prompt_hash") or _hash_prompt(it["prompt"])
        it["prompt_hash"] = h
        if h not in by_hash:
            by_hash[h] = it
        else:
            if prefer_higher:
                if it["accuracy"] > by_hash[h]["accuracy"]:
                    by_hash[h] = it
            else:
                if it["accuracy"] < by_hash[h]["accuracy"]:
                    by_hash[h] = it
    return list(by_hash.values())


def update_stage1_queues(path: str, prompt: str, accuracy: float, k_best: int = 5, k_worst: int = 5):
    data = _load_stage1_queues(path, k_best, k_worst)
    now = datetime.now().isoformat()
    item = {
        "accuracy": float(accuracy),
        "prompt": prompt,
        "prompt_hash": _hash_prompt(prompt),
        "timestamp": now,
    }

    best = data.get("best", []) + [item]
    best = _dedup_keep_better(best, prefer_higher=True)
    best.sort(key=lambda x: x["accuracy"], reverse=True)
    best = best[:k_best]

    worst = data.get("worst", []) + [item]
    worst = _dedup_keep_better(worst, prefer_higher=False)
    worst.sort(key=lambda x: x["accuracy"])
    worst = worst[:k_worst]

    out = {
        "k_best": k_best,
        "k_worst": k_worst,
        "best": best,
        "worst": worst,
        "updated_at": now,
    }
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def evaluate_stage1(prompt_path):
    print("-" * 80)
    print("Starting Stage 1 evaluation...")
    print("-" * 80)
    try:
        config, prompt = load_prompt_config(prompt_path)
        print("Loaded prompt configuration")

        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        if prompt_length > MAX_PROMPT_CHARS:
            print(f"Prompt too long ({prompt_length} chars > {MAX_PROMPT_CHARS}), assigning score 0.0")
            return {
                "combined_score": 0.0,
                "null_prediction_rate": 1.0,
                "prompt_length": prompt_length,
                "reasoning_strategy": reasoning_sophistication,
            }

        dataset = load_hf_dataset(config)
        stage1_samples = 10
        print(f"Stage 1: Evaluating {stage1_samples} samples...")

        accuracy, correct, total, traces = evaluate_prompt(
            prompt, dataset, config, stage1_samples, return_traces=True
        )

        null_prediction_count = sum(1 for t in traces if t.get("prediction") is None)
        null_prediction_rate = null_prediction_count / total if total > 0 else 1.0

        qcfg = GLOBAL_CONFIG.get("reflection_prompt_queues", {})
        if qcfg.get("enabled", False):
            update_stage1_queues(
                path=qcfg.get("path", "stage1_prompt_queues.json"),
                prompt=prompt,
                accuracy=accuracy,
                k_best=int(qcfg.get("k_best", 5)),
                k_worst=int(qcfg.get("k_worst", 5)),
            )

        print(
            f"Stage 1 accuracy: {accuracy:.3f} ({correct}/{total}), "
            f"null_prediction_rate: {null_prediction_rate:.3f}"
        )
        print("-" * 80)
        print(
            f"Prompt features - Length: {prompt_length} chars, "
            f"Reasoning sophistication: {reasoning_sophistication:.3f}"
        )
        return {
            "combined_score": accuracy,
            "null_prediction_rate": null_prediction_rate,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        print(f"Stage 1 evaluation failed: {str(e)}")
        traceback.print_exc()
        print("-" * 80)
        try:
            with open(prompt_path, "r") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except Exception:
            prompt_length, reasoning_sophistication = 0, 0.0
        return {
            "combined_score": 0.0,
            "null_prediction_rate": 1.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def _save_stage2_trace(rows: list, program_path: str, metrics: dict):
    out_dir = os.environ.get("OE_STAGE2_SAVE_DIR", "").strip()
    if not out_dir:
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    trace_id = uuid.uuid4().hex
    out_path = Path(out_dir) / f"stage2_{trace_id}.jsonl"

    header = {
        "timestamp": time.time(),
        "program_path": program_path,
        "metrics": metrics,
    }

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", **header}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps({"type": "row", **r}, ensure_ascii=False) + "\n")


def evaluate_stage2(prompt_path):
    print("-" * 80)
    print("Starting Stage 2 evaluation...")
    print("-" * 80)
    try:
        config, prompt = load_prompt_config(prompt_path)
        print("Loaded prompt configuration")

        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        if prompt_length > MAX_PROMPT_CHARS:
            print(f"Prompt too long ({prompt_length} chars > {MAX_PROMPT_CHARS}), assigning score 0.0")
            return {
                "combined_score": 0.0,
                "null_prediction_rate": 1.0,
                "parseable_answer_rate": 0.0,
                "avg_output_tokens": 0.0,
                "prompt_length": prompt_length,
                "reasoning_strategy": reasoning_sophistication,
            }

        dataset = load_hf_dataset(config)
        stage2_samples = 50
        print(f"Stage 2: Evaluating {stage2_samples} samples...")

        accuracy, correct, total, traces = evaluate_prompt(
            prompt, dataset, config, stage2_samples, return_traces=True
        )

        null_prediction_count = sum(1 for t in traces if t.get("prediction") is None)
        null_prediction_rate = null_prediction_count / total if total > 0 else 1.0
        parseable_answer_rate = 1.0 - null_prediction_rate

        output_token_lengths = [
            t.get("output_token_len", 0) for t in traces if isinstance(t.get("output_token_len", 0), (int, float))
        ]
        avg_output_tokens = (
            sum(output_token_lengths) / len(output_token_lengths)
            if output_token_lengths else 0.0
        )

        print(
            f"Stage 2 accuracy: {accuracy:.3f} ({correct}/{total}), "
            f"null_prediction_rate: {null_prediction_rate:.3f}, "
            f"avg_output_tokens: {avg_output_tokens:.1f}"
        )
        print("-" * 80)
        print(
            f"Prompt features - Length: {prompt_length} chars, "
            f"Reasoning sophistication: {reasoning_sophistication:.3f}"
        )

        metrics = {
            "combined_score": accuracy,
            "null_prediction_rate": null_prediction_rate,
            "parseable_answer_rate": parseable_answer_rate,
            "avg_output_tokens": avg_output_tokens,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

        _save_stage2_trace(rows=traces, program_path=prompt_path, metrics=metrics)

        return metrics

    except Exception as e:
        print(f"Stage 2 evaluation failed: {str(e)}")
        traceback.print_exc()
        print("-" * 80)
        try:
            with open(prompt_path, "r") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except Exception:
            prompt_length, reasoning_sophistication = 0, 0.0
        return {
            "combined_score": 0.0,
            "null_prediction_rate": 1.0,
            "parseable_answer_rate": 0.0,
            "avg_output_tokens": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate(prompt_path):
    return evaluate_stage2(prompt_path)