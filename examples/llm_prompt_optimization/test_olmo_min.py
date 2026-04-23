import os
import random
import numpy as np
import torch

from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast


MODEL_NAME = "allenai/OLMo-7B-Instruct"
PROMPTS = [
    "Question: If John has 3 apples and buys 2 more, how many apples does he have?\nAnswer:",
    "Solve the problem and give only the final answer.\n\nQuestion: If John has 3 apples and buys 2 more, how many apples does he have?\nFinal answer:",
    "What is 3 + 2? Answer with only the number.\nAnswer:",
]

SEED = 1234


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_one(model, tokenizer, prompt: str, do_sample: bool):
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    inputs.pop("token_type_ids", None)
    inputs = move_to_device(inputs, device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=80,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    if do_sample:
        gen_kwargs.update(
            dict(
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
            )
        )
    else:
        gen_kwargs.update(
            dict(
                do_sample=False,
                num_beams=1,
            )
        )

    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    full_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    new_ids = outputs[0][input_len:]
    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    print("=" * 100)
    print(f"PROMPT: {repr(prompt)}")
    print(f"MODE: {'sampling' if do_sample else 'greedy'}")
    print(f"INPUT TOKENS: {input_len}")
    print(f"NEW TOKENS: {new_ids.shape[0]}")
    print("-" * 100)
    print("FULL OUTPUT:")
    print(repr(full_text))
    print("-" * 100)
    print("NEW TEXT ONLY:")
    print(repr(new_text))
    print("=" * 100)
    print()


def main():
    set_seed(SEED)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = OLMoTokenizerFast.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}")
    model = OLMoForCausalLM.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()

    print("Model device:", next(model.parameters()).device)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("eos_token_id:", tokenizer.eos_token_id)
    print()

    for prompt in PROMPTS:
        run_one(model, tokenizer, prompt, do_sample=False)
        run_one(model, tokenizer, prompt, do_sample=True)

    print("Repeat stability check on one prompt:")
    test_prompt = "Question: What is 27 + 15?\nAnswer:"
    for i in range(3):
        set_seed(SEED + i)
        run_one(model, tokenizer, test_prompt, do_sample=False)


if __name__ == "__main__":
    main()