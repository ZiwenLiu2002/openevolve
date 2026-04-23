from evaluator import evaluate_flip_pairs_and_save

res = evaluate_flip_pairs_and_save(
    flip_json_path="code/data/flip_pairs.json",
    baseline_prompt_path="gsm8k_prompt.txt",
    evolved_prompt_path="aya_output_0402/best/best_program.txt",
    outdir="ffn_flip_evalstyle_saved",
    mode="baseline_wrong_evolved_right",   # 或 baseline_right_evolved_wrong
    max_cases=-1,
    T=256,
    layer=15,
)
print(res)