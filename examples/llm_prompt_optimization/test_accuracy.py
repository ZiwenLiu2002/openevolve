import os

os.environ["OE_STAGE2_SAVE_DIR"] = "stage2_traces_glm"
os.environ["OPENEVOLVE_PROMPT"] = "gsm8k_prompt.txt"

os.environ["OPENEVOLVE_MODEL_PATH"] = "../../../gsm8k/zai-org/GLM-4-9B-0414-wanda-s065"

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

from evaluator import evaluate_stage2

res = evaluate_stage2("gsm8k_prompt.txt")
print(res)