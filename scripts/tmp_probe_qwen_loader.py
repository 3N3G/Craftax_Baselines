from pathlib import Path

p = Path(/data/user_data/geney/.conda/envs/craftax_fast_llm/lib/python3.10/site-packages/vllm/model_executor/models/qwen2.py)
print(path_exists, p.exists(), p)
if p.exists():
    text = p.read_text(encoding=utf-8).splitlines()
    for i in range(420, 620):
        if i <= len(text):
            print(f"{i:04d}: {text[i-1]}")
