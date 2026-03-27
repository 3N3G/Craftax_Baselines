#!/usr/bin/env bash
set -eo pipefail
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm
cd ~/Craftax_Baselines
python3 - <<'PY'
import inspect
import json
import pathlib
import urllib.request
import sys

import transformers
import vllm
from vllm.transformers_utils.configs.speculators import base as spec_base

print("python", sys.version)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
base_path = inspect.getsourcefile(spec_base)
print("spec_base_path", base_path)
text = pathlib.Path(base_path).read_text()
for i, line in enumerate(text.splitlines(), 1):
    if "class SpeculatorsConfig" in line or "def from_pretrained" in line or "return cls(" in line or "method" in line:
        print(f"{i:04d}: {line}")

print("--- qwen35 raw keys ---")
with urllib.request.urlopen("https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json", timeout=20) as r:
    cfg = json.loads(r.read().decode())
print("model_type", cfg.get("model_type"))
print("architectures", cfg.get("architectures"))
print("has_auto_map", "auto_map" in cfg)
print("method_keys", [k for k in cfg if "method" in k.lower()])

from transformers import AutoConfig
try:
    c = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")
    print("autoconfig_no_remote_ok", c.model_type)
except Exception as e:
    print("autoconfig_no_remote_err", type(e).__name__, str(e)[:220])

try:
    c = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    print("autoconfig_remote_ok", c.model_type)
except Exception as e:
    print("autoconfig_remote_err", type(e).__name__, str(e)[:220])
PY
