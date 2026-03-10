# Prompt Iteration Webapp (Fixed 10-State Set)

This tool lets you edit prompt sections and run Qwen/vLLM responses on a deterministic 10-state Craftax set with strict coordinate validation.

## Access Model (Important)
When Streamlit runs on Babel with `--server.address 127.0.0.1`, that `127.0.0.1` is Babel's loopback interface, not your laptop.

So you cannot open Babel's app directly from your local browser unless you tunnel it.

Use SSH local port-forwarding:
- Babel side: run Streamlit bound to `127.0.0.1:<port>`
- Local side: `ssh -L <local_port>:127.0.0.1:<remote_port> babel`
- Browser side: open `http://127.0.0.1:<local_port>` on your laptop

## Files
- `configs/prompt_iter/fixed_states_v1.json`
- `scripts/prompt_iter_backend.py`
- `scripts/prompt_iter_webapp.py`

## Fixed State Set
`fixed_states_v1.json` contains exactly 10 states:
- 5 golden bundle states: `step_0006`, `step_0240`, `step_0619`, `step_1103`, `step_1456`
- 5 trajectory states from `traj_20260310_134702/text_obs.jsonl`: `t={0,155,240,425,567}`

Every loaded state is passed through:
- `filter_text_obs(..., strict_map_validation=True)`

If coordinate formatting is malformed, loading fails early.

## Launch On Babel

### Quickstart (Two Terminals)
Terminal A (keep running):
```bash
zsh -lic 'ssh babel "cd ~/Craftax_Baselines && PYTHONPATH=\$PWD python3 -m streamlit run scripts/prompt_iter_webapp.py --server.address 127.0.0.1 --server.port 8501"'
```

Terminal B (local tunnel):
```bash
zsh -lic 'ssh -N -L 8501:127.0.0.1:8501 babel'
```

Then open locally:
`http://127.0.0.1:8501`

### 1) Start Streamlit on Babel
```bash
zsh -lic 'ssh babel "cd ~/Craftax_Baselines && PYTHONPATH=\$PWD python3 -m streamlit run scripts/prompt_iter_webapp.py --server.address 127.0.0.1 --server.port 8501"'
```

If `streamlit` is missing:
```bash
zsh -lic 'ssh babel "python3 -m pip install --user streamlit"'
```

### 2) Port-forward locally
```bash
zsh -lic 'ssh -N -L 8501:127.0.0.1:8501 babel'
```

Open: `http://127.0.0.1:8501`

## Backend/Manifest Smoke Checks
List states:
```bash
zsh -lic 'ssh babel "cd ~/Craftax_Baselines && PYTHONPATH=\$PWD python3 scripts/prompt_iter_backend.py --list-states --show-examples"'
```

Single completion test:
```bash
zsh -lic 'ssh babel "cd ~/Craftax_Baselines && PYTHONPATH=\$PWD python3 scripts/prompt_iter_backend.py --state-id traj_20260310_134702_t0155 --prompt-variant future_based_opt --server-url http://127.0.0.1:8000 --model-name ./configs/vllm_hidden_qwen4b --max-tokens 256 --temperature 0.7"'
```

10-state smoke (run against an active online-RL job's vLLM endpoint):
```bash
zsh -lic 'ssh babel "srun --jobid=<running_jobid> --overlap --ntasks=1 --cpus-per-task=1 bash -lc '\''source ~/.bashrc && conda activate /data/user_data/geney/.conda/envs/craftax && cd ~/Craftax_Baselines && PYTHONPATH=\$PWD python3 scripts/prompt_iter_smoke.py --variant default --server-url http://127.0.0.1:<job_vllm_port> --model-name ./configs/vllm_hidden_qwen4b --max-tokens 8 --output-json /home/geney/Craftax_Baselines/logs/prompt_iter_smoke.json'\''"'
```

## Online Run Replacement Checks

Queue check:
```bash
zsh -lic 'ssh babel "squeue -u geney -o \"%.18i %.36j %.10T %.10M %.9P %.30R\""'
```

Check strict coordinate guard failures in job logs:
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && rg -n "Malformed '\''Map \(interesting tiles only\)'\''|Failed to parse map block" logs/online_rl_hidden_jax_*.out'
```

Check CoT prompt samples for malformed tokens like `-5:tree`:
```bash
zsh -lic 'ssh babel "python - <<'\''PY'\''
import json
import re
from pathlib import Path

paths = sorted(Path(\"/data/group_data/rl/geney/online_rl_hidden_models/cot_logs\").glob(\"online-cot-*.jsonl\"))[-4:]
bad = re.compile(r\"(^|,\\s*)-?\\d+\\s*:[^,]+\")
for p in paths:
    seen = 0
    broken = 0
    with p.open() as f:
        for line in f:
            rec = json.loads(line)
            for s in rec.get(\"samples\", []):
                prompt = s.get(\"prompt\", \"\")
                marker = \"Map (interesting tiles only): \"
                if marker in prompt:
                    payload = prompt.split(marker, 1)[1].split(\"\\n\", 1)[0]
                    seen += 1
                    if bad.search(payload):
                        broken += 1
            if seen >= 8:
                break
    print(f\"{p.name}: checked={seen} malformed={broken}\")
PY"'
```
