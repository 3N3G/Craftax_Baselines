#!/bin/bash
#SBATCH --job-name=probe_eval_env
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/probe_eval_env_%j.out
#SBATCH --error=logs/probe_eval_env_%j.err

set -euo pipefail
set +e
set +u
set +o pipefail
source ~/.bashrc
set -e
set -u
set -o pipefail

cd ~/Craftax_Baselines

envs=(
  /data/user_data/geney/.conda/envs/imaug
  /data/user_data/geney/.conda/envs/craftax
  /data/user_data/geney/.conda/envs/test
  /data/user_data/geney/.conda/envs/craftax_fast_llm
  /data/user_data/geney/.conda/envs/flash_env
  /data/user_data/geney/.conda/envs/tf_env_conda
  /data/user_data/geney/.conda/envs/verl
)

for envp in "${envs[@]}"; do
  py="${envp}/bin/python"
  if [[ ! -x "${py}" ]]; then
    echo "=== ${envp} missing python ==="
    continue
  fi
  echo "=== ${envp} ==="
  "${py}" - <<"PY"
import importlib
mods=["torch","jax","craftax","flax","optax","orbax.checkpoint","distrax","cv2"]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, "ok")
    except Exception as exc:
        print(m, "FAIL", exc.__class__.__name__, exc)
try:
    import jax
    print("jax_default_backend", jax.default_backend())
    print("jax_devices", [str(d) for d in jax.devices()])
except Exception as exc:
    print("jax_probe_FAIL", exc.__class__.__name__, exc)
PY
  echo
 done
