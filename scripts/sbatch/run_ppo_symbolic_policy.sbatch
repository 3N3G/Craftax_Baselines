#!/bin/bash
#SBATCH --job-name=ppo_symbolic_policy
#SBATCH --output=/home/geney/Craftax_Baselines/logs/ppo_symbolic_policy_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/ppo_symbolic_policy_%j.err
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00

set -euo pipefail

# Source bashrc defensively.
set +e
set +u
set +o pipefail
source ~/.bashrc
BASHRC_RC=$?
set -e
set -u
set -o pipefail
if [[ $BASHRC_RC -ne 0 ]]; then
    echo "WARNING: source ~/.bashrc exited with code $BASHRC_RC; continuing"
fi

cd ~/Craftax_Baselines
mkdir -p logs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found after sourcing bashrc"
    exit 1
fi

source scripts/sbatch/conda_env_setup.sh
sbatch_require_conda

PPO_ENV_PATH="${PPO_ENV_PATH:-/data/user_data/geney/.conda/envs/craftax}"
sbatch_activate_conda_env "ppo-symbolic" "${PPO_ENV_PATH}"

python_import_preflight() {
    python - <<'PY' >/dev/null
import importlib
mods = ["jax", "flax", "optax", "distrax", "craftax", "numpy", "wandb"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m}: {exc.__class__.__name__}: {exc}")
if missing:
    raise SystemExit("\n".join(missing))
PY
}

python_import_preflight

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1e10}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-64}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-ppo_symbolic_policy_${SLURM_JOB_ID}}"
WANDB_PROJECT="${WANDB_PROJECT:-unaugmented_craftax_ppo}"
WANDB_ENTITY="${WANDB_ENTITY:-iris-sobolmark}"
POLICY_SAVE_DIR="${POLICY_SAVE_DIR:-/data/group_data/rl/geney/ppo_symbolic_policies/${RUN_NAME}}"
SAVE_POLICY_EVERY_STEPS="${SAVE_POLICY_EVERY_STEPS:-25000000}"
INIT_POLICY_PATH="${INIT_POLICY_PATH:-}"
TIMESTEP_OFFSET="${TIMESTEP_OFFSET:-0}"
mkdir -p "${POLICY_SAVE_DIR}"

CMD=(
    python -u online_rl/ppo.py
    --env_name Craftax-Symbolic-v1
    --total_timesteps "${TOTAL_TIMESTEPS}"
    --num_envs "${NUM_ENVS}"
    --num_steps "${NUM_STEPS}"
    --seed "${SEED}"
    --use_wandb
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
    --save_policy
    --policy_save_dir "${POLICY_SAVE_DIR}"
    --save_policy_every_steps "${SAVE_POLICY_EVERY_STEPS}"
    --save_policy_final
)

if [[ -n "${INIT_POLICY_PATH}" ]]; then
    CMD+=(--init_policy_path "${INIT_POLICY_PATH}")
fi
if [[ "${TIMESTEP_OFFSET}" != "0" ]]; then
    CMD+=(--timestep_offset "${TIMESTEP_OFFSET}")
fi

echo "Launching PPO symbolic policy run"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done: $(date)"
echo "Policy artifacts: ${POLICY_SAVE_DIR}"
