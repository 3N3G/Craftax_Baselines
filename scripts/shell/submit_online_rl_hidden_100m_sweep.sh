#!/bin/bash
set -euo pipefail

# Submit 100M-step online RL + LLM jobs at skip_n in {1,5,25,125}.
# Run this on babel login node from repo root:
#   bash scripts/shell/submit_online_rl_hidden_100m_sweep.sh

ENVS="${ENVS:-8}"
TIMESTEPS="${TIMESTEPS:-1e8}"
LAYER="${LAYER:--1}"
TOKENS="${TOKENS:-1}"
SKIPS=(1 5 25 125)

echo "Submitting online_rl_hidden_jax sweep"
echo "  envs=${ENVS} timesteps=${TIMESTEPS} layer=${LAYER} tokens=${TOKENS}"
echo "  skips=${SKIPS[*]}"
echo ""

for skip in "${SKIPS[@]}"; do
    job_name="orlhj_s${skip}"
    out_log="logs/online_rl_hidden_jax_skip${skip}_%j.out"
    err_log="logs/online_rl_hidden_jax_skip${skip}_%j.err"
    submit_out=$(
        sbatch \
            --job-name="${job_name}" \
            --output="${out_log}" \
            --error="${err_log}" \
            scripts/sbatch/run_online_rl_hidden_jax.sbatch \
            "${ENVS}" "${TIMESTEPS}" "${skip}" "${LAYER}" "${TOKENS}"
    )
    job_id="$(echo "${submit_out}" | awk '{print $4}')"
    echo "skip=${skip} job_id=${job_id} name=${job_name}"
    echo "  out=${out_log}"
    echo "  err=${err_log}"
done

echo ""
echo "Queue snapshot:"
squeue -u "${USER}" -o "%.18i %.9P %.20j %.8T %.10M %.10l %.6D %R" | grep -E "JOBID|orlhj_s"
