#!/bin/bash
#SBATCH --job-name=online_rl_sps_diag
#SBATCH --partition=rl
#SBATCH --gres=gpu:RTX_PRO_6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/online_rl_sps_diag_%j.out
#SBATCH --error=logs/online_rl_sps_diag_%j.err

# End-to-end diagnosis for low SPS in online RL + LLM.
# Runs:
#   1) no-LLM baseline
#   2) hidden-state extractor microbenchmark
#   3) LLM skip_n sweep (1,4,16,64 by default)

set -eo pipefail

# Some cluster bashrc/profile.d scripts return non-zero under strict mode
# (e.g. missing optional files), which can kill this batch script before logging.
# Source bashrc with strict flags disabled, then restore strict mode.
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

RUN_DIR="logs/diag_online_rl_sps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

ENVS=${1:-8}
NUM_STEPS=${2:-16}
TIMESTEPS_NO_LLM=${3:-5120}
TIMESTEPS_LLM=${4:-2560}
LAYER=${5:--1}
TOKENS=${6:-1}
SKIP_CSV=${7:-1,4,16,64}
EXTRACT_ITERS=${8:-20}
EXTRACT_WARMUP=${9:-3}

IFS=',' read -r -a SKIPS <<< "${SKIP_CSV}"

echo "======================================================================"
echo "Online RL SPS Diagnostics"
echo "job_id=${SLURM_JOB_ID}"
echo "run_dir=${RUN_DIR}"
echo "host=$(hostname)"
echo "gpu=$(nvidia-smi --list-gpus | head -1)"
echo "params: envs=${ENVS} num_steps=${NUM_STEPS} timesteps_no_llm=${TIMESTEPS_NO_LLM} timesteps_llm=${TIMESTEPS_LLM}"
echo "params: layer=${LAYER} tokens=${TOKENS} skips=${SKIP_CSV}"
echo "======================================================================"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found after sourcing bashrc"
    exit 1
fi

activate_conda_for() {
    local purpose="$1"
    shift
    local env_name
    for env_name in "$@"; do
        if conda activate "${env_name}" >/dev/null 2>&1; then
            echo "Activated conda env for ${purpose}: ${env_name}"
            return 0
        fi
    done
    echo "ERROR: failed to activate conda env for ${purpose}"
    echo "Tried: $*"
    conda env list || true
    return 1
}

VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        echo "Stopping vLLM server (pid=${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo ""
echo "[1/4] No-LLM baseline (no vLLM server)"
activate_conda_for "training(no-llm)" \
    /data/user_data/geney/.conda/envs/craftax \
    craftax \
    base
python -u online_rl_llm/online_rl_hidden_jax.py \
    --no-llm \
    --envs "${ENVS}" \
    --num-steps "${NUM_STEPS}" \
    --timesteps "${TIMESTEPS_NO_LLM}" \
    --no-wandb | tee "${RUN_DIR}/00_no_llm.log"

echo ""
echo "[2/4] Start vLLM server"
activate_conda_for "vllm-server" \
    /data/user_data/geney/.conda/envs/craftax_fast_llm \
    craftax_fast_llm \
    base
bash scripts/start_vllm_hidden.sh --mode last_token > "${RUN_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!
echo "vLLM pid=${VLLM_PID}"

for i in {1..300}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM process exited early"
        tail -50 "${RUN_DIR}/vllm_server.log" || true
        exit 1
    fi
    sleep 1
    if [[ "${i}" == "300" ]]; then
        echo "ERROR: vLLM failed to become healthy in 300s"
        tail -50 "${RUN_DIR}/vllm_server.log" || true
        exit 1
    fi
done

echo ""
echo "[3/4] Hidden extractor microbenchmark"
activate_conda_for "training(extractor)" \
    /data/user_data/geney/.conda/envs/craftax \
    craftax \
    base
python -u scripts/diagnostics/bench_hidden_extractor.py \
    --envs "${ENVS}" \
    --iters "${EXTRACT_ITERS}" \
    --warmup "${EXTRACT_WARMUP}" \
    --tokens "${TOKENS}" \
    --layer "${LAYER}" | tee "${RUN_DIR}/10_extractor.log"

echo ""
echo "[4/4] LLM skip_n sweep"
for skip in "${SKIPS[@]}"; do
    log_file="${RUN_DIR}/20_skip${skip}.log"
    echo "Running skip_n=${skip} -> ${log_file}"
    python -u online_rl_llm/online_rl_hidden_jax.py \
        --envs "${ENVS}" \
        --num-steps "${NUM_STEPS}" \
        --timesteps "${TIMESTEPS_LLM}" \
        --skip-n "${skip}" \
        --layer "${LAYER}" \
        --tokens "${TOKENS}" \
        --no-wandb | tee "${log_file}"
done

echo ""
echo "[5/5] Parse summary"
python -u scripts/diagnostics/parse_online_rl_sps_diag.py "${RUN_DIR}" | tee "${RUN_DIR}/99_summary.log"

echo ""
echo "Diagnostics completed."
echo "Summary: ${RUN_DIR}/99_summary.log"
