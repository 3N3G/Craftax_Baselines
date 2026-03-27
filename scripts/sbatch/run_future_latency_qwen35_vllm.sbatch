#!/bin/bash
#SBATCH --job-name=latbench_q35_vllm
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/latbench_q35_vllm_%j.out
#SBATCH --error=logs/latbench_q35_vllm_%j.err

set -eo pipefail
if [[ "${DEBUG_LATBENCH_SBATCH:-0}" == "1" ]]; then
  set -x
fi

cd ~/Craftax_Baselines
source ~/.bashrc >/dev/null 2>&1 || true

USER_NAME="${USER:-${LOGNAME:-geney}}"
BENCH_ENV_PATH="${BENCH_ENV_PATH:-/data/user_data/${USER_NAME}/.conda/envs/craftax_fast_llm}"
if [[ ! -d "${BENCH_ENV_PATH}" ]]; then
  echo "ERROR: benchmark env missing: ${BENCH_ENV_PATH}"
  exit 2
fi

export PATH="${BENCH_ENV_PATH}/bin:${PATH}"

if [[ -z "${TMP_ROOT:-}" ]]; then
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    TMP_ROOT="${SLURM_TMPDIR}/latbench_${SLURM_JOB_ID}"
  else
    TMP_ROOT="/tmp/latbench_${SLURM_JOB_ID}"
  fi
fi
if ! mkdir -p "${TMP_ROOT}"; then
  TMP_ROOT="/scratch/${USER_NAME}/latbench_${SLURM_JOB_ID}"
  mkdir -p "${TMP_ROOT}"
fi
export TMP_ROOT

echo "[info] script_started=1"
echo "[info] host=$(hostname)"
echo "[info] user=${USER_NAME}"
echo "[info] bench_env=${BENCH_ENV_PATH}"
echo "[info] tmp_root=${TMP_ROOT}"
echo "[info] run_tag=${RUN_TAG:-latbench_q35_vllm}"
echo "[info] qwen35_backend=${QWEN35_BACKEND:-vllm_hidden}"
echo "[info] run_qwen9b=${RUN_QWEN9B:-1}"
echo "[info] run_qwen27b=${RUN_QWEN27B:-1}"
echo "[info] prompt_state_dir=${PROMPT_STATE_DIR:-analysis/future_imagination_latency_bench/prompts_single/predict_state_only}"
echo "[info] prompt_history_dir=${PROMPT_HISTORY_DIR:-analysis/future_imagination_latency_bench/prompts_single/predict_history_k5}"

RUN_TAG="${RUN_TAG:-latbench_q35_vllm}" \
QWEN35_BACKEND="${QWEN35_BACKEND:-vllm_hidden}" \
RUN_QWEN4B="${RUN_QWEN4B:-0}" \
RUN_QWEN14B="${RUN_QWEN14B:-0}" \
RUN_QWEN9B="${RUN_QWEN9B:-1}" \
RUN_QWEN27B="${RUN_QWEN27B:-1}" \
REPEATS="${REPEATS:-1}" \
MAX_TOKENS="${MAX_TOKENS:-256}" \
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}" \
PROMPT_STATE_DIR="${PROMPT_STATE_DIR:-analysis/future_imagination_latency_bench/prompts_single/predict_state_only}" \
PROMPT_HISTORY_DIR="${PROMPT_HISTORY_DIR:-analysis/future_imagination_latency_bench/prompts_single/predict_history_k5}" \
REQUIRE_NONPRIMARY_ENV_FOR_QWEN35="${REQUIRE_NONPRIMARY_ENV_FOR_QWEN35:-0}" \
GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
bash scripts/shell/run_future_latency_benchmark_models.sh

echo "[done] qwen3.5 vllm latency benchmark finished"
