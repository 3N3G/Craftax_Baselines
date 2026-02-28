#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

POLICY_SAVE_DIR="${POLICY_SAVE_DIR:-/data/group_data/rl/geney/online_rl_hidden_models}"
TARGET_TIMESTEPS="${TARGET_TIMESTEPS:-300000000}"
ENVS="${ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-64}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-10000000}"
LAYER="${LAYER:-24}"
TOKENS="${TOKENS:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SAVE_TRAJ_ONLINE="${SAVE_TRAJ_ONLINE:-0}"
TRAJ_SAVE_EVERY_UPDATES="${TRAJ_SAVE_EVERY_UPDATES:-50}"
TRAJ_FREE_SPACE_MIN_GB="${TRAJ_FREE_SPACE_MIN_GB:-150}"
TRAJ_SCHEMA="${TRAJ_SCHEMA:-minimal_core}"

SKIP5_RESUME_FROM="${SKIP5_RESUME_FROM:-/data/group_data/rl/geney/online_rl_hidden_models/online-jax-128env-skip5-dualconcat-l24-100m-rerun4_resume_step000070000640_20260227_164805_120908.pkl}"
SKIP25_RESUME_FROM="${SKIP25_RESUME_FROM:-/data/group_data/rl/geney/online_rl_hidden_models/online-jax-128env-skip25-dualconcat-l24-100m-rerun4-part2_resume_step000149995520_20260226_215457_871492.pkl}"

submit_chain() {
    local skip="$1"
    local resume_from="$2"
    local run_name="$3"
    local checkpoint_dir="${POLICY_SAVE_DIR}/${run_name}_resume"
    local traj_dir=""
    if [[ "${SAVE_TRAJ_ONLINE}" == "1" ]]; then
        traj_dir="${POLICY_SAVE_DIR}/online_traj/${run_name}"
    fi

    ENVS="${ENVS}" \
    TARGET_TIMESTEPS="${TARGET_TIMESTEPS}" \
    SKIP_N="${skip}" \
    LAYER="${LAYER}" \
    TOKENS="${TOKENS}" \
    NUM_STEPS="${NUM_STEPS}" \
    CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS}" \
    POLICY_SAVE_DIR="${POLICY_SAVE_DIR}" \
    HIDDEN_POOLING="last_token" \
    HIDDEN_POOLING_K="1" \
    TEMPERATURE="${TEMPERATURE}" \
    SAVE_TRAJ_ONLINE="${SAVE_TRAJ_ONLINE}" \
    TRAJ_SAVE_DIR="${traj_dir}" \
    TRAJ_SAVE_EVERY_UPDATES="${TRAJ_SAVE_EVERY_UPDATES}" \
    TRAJ_FREE_SPACE_MIN_GB="${TRAJ_FREE_SPACE_MIN_GB}" \
    TRAJ_SCHEMA="${TRAJ_SCHEMA}" \
    RUN_NAME="${run_name}" \
    CHECKPOINT_DIR="${checkpoint_dir}" \
    RESUME_FROM="${resume_from}" \
    JOB_NAME="orlhj_s${skip}_300m" \
    bash scripts/shell/submit_online_rl_hidden_chain.sh
}

TS="$(date +%Y%m%d_%H%M%S)"
submit_chain 5 "${SKIP5_RESUME_FROM}" "online-jax-skip5-300m-${TS}"
submit_chain 25 "${SKIP25_RESUME_FROM}" "online-jax-skip25-300m-${TS}"
