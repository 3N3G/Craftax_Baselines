#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Defaults (override via flags or env vars)
PARTITION="${PARTITION:-general}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data/user_data/geney/.conda/envs/craftax_fast_llm}"
RUN_CONFIG_PATH="${RUN_CONFIG_PATH:-configs/future_imagination/run_config_predict_state_and_historyk5_key7_concise_traj_20260311_100410.json}"

GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"
GEMINI_API_ENV="${GEMINI_API_ENV:-GEMINI_API_KEY}"
GEMINI_MEM="${GEMINI_MEM:-24G}"
GEMINI_CPUS="${GEMINI_CPUS:-4}"
GEMINI_GPUS="${GEMINI_GPUS:-1}"

QWEN34_MODEL="${QWEN34_MODEL:-Qwen/Qwen3-4B}"
QWEN35_MODEL="${QWEN35_MODEL:-Qwen/Qwen3.5-9B}"
QWEN27_MODEL="${QWEN27_MODEL:-Qwen/Qwen3.5-27B}"
QWEN35A3B_MODEL="${QWEN35A3B_MODEL:-Qwen/Qwen3.5-35B-A3B}"

QWEN34_MEM="${QWEN34_MEM:-90G}"
QWEN35_MEM="${QWEN35_MEM:-120G}"
QWEN27_MEM="${QWEN27_MEM:-220G}"
QWEN35A3B_MEM="${QWEN35A3B_MEM:-240G}"

QWEN34_GPUS="${QWEN34_GPUS:-1}"
QWEN35_GPUS="${QWEN35_GPUS:-1}"
QWEN27_GPUS="${QWEN27_GPUS:-2}"
QWEN35A3B_GPUS="${QWEN35A3B_GPUS:-2}"

QWEN34_CPUS="${QWEN34_CPUS:-8}"
QWEN35_CPUS="${QWEN35_CPUS:-8}"
QWEN27_CPUS="${QWEN27_CPUS:-16}"
QWEN35A3B_CPUS="${QWEN35A3B_CPUS:-16}"

ORACLE_KEY5_DIR="${ORACLE_KEY5_DIR:-analysis/future_imagination/20260315_233711_traj_20260311_100410_oracle_key5_stride2_promptv2_gemini3flashpreview}"
ORACLE_ADDITIONAL_DIR="${ORACLE_ADDITIONAL_DIR:-analysis/future_imagination/20260315_233831_traj_20260311_100410_oracle_additional_promptv2_gemini3flashpreview}"
GEMINI_PRED_DIR="${GEMINI_PRED_DIR:-analysis/future_imagination/20260315_233921_traj_20260311_100410_predict_stateplushistoryk5_key7_promptv2_gemini3flashpreview}"
QWEN34_EXISTING_DIR="${QWEN34_EXISTING_DIR:-analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6}"
QWEN35_EXISTING_DIR="${QWEN35_EXISTING_DIR:-analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3.5-9b-p6}"
QWEN27_EXISTING_DIR="${QWEN27_EXISTING_DIR:-analysis/future_imagination/20260318_021455_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3.5-27b-p6plus-g2}"
QWEN35A3B_EXISTING_DIR="${QWEN35A3B_EXISTING_DIR:-analysis/future_imagination/20260318_022205_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3.5-35b-a3b-p6plus-g2}"
TRAJ_DIR="${TRAJ_DIR:-play_data/trajectory_records/traj_20260311_100410}"
FRAME_DIR="${FRAME_DIR:-play_data/trajectory_records/traj_20260311_100410/render_frames_bs16}"

LOCAL_PYTHON="${LOCAL_PYTHON:-/Users/gene/anaconda3/envs/imaug/bin/python}"

SYNC_PATHS=(
  "configs/future_imagination/templates/predict_state_only_prompt_concise.txt"
  "configs/future_imagination/templates/predict_history_k_prompt_concise.txt"
  "configs/future_imagination/templates/predict_state_only_prompt.txt"
  "configs/future_imagination/templates/predict_history_k_prompt.txt"
  "scripts/shell/run_future_imagination_gemini_eval.sh"
  "scripts/shell/run_future_imagination_hf_local_eval.sh"
)

TAG=""
OVERVIEW_TEXT=""
WAIT_FOR_COMPLETION=1
BUILD_COMBINED_REPORT=1
RUN_GEMINI=0
RUN_QWEN34=1
RUN_QWEN35=1
RUN_QWEN27=0
RUN_QWEN35A3B=0

usage() {
  cat <<USAGE
Usage:
  scripts/shell/run_future_prompt_iteration_wave.sh --tag <tag> [options]

Options:
  --tag <tag>                 Required. Suffix for job/model tags and output folder labels.
  --overview-text <text>      If set, replaces the single line immediately after 'Craftax overview:' in all 4 prediction templates.
  --run-config <path>         Run config path on Babel. Default: ${RUN_CONFIG_PATH}
  --run-gemini                Re-run Gemini prediction outputs and use them in the combined report.
  --skip-qwen34               Do not re-run Qwen3-4B.
  --skip-qwen35               Do not re-run Qwen3.5-9B.
  --run-qwen27                Re-run Qwen3.5-27B and use it in the combined report.
  --run-qwen35a3b             Re-run Qwen3.5-35B-A3B and use it in the combined report.
  --no-wait                   Submit jobs and exit without waiting/pulling/report rebuild.
  --no-report                 Skip combined report rebuild (still submits/runs/pulls by default).
  --partition <name>          Slurm partition for jobs. Default: ${PARTITION}
  --time-limit <HH:MM:SS>     Slurm time limit. Default: ${TIME_LIMIT}
  -h, --help                  Show this help.

Environment overrides:
  CONDA_ENV_PATH, GEMINI_MODEL, GEMINI_API_ENV, GEMINI_MEM, GEMINI_CPUS, GEMINI_GPUS,
  QWEN34_MODEL, QWEN35_MODEL, QWEN27_MODEL, QWEN35A3B_MODEL,
  QWEN34_MEM, QWEN35_MEM, QWEN27_MEM, QWEN35A3B_MEM,
  QWEN34_GPUS, QWEN35_GPUS, QWEN27_GPUS, QWEN35A3B_GPUS,
  QWEN34_CPUS, QWEN35_CPUS, QWEN27_CPUS, QWEN35A3B_CPUS,
  ORACLE_KEY5_DIR, ORACLE_ADDITIONAL_DIR, GEMINI_PRED_DIR,
  QWEN34_EXISTING_DIR, QWEN35_EXISTING_DIR, QWEN27_EXISTING_DIR, QWEN35A3B_EXISTING_DIR,
  TRAJ_DIR, FRAME_DIR, LOCAL_PYTHON
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --overview-text)
      OVERVIEW_TEXT="${2:-}"
      shift 2
      ;;
    --run-config)
      RUN_CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --run-gemini)
      RUN_GEMINI=1
      shift
      ;;
    --skip-qwen34)
      RUN_QWEN34=0
      shift
      ;;
    --skip-qwen35)
      RUN_QWEN35=0
      shift
      ;;
    --run-qwen27)
      RUN_QWEN27=1
      shift
      ;;
    --run-qwen35a3b)
      RUN_QWEN35A3B=1
      shift
      ;;
    --no-wait)
      WAIT_FOR_COMPLETION=0
      shift
      ;;
    --no-report)
      BUILD_COMBINED_REPORT=0
      shift
      ;;
    --partition)
      PARTITION="${2:-}"
      shift 2
      ;;
    --time-limit)
      TIME_LIMIT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${TAG}" ]]; then
  echo "ERROR: --tag is required"
  usage
  exit 2
fi

if [[ ! -x "${LOCAL_PYTHON}" ]]; then
  LOCAL_PYTHON="python3"
fi

if [[ "${RUN_GEMINI}" == "1" && -z "${!GEMINI_API_ENV:-}" ]]; then
  echo "ERROR: ${GEMINI_API_ENV} must be set locally when --run-gemini is enabled"
  exit 1
fi

set_overview_line() {
  local file_path="$1"
  local replacement="$2"
  local tmp
  tmp="$(mktemp)"
  awk -v replacement="${replacement}" '
    BEGIN { updated = 0 }
    {
      if ($0 ~ /^Craftax overview:[[:space:]]*$/) {
        print $0
        if (getline next_line) {
          print replacement
          updated = 1
          next
        }
      }
      print $0
    }
    END {
      if (updated == 0) {
        exit 3
      }
    }
  ' "${file_path}" > "${tmp}" || {
    rm -f "${tmp}"
    echo "ERROR: failed updating overview in ${file_path}"
    return 1
  }
  mv "${tmp}" "${file_path}"
}

extract_job_id() {
  local submit_out="$1"
  echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1
}

submit_hf_job() {
  local model_id="$1"
  local model_tag="$2"
  local job_name="$3"
  local mem="$4"
  local gpus="$5"
  local cpus="$6"
  local install_main="$7"
  local log_prefix="$8"

  local wrap_cmd
  wrap_cmd="source ~/.bashrc && conda activate ${CONDA_ENV_PATH} && "
  if [[ "${install_main}" == "1" ]]; then
    wrap_cmd+="INSTALL_TRANSFORMERS_MAIN=1 "
  fi
  wrap_cmd+="RUN_CONFIG_PATH=${RUN_CONFIG_PATH} scripts/shell/run_future_imagination_hf_local_eval.sh ${model_id} ${model_tag}"

  ssh babel "cd ~/Craftax_Baselines && sbatch --job-name ${job_name} --partition ${PARTITION} --gres=gpu:${gpus} --cpus-per-task=${cpus} --mem=${mem} --time=${TIME_LIMIT} --output logs/${log_prefix}_%j.out --error logs/${log_prefix}_%j.err --wrap 'bash -lc \"${wrap_cmd}\"'"
}

submit_gemini_job() {
  local model_id="$1"
  local model_tag="$2"
  local job_name="$3"
  local mem="$4"
  local cpus="$5"
  local gpus="$6"
  local log_prefix="$7"
  local api_value="${!GEMINI_API_ENV:-}"
  local wrap_cmd

  wrap_cmd="source ~/.bashrc && conda activate ${CONDA_ENV_PATH} && RUN_CONFIG_PATH=${RUN_CONFIG_PATH} API_KEY_ENV=${GEMINI_API_ENV} scripts/shell/run_future_imagination_gemini_eval.sh ${model_id} ${model_tag}"

  ssh babel "cd ~/Craftax_Baselines && sbatch --job-name ${job_name} --partition ${PARTITION} --gres=gpu:${gpus} --cpus-per-task=${cpus} --mem=${mem} --time=${TIME_LIMIT} --output logs/${log_prefix}_%j.out --error logs/${log_prefix}_%j.err --export=ALL,${GEMINI_API_ENV}=${api_value} --wrap 'bash -lc \"${wrap_cmd}\"'"
}

extract_out_dir() {
  local log_file="$1"
  local out_dir
  out_dir="$(grep -m1 '^\[info\] out_dir=' "${log_file}" | sed 's/^\[info\] out_dir=//' || true)"
  if [[ -z "${out_dir}" ]]; then
    out_dir="$(grep -m1 '^output_dir:' "${log_file}" | sed 's/^output_dir:[[:space:]]*//; s|^/home/geney/Craftax_Baselines/||' || true)"
  fi
  echo "${out_dir}"
}

require_local_dir() {
  local dir_path="$1"
  if [[ -z "${dir_path}" || ! -d "${dir_path}" ]]; then
    echo "ERROR: required directory missing: ${dir_path}"
    exit 1
  fi
}

if [[ -n "${OVERVIEW_TEXT}" ]]; then
  echo "[local] updating Craftax overview line in templates"
  for f in "${SYNC_PATHS[@]:0:4}"; do
    set_overview_line "${f}" "${OVERVIEW_TEXT}"
    echo "  updated: ${f}"
  done
fi

chmod +x scripts/shell/run_future_imagination_gemini_eval.sh

echo "[local] pushing prompt/template helper files to Babel"
"${ROOT_DIR}/scripts/shell/babel.sh" push "${SYNC_PATHS[@]}"

GEMINI_MODEL_TAG="gemini3flashpreview-${TAG}"
QWEN34_MODEL_TAG="qwen3-4b-${TAG}"
QWEN35_MODEL_TAG="qwen3.5-9b-${TAG}"
QWEN27_MODEL_TAG="qwen3.5-27b-${TAG}"
QWEN35A3B_MODEL_TAG="qwen3.5-35b-a3b-${TAG}"

GEMINI_JOB_NAME="futimg_gemini_${TAG}"
QWEN34_JOB_NAME="futimg_qwen34b_hf_${TAG}"
QWEN35_JOB_NAME="futimg_qwen359b_hf_${TAG}"
QWEN27_JOB_NAME="futimg_qwen3527b_hf_${TAG}"
QWEN35A3B_JOB_NAME="futimg_qwen3535a3b_hf_${TAG}"

GEMINI_LOG_PREFIX="future_imag_gemini_${TAG}"
QWEN34_LOG_PREFIX="future_imag_qwen34b_hf_${TAG}"
QWEN35_LOG_PREFIX="future_imag_qwen359b_hf_${TAG}"
QWEN27_LOG_PREFIX="future_imag_qwen3527b_hf_${TAG}"
QWEN35A3B_LOG_PREFIX="future_imag_qwen3535a3b_hf_${TAG}"

GEMINI_JOB_ID=""
QWEN34_JOB_ID=""
QWEN35_JOB_ID=""
QWEN27_JOB_ID=""
QWEN35A3B_JOB_ID=""

if [[ "${RUN_GEMINI}" == "1" ]]; then
  echo "[remote] submitting Gemini prediction job"
  SUBMIT_OUT="$(submit_gemini_job "${GEMINI_MODEL}" "${GEMINI_MODEL_TAG}" "${GEMINI_JOB_NAME}" "${GEMINI_MEM}" "${GEMINI_CPUS}" "${GEMINI_GPUS}" "${GEMINI_LOG_PREFIX}")"
  GEMINI_JOB_ID="$(extract_job_id "${SUBMIT_OUT}")"
  if [[ -z "${GEMINI_JOB_ID}" ]]; then
    echo "ERROR: failed to parse gemini job id"
    echo "${SUBMIT_OUT}"
    exit 1
  fi
  echo "  job_id=${GEMINI_JOB_ID}"
fi

if [[ "${RUN_QWEN34}" == "1" ]]; then
  echo "[remote] submitting Qwen3-4B job"
  SUBMIT_OUT="$(submit_hf_job "${QWEN34_MODEL}" "${QWEN34_MODEL_TAG}" "${QWEN34_JOB_NAME}" "${QWEN34_MEM}" "${QWEN34_GPUS}" "${QWEN34_CPUS}" "0" "${QWEN34_LOG_PREFIX}")"
  QWEN34_JOB_ID="$(extract_job_id "${SUBMIT_OUT}")"
  if [[ -z "${QWEN34_JOB_ID}" ]]; then
    echo "ERROR: failed to parse qwen34 job id"
    echo "${SUBMIT_OUT}"
    exit 1
  fi
  echo "  job_id=${QWEN34_JOB_ID}"
fi

if [[ "${RUN_QWEN35}" == "1" ]]; then
  echo "[remote] submitting Qwen3.5-9B job"
  SUBMIT_OUT="$(submit_hf_job "${QWEN35_MODEL}" "${QWEN35_MODEL_TAG}" "${QWEN35_JOB_NAME}" "${QWEN35_MEM}" "${QWEN35_GPUS}" "${QWEN35_CPUS}" "1" "${QWEN35_LOG_PREFIX}")"
  QWEN35_JOB_ID="$(extract_job_id "${SUBMIT_OUT}")"
  if [[ -z "${QWEN35_JOB_ID}" ]]; then
    echo "ERROR: failed to parse qwen35 job id"
    echo "${SUBMIT_OUT}"
    exit 1
  fi
  echo "  job_id=${QWEN35_JOB_ID}"
fi

if [[ "${RUN_QWEN27}" == "1" ]]; then
  echo "[remote] submitting Qwen3.5-27B job"
  SUBMIT_OUT="$(submit_hf_job "${QWEN27_MODEL}" "${QWEN27_MODEL_TAG}" "${QWEN27_JOB_NAME}" "${QWEN27_MEM}" "${QWEN27_GPUS}" "${QWEN27_CPUS}" "1" "${QWEN27_LOG_PREFIX}")"
  QWEN27_JOB_ID="$(extract_job_id "${SUBMIT_OUT}")"
  if [[ -z "${QWEN27_JOB_ID}" ]]; then
    echo "ERROR: failed to parse qwen27 job id"
    echo "${SUBMIT_OUT}"
    exit 1
  fi
  echo "  job_id=${QWEN27_JOB_ID}"
fi

if [[ "${RUN_QWEN35A3B}" == "1" ]]; then
  echo "[remote] submitting Qwen3.5-35B-A3B job"
  SUBMIT_OUT="$(submit_hf_job "${QWEN35A3B_MODEL}" "${QWEN35A3B_MODEL_TAG}" "${QWEN35A3B_JOB_NAME}" "${QWEN35A3B_MEM}" "${QWEN35A3B_GPUS}" "${QWEN35A3B_CPUS}" "1" "${QWEN35A3B_LOG_PREFIX}")"
  QWEN35A3B_JOB_ID="$(extract_job_id "${SUBMIT_OUT}")"
  if [[ -z "${QWEN35A3B_JOB_ID}" ]]; then
    echo "ERROR: failed to parse qwen35a3b job id"
    echo "${SUBMIT_OUT}"
    exit 1
  fi
  echo "  job_id=${QWEN35A3B_JOB_ID}"
fi

JOB_IDS=()
for jid in "${GEMINI_JOB_ID}" "${QWEN34_JOB_ID}" "${QWEN35_JOB_ID}" "${QWEN27_JOB_ID}" "${QWEN35A3B_JOB_ID}"; do
  if [[ -n "${jid}" ]]; then
    JOB_IDS+=("${jid}")
  fi
done

if [[ "${WAIT_FOR_COMPLETION}" != "1" ]]; then
  echo "[done] submitted jobs only"
  if [[ -n "${GEMINI_JOB_ID}" ]]; then
    echo "  gemini_job=${GEMINI_JOB_ID}"
  fi
  if [[ -n "${QWEN34_JOB_ID}" ]]; then
    echo "  qwen34_job=${QWEN34_JOB_ID}"
  fi
  if [[ -n "${QWEN35_JOB_ID}" ]]; then
    echo "  qwen35_job=${QWEN35_JOB_ID}"
  fi
  if [[ -n "${QWEN27_JOB_ID}" ]]; then
    echo "  qwen27_job=${QWEN27_JOB_ID}"
  fi
  if [[ -n "${QWEN35A3B_JOB_ID}" ]]; then
    echo "  qwen35a3b_job=${QWEN35A3B_JOB_ID}"
  fi
  exit 0
fi

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
  JOB_IDS_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"
  echo "[remote] waiting for jobs: ${JOB_IDS_CSV}"
  while true; do
    ACTIVE="$(ssh babel "squeue -h -j ${JOB_IDS_CSV} -o '%i %T %M'" || true)"
    if [[ -z "${ACTIVE}" ]]; then
      break
    fi
    TS="$(date +%H:%M:%S)"
    echo "[wait ${TS}] ${ACTIVE//$'\n'/ | }"
    sleep 20
  done

  echo "[remote] final states"
  ssh babel "sacct -j ${JOB_IDS_CSV} --format=JobID,State,ExitCode,Elapsed,NodeList -P -n"

  echo "[local] pulling logs"
  for jid in "${JOB_IDS[@]}"; do
    "${ROOT_DIR}/scripts/shell/babel.sh" logs "${jid}"
  done
fi

GEMINI_RUN_DIR=""
QWEN34_RUN_DIR=""
QWEN35_RUN_DIR=""
QWEN27_RUN_DIR=""
QWEN35A3B_RUN_DIR=""

if [[ -n "${GEMINI_JOB_ID}" ]]; then
  LOG_FILE="logs/${GEMINI_LOG_PREFIX}_${GEMINI_JOB_ID}.out"
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "ERROR: missing local log ${LOG_FILE}"
    exit 1
  fi
  GEMINI_RUN_DIR="$(extract_out_dir "${LOG_FILE}")"
  if [[ -z "${GEMINI_RUN_DIR}" ]]; then
    echo "ERROR: failed to parse gemini out_dir from ${LOG_FILE}"
    exit 1
  fi
fi

if [[ -n "${QWEN34_JOB_ID}" ]]; then
  LOG_FILE="logs/${QWEN34_LOG_PREFIX}_${QWEN34_JOB_ID}.out"
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "ERROR: missing local log ${LOG_FILE}"
    exit 1
  fi
  QWEN34_RUN_DIR="$(extract_out_dir "${LOG_FILE}")"
  if [[ -z "${QWEN34_RUN_DIR}" ]]; then
    echo "ERROR: failed to parse qwen34 out_dir from ${LOG_FILE}"
    exit 1
  fi
fi

if [[ -n "${QWEN35_JOB_ID}" ]]; then
  LOG_FILE="logs/${QWEN35_LOG_PREFIX}_${QWEN35_JOB_ID}.out"
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "ERROR: missing local log ${LOG_FILE}"
    exit 1
  fi
  QWEN35_RUN_DIR="$(extract_out_dir "${LOG_FILE}")"
  if [[ -z "${QWEN35_RUN_DIR}" ]]; then
    echo "ERROR: failed to parse qwen35 out_dir from ${LOG_FILE}"
    exit 1
  fi
fi

if [[ -n "${QWEN27_JOB_ID}" ]]; then
  LOG_FILE="logs/${QWEN27_LOG_PREFIX}_${QWEN27_JOB_ID}.out"
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "ERROR: missing local log ${LOG_FILE}"
    exit 1
  fi
  QWEN27_RUN_DIR="$(extract_out_dir "${LOG_FILE}")"
  if [[ -z "${QWEN27_RUN_DIR}" ]]; then
    echo "ERROR: failed to parse qwen27 out_dir from ${LOG_FILE}"
    exit 1
  fi
fi

if [[ -n "${QWEN35A3B_JOB_ID}" ]]; then
  LOG_FILE="logs/${QWEN35A3B_LOG_PREFIX}_${QWEN35A3B_JOB_ID}.out"
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "ERROR: missing local log ${LOG_FILE}"
    exit 1
  fi
  QWEN35A3B_RUN_DIR="$(extract_out_dir "${LOG_FILE}")"
  if [[ -z "${QWEN35A3B_RUN_DIR}" ]]; then
    echo "ERROR: failed to parse qwen35a3b out_dir from ${LOG_FILE}"
    exit 1
  fi
fi

RUN_DIRS_TO_PULL=()
for run_dir in "${GEMINI_RUN_DIR}" "${QWEN34_RUN_DIR}" "${QWEN35_RUN_DIR}" "${QWEN27_RUN_DIR}" "${QWEN35A3B_RUN_DIR}"; do
  if [[ -n "${run_dir}" ]]; then
    RUN_DIRS_TO_PULL+=("${run_dir}")
  fi
done

if [[ ${#RUN_DIRS_TO_PULL[@]} -gt 0 ]]; then
  echo "[local] pulling run dirs"
  "${ROOT_DIR}/scripts/shell/babel.sh" pull "${RUN_DIRS_TO_PULL[@]}"
fi

GEMINI_REPORT_DIR="${GEMINI_PRED_DIR}"
QWEN34_REPORT_DIR="${QWEN34_EXISTING_DIR}"
QWEN35_REPORT_DIR="${QWEN35_EXISTING_DIR}"
QWEN27_REPORT_DIR="${QWEN27_EXISTING_DIR}"
QWEN35A3B_REPORT_DIR="${QWEN35A3B_EXISTING_DIR}"

if [[ -n "${GEMINI_RUN_DIR}" ]]; then
  GEMINI_REPORT_DIR="${GEMINI_RUN_DIR}"
fi
if [[ -n "${QWEN34_RUN_DIR}" ]]; then
  QWEN34_REPORT_DIR="${QWEN34_RUN_DIR}"
fi
if [[ -n "${QWEN35_RUN_DIR}" ]]; then
  QWEN35_REPORT_DIR="${QWEN35_RUN_DIR}"
fi
if [[ -n "${QWEN27_RUN_DIR}" ]]; then
  QWEN27_REPORT_DIR="${QWEN27_RUN_DIR}"
fi
if [[ -n "${QWEN35A3B_RUN_DIR}" ]]; then
  QWEN35A3B_REPORT_DIR="${QWEN35A3B_RUN_DIR}"
fi

if [[ "${BUILD_COMBINED_REPORT}" != "1" ]]; then
  echo "[done] reruns completed"
  echo "  gemini_run_dir=${GEMINI_REPORT_DIR}"
  echo "  qwen34_run_dir=${QWEN34_REPORT_DIR}"
  echo "  qwen35_run_dir=${QWEN35_REPORT_DIR}"
  echo "  qwen27_run_dir=${QWEN27_REPORT_DIR}"
  echo "  qwen35a3b_run_dir=${QWEN35A3B_REPORT_DIR}"
  exit 0
fi

require_local_dir "${ORACLE_KEY5_DIR}"
require_local_dir "${ORACLE_ADDITIONAL_DIR}"
require_local_dir "${GEMINI_REPORT_DIR}"
require_local_dir "${QWEN34_REPORT_DIR}"
require_local_dir "${QWEN35_REPORT_DIR}"
require_local_dir "${QWEN27_REPORT_DIR}"
require_local_dir "${QWEN35A3B_REPORT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
COMBINED_OUT="analysis/future_imagination/oracle_plus_predict_state_history_key7_unabridged_tb0_viewer_${TAG}_${TS}"

REPORT_CMD=(
  "${LOCAL_PYTHON}"
  "scripts/render_future_combined_report.py"
  --run-dir "${ORACLE_KEY5_DIR}"
  --run-dir "${ORACLE_ADDITIONAL_DIR}"
  --run-dir "${GEMINI_REPORT_DIR}"
  --run-dir "${QWEN34_REPORT_DIR}"
  --run-dir "${QWEN35_REPORT_DIR}"
  --run-dir "${QWEN27_REPORT_DIR}"
  --run-dir "${QWEN35A3B_REPORT_DIR}"
  --trajectory-dir "${TRAJ_DIR}"
  --viewer-frame-dir "${FRAME_DIR}"
  --output-dir "${COMBINED_OUT}"
)

echo "[local] rebuilding combined report"
"${REPORT_CMD[@]}"

echo "[done]"
echo "  gemini_run_dir=${GEMINI_REPORT_DIR}"
echo "  qwen34_run_dir=${QWEN34_REPORT_DIR}"
echo "  qwen35_run_dir=${QWEN35_REPORT_DIR}"
echo "  qwen27_run_dir=${QWEN27_REPORT_DIR}"
echo "  qwen35a3b_run_dir=${QWEN35A3B_REPORT_DIR}"
echo "  combined_report_dir=${COMBINED_OUT}"
