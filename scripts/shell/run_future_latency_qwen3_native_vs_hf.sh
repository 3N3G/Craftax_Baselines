#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-qwen_native_vs_hf_fastpath}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="analysis/future_imagination_latency_bench/${TS}_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

PROMPT_ROOT="${PROMPT_ROOT:-analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts}"
PROMPT_STATE_DIR="${PROMPT_STATE_DIR:-${PROMPT_ROOT}/predict_state_only}"
PROMPT_HISTORY_DIR="${PROMPT_HISTORY_DIR:-${PROMPT_ROOT}/predict_history_k5}"

MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.95}"
REPEATS="${REPEATS:-1}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-7168}"
SAFE_MODE="${SAFE_MODE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
BENCHMARK_MODE="${BENCHMARK_MODE:-hidden_only}"
WARMUP_PROMPTS_PER_VARIANT="${WARMUP_PROMPTS_PER_VARIANT:-1}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"

QWEN4B_MODEL_ID="${QWEN4B_MODEL_ID:-Qwen/Qwen3-4B}"
QWEN8B_MODEL_ID="${QWEN8B_MODEL_ID:-Qwen/Qwen3-8B}"
QWEN14B_MODEL_ID="${QWEN14B_MODEL_ID:-Qwen/Qwen3-14B}"
QWEN9B_MODEL_ID="${QWEN9B_MODEL_ID:-Qwen/Qwen3.5-9B}"
QWEN27B_MODEL_ID="${QWEN27B_MODEL_ID:-Qwen/Qwen3.5-27B}"
QWEN35_COMPAT_ROOT="${QWEN35_COMPAT_ROOT:-${OUT_DIR}/qwen35_compat}"
QWEN9B_COMPAT_MODEL_DIR="${QWEN9B_COMPAT_MODEL_DIR:-${QWEN35_COMPAT_ROOT}/qwen3_5_9b}"
QWEN27B_COMPAT_MODEL_DIR="${QWEN27B_COMPAT_MODEL_DIR:-${QWEN35_COMPAT_ROOT}/qwen3_5_27b}"

RUN_QWEN4B="${RUN_QWEN4B:-1}"
RUN_QWEN8B="${RUN_QWEN8B:-1}"
RUN_QWEN14B="${RUN_QWEN14B:-1}"
RUN_QWEN9B="${RUN_QWEN9B:-1}"
RUN_QWEN27B="${RUN_QWEN27B:-1}"
RUN_QWEN4B_HF="${RUN_QWEN4B_HF:-1}"
RUN_QWEN8B_HF="${RUN_QWEN8B_HF:-1}"
RUN_QWEN14B_HF="${RUN_QWEN14B_HF:-1}"
RUN_QWEN9B_HF="${RUN_QWEN9B_HF:-1}"
RUN_QWEN27B_HF="${RUN_QWEN27B_HF:-1}"

QWEN4B_TP="${QWEN4B_TP:-1}"
QWEN8B_TP="${QWEN8B_TP:-1}"
QWEN14B_TP="${QWEN14B_TP:-1}"
QWEN9B_TP="${QWEN9B_TP:-1}"
QWEN27B_TP="${QWEN27B_TP:-2}"
QWEN4B_GPU_UTIL="${QWEN4B_GPU_UTIL:-0.70}"
QWEN8B_GPU_UTIL="${QWEN8B_GPU_UTIL:-0.80}"
QWEN14B_GPU_UTIL="${QWEN14B_GPU_UTIL:-0.90}"
QWEN9B_GPU_UTIL="${QWEN9B_GPU_UTIL:-0.85}"
QWEN27B_GPU_UTIL="${QWEN27B_GPU_UTIL:-0.92}"

QWEN_HF_DTYPE="${QWEN_HF_DTYPE:-bfloat16}"
QWEN_HF_DEVICE_MAP="${QWEN_HF_DEVICE_MAP:-auto}"
QWEN_HF_ATTN_IMPL="${QWEN_HF_ATTN_IMPL:-}"
QWEN_HF_TRUST_REMOTE_CODE="${QWEN_HF_TRUST_REMOTE_CODE:-1}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-python3}"
HF_PYTHON_BIN="${HF_PYTHON_BIN:-python3}"
PREP_PYTHON_BIN="${PREP_PYTHON_BIN:-${VLLM_PYTHON_BIN}}"

if [[ ! -d "${PROMPT_STATE_DIR}" || ! -d "${PROMPT_HISTORY_DIR}" ]]; then
  echo "ERROR: prompt dirs not found:"
  echo "  state=${PROMPT_STATE_DIR}"
  echo "  history=${PROMPT_HISTORY_DIR}"
  exit 1
fi

echo "[info] out_dir=${OUT_DIR}"
echo "[info] prompt_state_dir=${PROMPT_STATE_DIR}"
echo "[info] prompt_history_dir=${PROMPT_HISTORY_DIR}"
echo "[info] benchmark_mode=${BENCHMARK_MODE}"
echo "[info] warmup_prompts_per_variant=${WARMUP_PROMPTS_PER_VARIANT}"
echo "[info] qwen35_compat_root=${QWEN35_COMPAT_ROOT}"
echo "[info] max_model_len=${MAX_MODEL_LEN}"
echo "[info] vllm_enable_prefix_caching=${VLLM_ENABLE_PREFIX_CACHING}"
echo "[info] vllm_enable_chunked_prefill=${VLLM_ENABLE_CHUNKED_PREFILL}"

prepare_qwen35_compat() {
  local model_id="$1"
  local compat_dir="$2"
  local target_arch="$3"
  if [[ -f "${compat_dir}/config.json" ]]; then
    echo "[info] reusing qwen3.5 compat model ${compat_dir}"
    return 0
  fi
  mkdir -p "$(dirname "${compat_dir}")"
  echo "[step] prepare qwen3.5 compat model ${model_id} -> ${compat_dir}"
  "${PREP_PYTHON_BIN}" scripts/prepare_qwen35_compat_model.py \
    --model-id "${model_id}" \
    --output-dir "${compat_dir}" \
    --target-architecture "${target_arch}"
}

run_vllm_native_bench() {
  local label="$1"
  local model_name_for_report="$2"
  local model_id="$3"
  local tp_size="$4"
  local gpu_util="$5"

  local bench_out="${OUT_DIR}/${label}"
  mkdir -p "${bench_out}"

  EXTRA_ARGS=()
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    EXTRA_ARGS+=(--vllm-trust-remote-code)
  fi
  if [[ "${SAFE_MODE}" == "1" ]]; then
    EXTRA_ARGS+=(--vllm-enforce-eager)
  fi
  if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "1" ]]; then
    EXTRA_ARGS+=(--vllm-enable-prefix-caching)
  fi
  if [[ "${VLLM_ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
    EXTRA_ARGS+=(--vllm-enable-chunked-prefill)
  fi

  "${VLLM_PYTHON_BIN}" scripts/benchmark_future_imagination_latency.py \
    --provider vllm_offline \
    --model "${model_name_for_report}" \
    --vllm-model-id "${model_id}" \
    --vllm-tensor-parallel-size "${tp_size}" \
    --vllm-dtype bfloat16 \
    --vllm-max-model-len "${MAX_MODEL_LEN}" \
    --vllm-gpu-memory-utilization "${gpu_util}" \
    "${EXTRA_ARGS[@]}" \
    --prompt-dir-state "${PROMPT_STATE_DIR}" \
    --prompt-dir-history "${PROMPT_HISTORY_DIR}" \
    --benchmark-mode "${BENCHMARK_MODE}" \
    --warmup-prompts-per-variant "${WARMUP_PROMPTS_PER_VARIANT}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --repeats "${REPEATS}" \
    --timeout-s "${REQUEST_TIMEOUT_S}" \
    --hidden-target-layer -1 \
    --output-dir "${bench_out}" \
    --run-label "${label}"
}

run_qwen_hf_bench() {
  local label="$1"
  local model_id="$2"
  local bench_out="${OUT_DIR}/${label}"
  mkdir -p "${bench_out}"

  HF_ARGS=()
  if [[ "${QWEN_HF_TRUST_REMOTE_CODE}" == "1" ]]; then
    HF_ARGS+=(--hf-trust-remote-code)
  fi
  if [[ -n "${QWEN_HF_ATTN_IMPL}" ]]; then
    HF_ARGS+=(--hf-attn-implementation "${QWEN_HF_ATTN_IMPL}")
  fi

  "${HF_PYTHON_BIN}" scripts/benchmark_future_imagination_latency.py \
    --provider hf_local \
    --model "${model_id}" \
    --hf-model-id "${model_id}" \
    --hf-device-map "${QWEN_HF_DEVICE_MAP}" \
    --hf-dtype "${QWEN_HF_DTYPE}" \
    "${HF_ARGS[@]}" \
    --prompt-dir-state "${PROMPT_STATE_DIR}" \
    --prompt-dir-history "${PROMPT_HISTORY_DIR}" \
    --benchmark-mode "${BENCHMARK_MODE}" \
    --warmup-prompts-per-variant "${WARMUP_PROMPTS_PER_VARIANT}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --repeats "${REPEATS}" \
    --timeout-s "${REQUEST_TIMEOUT_S}" \
    --hidden-target-layer -1 \
    --output-dir "${bench_out}" \
    --run-label "${label}"
}

if [[ "${RUN_QWEN4B}" == "1" ]]; then
  run_vllm_native_bench "qwen3_4b_vllm_native_hidden" "${QWEN4B_MODEL_ID}" "${QWEN4B_MODEL_ID}" "${QWEN4B_TP}" "${QWEN4B_GPU_UTIL}"
fi
if [[ "${RUN_QWEN4B_HF}" == "1" ]]; then
  run_qwen_hf_bench "qwen3_4b_hf_hidden" "${QWEN4B_MODEL_ID}"
fi
if [[ "${RUN_QWEN8B}" == "1" ]]; then
  run_vllm_native_bench "qwen3_8b_vllm_native_hidden" "${QWEN8B_MODEL_ID}" "${QWEN8B_MODEL_ID}" "${QWEN8B_TP}" "${QWEN8B_GPU_UTIL}"
fi
if [[ "${RUN_QWEN8B_HF}" == "1" ]]; then
  run_qwen_hf_bench "qwen3_8b_hf_hidden" "${QWEN8B_MODEL_ID}"
fi
if [[ "${RUN_QWEN14B}" == "1" ]]; then
  run_vllm_native_bench "qwen3_14b_vllm_native_hidden" "${QWEN14B_MODEL_ID}" "${QWEN14B_MODEL_ID}" "${QWEN14B_TP}" "${QWEN14B_GPU_UTIL}"
fi
if [[ "${RUN_QWEN14B_HF}" == "1" ]]; then
  run_qwen_hf_bench "qwen3_14b_hf_hidden" "${QWEN14B_MODEL_ID}"
fi
if [[ "${RUN_QWEN9B}" == "1" ]]; then
  prepare_qwen35_compat "${QWEN9B_MODEL_ID}" "${QWEN9B_COMPAT_MODEL_DIR}" "Qwen3_5ForCausalLM"
  run_vllm_native_bench "qwen3_5_9b_vllm_native_hidden" "${QWEN9B_MODEL_ID}" "${QWEN9B_COMPAT_MODEL_DIR}" "${QWEN9B_TP}" "${QWEN9B_GPU_UTIL}"
fi
if [[ "${RUN_QWEN9B_HF}" == "1" ]]; then
  run_qwen_hf_bench "qwen3_5_9b_hf_hidden" "${QWEN9B_MODEL_ID}"
fi
if [[ "${RUN_QWEN27B}" == "1" ]]; then
  prepare_qwen35_compat "${QWEN27B_MODEL_ID}" "${QWEN27B_COMPAT_MODEL_DIR}" "Qwen3_5MoeForCausalLM"
  run_vllm_native_bench "qwen3_5_27b_vllm_native_hidden" "${QWEN27B_MODEL_ID}" "${QWEN27B_COMPAT_MODEL_DIR}" "${QWEN27B_TP}" "${QWEN27B_GPU_UTIL}"
fi
if [[ "${RUN_QWEN27B_HF}" == "1" ]]; then
  run_qwen_hf_bench "qwen3_5_27b_hf_hidden" "${QWEN27B_MODEL_ID}"
fi

echo "[step] build combined markdown"
OUT_DIR_FOR_PY="${OUT_DIR}" python3 - <<'PY'
import json
import os
import pathlib

out_dir = pathlib.Path(os.environ["OUT_DIR_FOR_PY"])
summary_rows = []

for child in sorted(out_dir.iterdir()):
    if not child.is_dir():
        continue
    summary_path = child / "summary.json"
    if not summary_path.exists():
        continue
    rows = json.loads(summary_path.read_text())
    for row in rows:
        row = dict(row)
        row["run"] = child.name
        summary_rows.append(row)

lines = []
lines.append("# Combined Latency Benchmark")
lines.append("")
lines.append(f"- output_dir: `{out_dir}`")
lines.append("")
lines.append("| run | model | prompt_variant | total_calls | successful_calls | error_calls | request_latency_s | request_calls_per_second | hidden_ready_latency_s | hidden_ready_calls_per_second | prompt_tokens | completion_tokens | total_tokens |")
lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in summary_rows:
    def fmt(x, nd=4):
        if x is None:
            return ""
        if isinstance(x, (int, float)):
            return f"{x:.{nd}f}"
        return str(x)
    lines.append(
        "| {run} | {model} | {variant} | {calls} | {ok_calls} | {error_calls} | {lat_mean} | {rps_mean} | {tot_lat_mean} | {tot_rps_mean} | {prompt_mean} | {comp_mean} | {total_mean} |".format(
            run=r.get("run", ""),
            model=r.get("model", ""),
            variant=r.get("variant", ""),
            calls=r.get("calls", ""),
            ok_calls=r.get("ok_calls", ""),
            error_calls=r.get("error_calls", ""),
            lat_mean=fmt(r.get("avg_request_latency_seconds")),
            rps_mean=fmt(r.get("avg_requests_per_second")),
            tot_lat_mean=fmt(r.get("avg_end_to_end_latency_seconds")),
            tot_rps_mean=fmt(r.get("avg_end_to_end_responses_per_second")),
            prompt_mean=fmt(r.get("avg_prompt_tokens"), 1),
            comp_mean=fmt(r.get("avg_completion_tokens"), 1),
            total_mean=fmt(r.get("avg_total_tokens"), 1),
        )
    )

(out_dir / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_dir / "combined_summary.md")
PY

echo "[done] out_dir=${OUT_DIR}"
