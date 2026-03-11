#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

set +u
source ~/.bashrc
set -u

PROMPT_ITER_ENV_PATH="${PROMPT_ITER_ENV_PATH:-/data/user_data/geney/.conda/envs/craftax_fast_llm}"
PROMPT_ITER_VLLM_URL="${PROMPT_ITER_VLLM_URL:-http://127.0.0.1:8000}"
PROMPT_ITER_REQUIRE_VLLM="${PROMPT_ITER_REQUIRE_VLLM:-1}"
PROMPT_ITER_HOST="${PROMPT_ITER_HOST:-127.0.0.1}"
PROMPT_ITER_PORT="${PROMPT_ITER_PORT:-8501}"

if [[ ! -d "${PROMPT_ITER_ENV_PATH}" ]]; then
  echo "ERROR: prompt-iter env path does not exist: ${PROMPT_ITER_ENV_PATH}" >&2
  exit 1
fi

conda activate "${PROMPT_ITER_ENV_PATH}"

echo "Prompt iteration env: ${PROMPT_ITER_ENV_PATH}"
echo "Checking Python deps..."
python - <<PY
import importlib
mods = ["streamlit", "requests"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m}: {exc.__class__.__name__}: {exc}")
if missing:
    raise SystemExit("\\n".join(missing))
print("dependency_check_ok")
PY

health_ok=0
for _ in $(seq 1 10); do
  if curl -fsS "${PROMPT_ITER_VLLM_URL}/health" >/dev/null 2>&1; then
    health_ok=1
    break
  fi
  sleep 2
done

if [[ "${health_ok}" -ne 1 ]]; then
  msg="vLLM not healthy at ${PROMPT_ITER_VLLM_URL}/health"
  if [[ "${PROMPT_ITER_REQUIRE_VLLM}" == "1" ]]; then
    echo "ERROR: ${msg}" >&2
    exit 1
  fi
  echo "WARNING: ${msg} (continuing because PROMPT_ITER_REQUIRE_VLLM=${PROMPT_ITER_REQUIRE_VLLM})"
else
  echo "vLLM healthy: ${PROMPT_ITER_VLLM_URL}/health"
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting Streamlit on ${PROMPT_ITER_HOST}:${PROMPT_ITER_PORT}"
python -m streamlit run scripts/prompt_iter_webapp.py \
  --server.address "${PROMPT_ITER_HOST}" \
  --server.port "${PROMPT_ITER_PORT}" \
  --server.headless true
