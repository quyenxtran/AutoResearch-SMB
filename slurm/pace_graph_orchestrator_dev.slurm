#!/bin/bash
# Run the graph-first orchestrator on two RTX6000 GPUs in one batch:
# sbatch --export=ALL,NUM_EXPERIMENTS=20,RESET_CHECKPOINT=1 jobs/pace_graph_orchestrator_dev.slurm
# Bootstrap environment if needed:
# sbatch --export=ALL,BOOTSTRAP_ENV=1 jobs/pace_graph_orchestrator_dev.slurm
# Override config paths if needed:
# sbatch --export=ALL,EXPERIMENT_FILE=configs/adsorption_equilibrium_gpt_graph.yaml,BASELINE_EXPERIMENT_FILE=configs/adsorption_equilibrium_graph_dev.yaml jobs/pace_graph_orchestrator_dev.slurm
# Override account or notifications at submit time if needed:
# sbatch --account=gts-sn73 --mail-user=qtran47@gatech.edu --mail-type=END,FAIL jobs/pace_graph_orchestrator_dev.slurm
#SBATCH --job-name=autoresearch-graph-orch
#SBATCH -A gts-sn73
#SBATCH --qos=inferno
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=qtran47@gatech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${DEFAULT_REPO_ROOT}" ]]; then
  DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-$REPO_ROOT/jobs/install.sh}"
DEFAULT_SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-${SCRATCH:-}}"
if [[ -z "${DEFAULT_SCRATCH_ROOT}" && -n "${USER:-}" && -d "/scratch/${USER}" ]]; then
  DEFAULT_SCRATCH_ROOT="/scratch/${USER}"
fi
if [[ -z "${DEFAULT_SCRATCH_ROOT}" && -n "${HOME:-}" && "${HOME}" =~ ^/storage/home/[^/]+/([^/]+)/([^/]+)$ ]]; then
  HOME_GROUP="${BASH_REMATCH[1]}"
  HOME_USER="${BASH_REMATCH[2]}"
  if [[ -d "/storage/scratch1/${HOME_GROUP}/${HOME_USER}" ]]; then
    DEFAULT_SCRATCH_ROOT="/storage/scratch1/${HOME_GROUP}/${HOME_USER}"
  fi
fi
PACE_VENV_ROOT="${PACE_VENV_ROOT:-${DEFAULT_SCRATCH_ROOT}}"
if [[ -n "${PACE_VENV_ROOT}" ]]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PACE_VENV_ROOT/AutoResearch-ML/.pip-cache}"
fi
if [[ -z "${VENV_PY:-}" ]]; then
  venv_candidates=()
  if [[ -n "${VENV_DIR:-}" ]]; then
    venv_candidates+=("${VENV_DIR}/bin/python")
  fi
  if [[ -n "${PACE_VENV_ROOT}" ]]; then
    venv_candidates+=("${PACE_VENV_ROOT}/AutoResearch-ML/.venv/bin/python")
  fi
  venv_candidates+=("${REPO_ROOT}/.venv/bin/python")
  if [[ -n "${HOME:-}" ]]; then
    venv_candidates+=("${HOME}/AutoResearch-ML/.venv/bin/python")
  fi

  for candidate in "${venv_candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      VENV_PY="${candidate}"
      VENV_DIR="$(cd "$(dirname "${candidate}")/.." && pwd)"
      break
    fi
  done
fi
CLI_MODULE="${CLI_MODULE:-autoresearch_ml.cli}"
EXPERIMENT_FILE="${EXPERIMENT_FILE:-configs/adsorption_equilibrium_gpt_graph.yaml}"
BASELINE_EXPERIMENT_FILE="${BASELINE_EXPERIMENT_FILE:-configs/adsorption_equilibrium_graph_dev.yaml}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-nano}"
export OPENAI_MODEL
LOG_DIR="${PACE_LOG_DIR:-logs}"
COORDINATOR_USE_LOCAL_LLM="${COORDINATOR_USE_LOCAL_LLM:-1}"
COORDINATOR_LOCAL_LLM_BACKEND="${COORDINATOR_LOCAL_LLM_BACKEND:-ollama}"
COORDINATOR_OLLAMA_MODEL="${COORDINATOR_OLLAMA_MODEL:-qwen3.5:9b}"
COORDINATOR_SHARED_LOCAL_MODEL="${COORDINATOR_SHARED_LOCAL_MODEL:-1}"
COORDINATOR_OLLAMA_BASE_URL="${COORDINATOR_OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
COORDINATOR_OLLAMA_HOST="${COORDINATOR_OLLAMA_HOST:-127.0.0.1:11434}"
COORDINATOR_OLLAMA_LOG="${COORDINATOR_OLLAMA_LOG:-$LOG_DIR/ollama-coordinator-${SLURM_JOB_ID:-local}.log}"
COORDINATOR_OLLAMA_BIN="${COORDINATOR_OLLAMA_BIN:-}"
COORDINATOR_LLM_STARTUP_TIMEOUT="${COORDINATOR_LLM_STARTUP_TIMEOUT:-90}"
COORDINATOR_LLM_STARTUP_POLL_INTERVAL="${COORDINATOR_LLM_STARTUP_POLL_INTERVAL:-3}"
COORDINATOR_OLLAMA_KEEP_ALIVE="${COORDINATOR_OLLAMA_KEEP_ALIVE:-30m}"
COORDINATOR_OLLAMA_MAX_LOADED_MODELS="${COORDINATOR_OLLAMA_MAX_LOADED_MODELS:-2}"
COORDINATOR_OLLAMA_NUM_PARALLEL="${COORDINATOR_OLLAMA_NUM_PARALLEL:-2}"
COORDINATOR_OLLAMA_LOAD_TIMEOUT="${COORDINATOR_OLLAMA_LOAD_TIMEOUT:-10m}"
COORDINATOR_VLLM_MODEL="${COORDINATOR_VLLM_MODEL:-}"
COORDINATOR_VLLM_HOST="${COORDINATOR_VLLM_HOST:-127.0.0.1}"
COORDINATOR_VLLM_PORT="${COORDINATOR_VLLM_PORT:-8000}"
COORDINATOR_VLLM_BASE_URL="${COORDINATOR_VLLM_BASE_URL:-http://${COORDINATOR_VLLM_HOST}:${COORDINATOR_VLLM_PORT}/v1}"
COORDINATOR_VLLM_LOG="${COORDINATOR_VLLM_LOG:-$LOG_DIR/vllm-coordinator-${SLURM_JOB_ID:-local}.log}"
COORDINATOR_VLLM_VISIBLE_DEVICES="${COORDINATOR_VLLM_VISIBLE_DEVICES:-0}"
COORDINATOR_VLLM_TENSOR_PARALLEL_SIZE="${COORDINATOR_VLLM_TENSOR_PARALLEL_SIZE:-1}"
COORDINATOR_VLLM_MAX_MODEL_LEN="${COORDINATOR_VLLM_MAX_MODEL_LEN:-8192}"
COORDINATOR_VLLM_DTYPE="${COORDINATOR_VLLM_DTYPE:-auto}"
COORDINATOR_VLLM_GPU_MEMORY_UTILIZATION="${COORDINATOR_VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
COORDINATOR_VLLM_ENFORCE_EAGER="${COORDINATOR_VLLM_ENFORCE_EAGER:-0}"
COORDINATOR_VLLM_TRUST_REMOTE_CODE="${COORDINATOR_VLLM_TRUST_REMOTE_CODE:-0}"
COORDINATOR_VLLM_EXTRA_ARGS="${COORDINATOR_VLLM_EXTRA_ARGS:-}"
COORDINATOR_VLLM_ISOLATED_VENV="${COORDINATOR_VLLM_ISOLATED_VENV:-1}"
COORDINATOR_VLLM_PACKAGE_SPEC="${COORDINATOR_VLLM_PACKAGE_SPEC:-vllm}"
if [[ -n "${PACE_VENV_ROOT}" ]]; then
  COORDINATOR_VLLM_VENV_DIR="${COORDINATOR_VLLM_VENV_DIR:-$PACE_VENV_ROOT/AutoResearch-ML/.venv-vllm}"
else
  COORDINATOR_VLLM_VENV_DIR="${COORDINATOR_VLLM_VENV_DIR:-$REPO_ROOT/.venv-vllm}"
fi
COORDINATOR_VLLM_PY="${COORDINATOR_VLLM_PY:-$COORDINATOR_VLLM_VENV_DIR/bin/python}"
if [[ "${COORDINATOR_LOCAL_LLM_BACKEND}" == "vllm" && "${COORDINATOR_VLLM_ISOLATED_VENV}" != "1" ]]; then
  INSTALL_VLLM="${INSTALL_VLLM:-1}"
else
  INSTALL_VLLM="${INSTALL_VLLM:-0}"
fi
export INSTALL_VLLM
CRITIC_ENABLED="${CRITIC_ENABLED:-1}"
CRITIC_USE_LOCAL_LLM="${CRITIC_USE_LOCAL_LLM:-1}"
CRITIC_PROVIDER_ORDER="${CRITIC_PROVIDER_ORDER:-ollama-first}"
if [[ -z "${CRITIC_OLLAMA_MODEL:-}" ]]; then
  if [[ "${COORDINATOR_SHARED_LOCAL_MODEL}" == "1" ]]; then
    CRITIC_OLLAMA_MODEL="${COORDINATOR_OLLAMA_MODEL}"
  else
    CRITIC_OLLAMA_MODEL="qwen3.5:9b"
  fi
fi
CRITIC_OLLAMA_BASE_URL="${CRITIC_OLLAMA_BASE_URL:-${COORDINATOR_OLLAMA_BASE_URL}}"
CRITIC_OPENAI_MODEL="${CRITIC_OPENAI_MODEL:-${OPENAI_MODEL}}"
export CRITIC_ENABLED CRITIC_USE_LOCAL_LLM CRITIC_PROVIDER_ORDER CRITIC_OLLAMA_MODEL CRITIC_OLLAMA_BASE_URL CRITIC_OPENAI_MODEL
NUM_EXPERIMENTS="${NUM_EXPERIMENTS:-20}"
MAX_WORKERS="${MAX_WORKERS:-2}"
ORCHESTRATOR_GPU_COUNT="${ORCHESTRATOR_GPU_COUNT:-2}"
ORCHESTRATOR_GPU_SLOTS="${ORCHESTRATOR_GPU_SLOTS:-}"
WORKER_GPU_OFFSET="${WORKER_GPU_OFFSET:-0}"
ORCHESTRATOR_ADAPTIVE_GPU_THROTTLE="${ORCHESTRATOR_ADAPTIVE_GPU_THROTTLE:-1}"
ORCHESTRATOR_GPU_OVERLOAD_UTIL_THRESHOLD="${ORCHESTRATOR_GPU_OVERLOAD_UTIL_THRESHOLD:-95}"
ORCHESTRATOR_GPU_OVERLOAD_CONSECUTIVE_SAMPLES="${ORCHESTRATOR_GPU_OVERLOAD_CONSECUTIVE_SAMPLES:-3}"
MAX_QUEUE="${MAX_QUEUE:-$((MAX_WORKERS + 2))}"
export ORCHESTRATOR_GPU_COUNT ORCHESTRATOR_GPU_SLOTS WORKER_GPU_OFFSET
export ORCHESTRATOR_ADAPTIVE_GPU_THROTTLE ORCHESTRATOR_GPU_OVERLOAD_UTIL_THRESHOLD ORCHESTRATOR_GPU_OVERLOAD_CONSECUTIVE_SAMPLES
COORDINATOR_OLLAMA_USE_GPU="${COORDINATOR_OLLAMA_USE_GPU:-1}"
OLLAMA_GPU_ID="${OLLAMA_GPU_ID:-0}"
SMB_CURRENT_DEFAULT_SOLVER_STACK="${SMB_CURRENT_DEFAULT_SOLVER_STACK:-pyomo+ipopt_sens+ma57}"
SMB_AVAILABLE_SOLVER_OPTIONS="${SMB_AVAILABLE_SOLVER_OPTIONS:-pyomo+ipopt_sens+ma57(current_default_cpu); pyomo+ipopt_sens+mumps(cpu_if_installed); pyomo+ipopt_sens+pardiso(cpu_if_installed); pyomo+ipopt_sens+ma97(cpu_if_installed); madnlp+cudss(experimental_gpu_if_installed)}"
SMB_AVAILABLE_ACCELERATION_OPTIONS="${SMB_AVAILABLE_ACCELERATION_OPTIONS:-optimized_blas_lapack_if_available; warm_starts; cpu_parallel_screening; fidelity_ladder; gpu_for_qwen_inference; experimental_gpu_nlp_solver_if_installed}"
SMB_RESOURCE_DECISION_POLICY="${SMB_RESOURCE_DECISION_POLICY:-model_may_choose_best_available_option_after_checking_installation_runtime_and_solution_quality}"
SMB_IPOPT_RESOURCE_FILE="${SMB_IPOPT_RESOURCE_FILE:-$REPO_ROOT/AutoResearch-SMB/IPOPT_SOLVER_RESOURCES.md}"
SMB_IPOPT_PROFILE_MENU="${SMB_IPOPT_PROFILE_MENU:-baseline_reproduction; screening_fast; balanced_cpu_mumps; robust_cpu_ma97; robust_cpu_mumps; high_end_pardiso; validation_strict}"
SMB_IPOPT_BASELINE_FALLBACK_TREE="${SMB_IPOPT_BASELINE_FALLBACK_TREE:-baseline_reproduction -> robust_cpu_mumps -> validation_strict}"
SMB_IPOPT_SCREENING_FALLBACK_TREE="${SMB_IPOPT_SCREENING_FALLBACK_TREE:-screening_fast -> balanced_cpu_mumps -> baseline_reproduction}"
SMB_IPOPT_HARD_PROBLEM_FALLBACK_TREE="${SMB_IPOPT_HARD_PROBLEM_FALLBACK_TREE:-robust_cpu_ma97 -> robust_cpu_mumps -> validation_strict}"
SMB_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE="${SMB_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE:-high_end_pardiso -> robust_cpu_ma97 -> robust_cpu_mumps}"
export SMB_CURRENT_DEFAULT_SOLVER_STACK SMB_AVAILABLE_SOLVER_OPTIONS SMB_AVAILABLE_ACCELERATION_OPTIONS SMB_RESOURCE_DECISION_POLICY
export SMB_IPOPT_RESOURCE_FILE SMB_IPOPT_PROFILE_MENU SMB_IPOPT_BASELINE_FALLBACK_TREE SMB_IPOPT_SCREENING_FALLBACK_TREE SMB_IPOPT_HARD_PROBLEM_FALLBACK_TREE SMB_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE
SMB_IPOPT_PRECHECK_SUMMARY=""
SMB_VERIFIED_IPOPT_EXECUTABLES=""
SMB_VERIFIED_IPOPT_LINEAR_SOLVERS=""
SMB_VERIFIED_IPOPT_PROFILE_MENU=""
SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE=""
export SMB_IPOPT_PRECHECK_SUMMARY SMB_VERIFIED_IPOPT_EXECUTABLES SMB_VERIFIED_IPOPT_LINEAR_SOLVERS SMB_VERIFIED_IPOPT_PROFILE_MENU
export SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE
ENSURE_BASELINES="${ENSURE_BASELINES:-1}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$REPO_ROOT/logs/$(basename "${EXPERIMENT_FILE%.*}").graph_orchestrator.checkpoint}"
STATE_DIR="${STATE_DIR:-$REPO_ROOT/artifacts/orchestrator/$(basename "${EXPERIMENT_FILE%.*}")}"
BASELINE_LEADERBOARD="${BASELINE_LEADERBOARD:-$REPO_ROOT/artifacts/leaderboards/$(basename "${BASELINE_EXPERIMENT_FILE%.*}").json}"
COORDINATOR_LOCAL_LLM_PID=""

append_resource_item() {
  local var_name="$1"
  local item="$2"
  local current="${!var_name:-}"
  if [[ -z "${item}" ]]; then
    return 0
  fi
  if [[ -z "${current}" ]]; then
    printf -v "${var_name}" '%s' "${item}"
  else
    printf -v "${var_name}" '%s; %s' "${current}" "${item}"
  fi
}

probe_ipopt_resources() {
  local combined_options=""
  local probe_output=""
  local solver_cmd=""

  if command -v ipopt_sens >/dev/null 2>&1; then
    append_resource_item SMB_VERIFIED_IPOPT_EXECUTABLES "ipopt_sens"
    probe_output="$(ipopt_sens --print-options 2>/dev/null || true)"
    if [[ -n "${probe_output}" ]]; then
      combined_options="${combined_options}"$'\n'"${probe_output}"
    fi
  fi

  if command -v ipopt >/dev/null 2>&1; then
    append_resource_item SMB_VERIFIED_IPOPT_EXECUTABLES "ipopt"
    probe_output="$(ipopt --print-options 2>/dev/null || true)"
    if [[ -n "${probe_output}" ]]; then
      combined_options="${combined_options}"$'\n'"${probe_output}"
    fi
  fi

  if printf '%s' "${combined_options}" | grep -qi "MA57 Linear Solver"; then
    append_resource_item SMB_VERIFIED_IPOPT_LINEAR_SOLVERS "ma57"
  fi
  if printf '%s' "${combined_options}" | grep -qi "Mumps Linear Solver"; then
    append_resource_item SMB_VERIFIED_IPOPT_LINEAR_SOLVERS "mumps"
  fi
  if printf '%s' "${combined_options}" | grep -qi "MA97 Linear Solver"; then
    append_resource_item SMB_VERIFIED_IPOPT_LINEAR_SOLVERS "ma97"
  fi
  if printf '%s' "${combined_options}" | grep -qi "Pardiso"; then
    append_resource_item SMB_VERIFIED_IPOPT_LINEAR_SOLVERS "pardiso"
  fi

  if [[ "${SMB_VERIFIED_IPOPT_EXECUTABLES}" == *"ipopt_sens"* && "${SMB_VERIFIED_IPOPT_LINEAR_SOLVERS}" == *"ma57"* ]]; then
    append_resource_item SMB_VERIFIED_IPOPT_PROFILE_MENU "baseline_reproduction"
    append_resource_item SMB_VERIFIED_IPOPT_PROFILE_MENU "validation_strict"
  fi
  if [[ ( "${SMB_VERIFIED_IPOPT_EXECUTABLES}" == *"ipopt_sens"* || "${SMB_VERIFIED_IPOPT_EXECUTABLES}" == *"ipopt"* ) && "${SMB_VERIFIED_IPOPT_LINEAR_SOLVERS}" == *"mumps"* ]]; then
    append_resource_item SMB_VERIFIED_IPOPT_PROFILE_MENU "screening_fast"
    append_resource_item SMB_VERIFIED_IPOPT_PROFILE_MENU "balanced_cpu_mumps"
    append_resource_item SMB_VERIFIED_IPOPT_PROFILE_MENU "robust_cpu_mumps"
  fi
  if [[ "${SMB_VERIFIED_IPOPT_EXECUTABLES}" == *"ipopt_sens"* && "${SMB_VERIFIED_IPOPT_LINEAR_SOLVERS}" == *"ma97"* ]]; then
    append_resource_item SMB_VERIFIED_IPOPT_PROFILE_MENU "robust_cpu_ma97"
  fi
  if [[ "${SMB_VERIFIED_IPOPT_EXECUTABLES}" == *"ipopt_sens"* && "${SMB_VERIFIED_IPOPT_LINEAR_SOLVERS}" == *"pardiso"* ]]; then
    append_resource_item SMB_VERIFIED_IPOPT_PROFILE_MENU "high_end_pardiso"
  fi

  for solver_cmd in baseline_reproduction robust_cpu_mumps validation_strict; do
    if [[ "${SMB_VERIFIED_IPOPT_PROFILE_MENU}" == *"${solver_cmd}"* ]]; then
      append_resource_item SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE "${solver_cmd}"
    fi
  done
  for solver_cmd in screening_fast balanced_cpu_mumps baseline_reproduction; do
    if [[ "${SMB_VERIFIED_IPOPT_PROFILE_MENU}" == *"${solver_cmd}"* ]]; then
      append_resource_item SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE "${solver_cmd}"
    fi
  done
  for solver_cmd in robust_cpu_ma97 robust_cpu_mumps validation_strict; do
    if [[ "${SMB_VERIFIED_IPOPT_PROFILE_MENU}" == *"${solver_cmd}"* ]]; then
      append_resource_item SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE "${solver_cmd}"
    fi
  done
  for solver_cmd in high_end_pardiso robust_cpu_ma97 robust_cpu_mumps; do
    if [[ "${SMB_VERIFIED_IPOPT_PROFILE_MENU}" == *"${solver_cmd}"* ]]; then
      append_resource_item SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE "${solver_cmd}"
    fi
  done

  if [[ -z "${SMB_VERIFIED_IPOPT_EXECUTABLES}" ]]; then
    SMB_IPOPT_PRECHECK_SUMMARY="No ipopt executable verified on PATH."
  else
    SMB_IPOPT_PRECHECK_SUMMARY="executables=${SMB_VERIFIED_IPOPT_EXECUTABLES:-none}; linear_solvers=${SMB_VERIFIED_IPOPT_LINEAR_SOLVERS:-none}; profiles=${SMB_VERIFIED_IPOPT_PROFILE_MENU:-none}"
  fi

  export SMB_IPOPT_PRECHECK_SUMMARY
  export SMB_VERIFIED_IPOPT_EXECUTABLES
  export SMB_VERIFIED_IPOPT_LINEAR_SOLVERS
  export SMB_VERIFIED_IPOPT_PROFILE_MENU
  export SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE
  export SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE
  export SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE
  export SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE
}

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}" mlruns artifacts
unset MLFLOW_TRACKING_URI
export PYTHONUNBUFFERED=1

if [[ "${BOOTSTRAP_ENV:-0}" == "1" ]] || [[ ! -x "${VENV_PY}" ]]; then
  bash "${INSTALL_SCRIPT}"
fi

if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: expected virtualenv python not found at ${VENV_PY}"
  echo "Run jobs/install.sh first or resubmit with BOOTSTRAP_ENV=1."
  exit 1
fi

if [[ "${COORDINATOR_LOCAL_LLM_BACKEND}" == "vllm" ]]; then
  if [[ "${COORDINATOR_VLLM_ISOLATED_VENV}" == "1" ]]; then
    BASE_PYTHON_BIN="${BASE_PYTHON_BIN:-$(command -v python3 || command -v python)}"
    if [[ -z "${BASE_PYTHON_BIN}" ]]; then
      echo "ERROR: no base Python interpreter found to create the isolated vLLM venv."
      exit 1
    fi
    if [[ "${BOOTSTRAP_ENV:-0}" == "1" ]] || [[ ! -x "${COORDINATOR_VLLM_PY}" ]]; then
      echo "Preparing isolated vLLM virtualenv at ${COORDINATOR_VLLM_VENV_DIR}"
      mkdir -p "$(dirname "${COORDINATOR_VLLM_VENV_DIR}")"
      "${BASE_PYTHON_BIN}" -m venv "${COORDINATOR_VLLM_VENV_DIR}"
      "${COORDINATOR_VLLM_PY}" -m pip install --upgrade pip
      "${COORDINATOR_VLLM_PY}" -m pip install "${COORDINATOR_VLLM_PACKAGE_SPEC}"
    fi
    if ! "${COORDINATOR_VLLM_PY}" -c "import vllm" >/dev/null 2>&1; then
      echo "ERROR: isolated vLLM environment at ${COORDINATOR_VLLM_VENV_DIR} does not contain vllm."
      exit 1
    fi
  elif ! "${VENV_PY}" -c "import vllm" >/dev/null 2>&1; then
    echo "ERROR: COORDINATOR_LOCAL_LLM_BACKEND=vllm but vllm is not installed in ${VENV_DIR}."
    echo "Resubmit with BOOTSTRAP_ENV=1 or set INSTALL_VLLM=1 so jobs/install.sh installs vllm."
    exit 1
  fi
fi

export PATH="${VENV_DIR}/bin:${PATH}"
if [[ -d "${HOME:-}/.local/ollama/bin" ]]; then
  export PATH="${HOME}/.local/ollama/bin:${PATH}"
fi

probe_ipopt_resources

resolve_coordinator_ollama_bin() {
  if [[ -n "${COORDINATOR_OLLAMA_BIN}" ]]; then
    echo "${COORDINATOR_OLLAMA_BIN}"
    return 0
  fi
  if command -v ollama >/dev/null 2>&1; then
    command -v ollama
    return 0
  fi
  if [[ -x "${HOME:-}/.local/ollama/bin/ollama" ]]; then
    echo "${HOME}/.local/ollama/bin/ollama"
    return 0
  fi
  return 1
}

ensure_coordinator_ollama_model() {
  local ollama_exec model_name
  if ! ollama_exec="$(resolve_coordinator_ollama_bin)"; then
    return 1
  fi
  for model_name in "$@"; do
    if [[ -z "${model_name}" ]]; then
      continue
    fi
    if OLLAMA_HOST="${COORDINATOR_OLLAMA_HOST}" "${ollama_exec}" list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${model_name}"; then
      echo "Coordinator Ollama model already present: ${model_name}"
      continue
    fi
    echo "[$(date '+%F %T')] PULL: coordinator Ollama model ${model_name}"
    OLLAMA_HOST="${COORDINATOR_OLLAMA_HOST}" "${ollama_exec}" pull "${model_name}"
  done
}

prewarm_coordinator_ollama_model() {
  local model_name prompt_payload response_file http_code chat_url
  chat_url="${COORDINATOR_OLLAMA_BASE_URL%/}/chat/completions"
  for model_name in "$@"; do
    if [[ -z "${model_name}" ]]; then
      continue
    fi
    echo "[$(date '+%F %T')] PREWARM: coordinator Ollama model ${model_name}"
    prompt_payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"stream":false}' "${model_name}")
    response_file="$(mktemp)"
    http_code="$(curl -sS \
      --max-time 60 \
      -o "${response_file}" \
      -w '%{http_code}' \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${OLLAMA_API_KEY:-ollama}" \
      -d "${prompt_payload}" \
      "${chat_url}" || true)"
    if [[ ! "${http_code}" =~ ^2 ]]; then
      echo "WARN: coordinator Ollama prewarm failed for model=${model_name} status=${http_code:-<curl-error>} endpoint=${chat_url}"
      if [[ -s "${response_file}" ]]; then
        echo "WARN: coordinator Ollama prewarm response body:"
        cat "${response_file}" || true
      fi
      rm -f "${response_file}"
      return 1
    fi
    rm -f "${response_file}"
  done
}

cleanup_coordinator_llm() {
  if [[ -n "${COORDINATOR_LOCAL_LLM_PID:-}" ]] && kill -0 "${COORDINATOR_LOCAL_LLM_PID}" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] STOP: coordinator local LLM server pid=${COORDINATOR_LOCAL_LLM_PID}"
    kill "${COORDINATOR_LOCAL_LLM_PID}" >/dev/null 2>&1 || true
    wait "${COORDINATOR_LOCAL_LLM_PID}" >/dev/null 2>&1 || true
  fi
  COORDINATOR_LOCAL_LLM_PID=""
}

trap cleanup_coordinator_llm EXIT

start_coordinator_local_llm() {
  local ollama_exec ready ready_url max_attempts
  if [[ "${COORDINATOR_USE_LOCAL_LLM}" != "1" && ! ( "${CRITIC_ENABLED}" == "1" && "${CRITIC_USE_LOCAL_LLM}" == "1" ) ]]; then
    return 1
  fi
  if [[ "${COORDINATOR_LOCAL_LLM_BACKEND}" != "ollama" && "${COORDINATOR_LOCAL_LLM_BACKEND}" != "vllm" ]]; then
    echo "WARN: Unsupported COORDINATOR_LOCAL_LLM_BACKEND=${COORDINATOR_LOCAL_LLM_BACKEND}; falling back to remote OpenAI."
    return 1
  fi

  if [[ "${COORDINATOR_LOCAL_LLM_BACKEND}" == "vllm" ]]; then
    if [[ -z "${COORDINATOR_VLLM_MODEL}" ]]; then
      echo "WARN: COORDINATOR_VLLM_MODEL is not set; falling back to remote OpenAI."
      return 1
    fi
    echo "[$(date '+%F %T')] START: coordinator local vLLM server (CUDA_VISIBLE_DEVICES=${COORDINATOR_VLLM_VISIBLE_DEVICES})"
    vllm_args=(
      -m vllm.entrypoints.openai.api_server
      --model "${COORDINATOR_VLLM_MODEL}"
      --host "${COORDINATOR_VLLM_HOST}"
      --port "${COORDINATOR_VLLM_PORT}"
      --tensor-parallel-size "${COORDINATOR_VLLM_TENSOR_PARALLEL_SIZE}"
      --max-model-len "${COORDINATOR_VLLM_MAX_MODEL_LEN}"
      --dtype "${COORDINATOR_VLLM_DTYPE}"
      --gpu-memory-utilization "${COORDINATOR_VLLM_GPU_MEMORY_UTILIZATION}"
    )
    if [[ "${COORDINATOR_VLLM_ENFORCE_EAGER}" == "1" ]]; then
      vllm_args+=(--enforce-eager)
    fi
    if [[ "${COORDINATOR_VLLM_TRUST_REMOTE_CODE}" == "1" ]]; then
      vllm_args+=(--trust-remote-code)
    fi
    if [[ -n "${COORDINATOR_VLLM_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      extra_vllm_args=(${COORDINATOR_VLLM_EXTRA_ARGS})
      vllm_args+=("${extra_vllm_args[@]}")
    fi
    local vllm_python
    if [[ "${COORDINATOR_VLLM_ISOLATED_VENV}" == "1" ]]; then
      vllm_python="${COORDINATOR_VLLM_PY}"
    else
      vllm_python="${VENV_PY}"
    fi
    CUDA_VISIBLE_DEVICES="${COORDINATOR_VLLM_VISIBLE_DEVICES}" "${vllm_python}" "${vllm_args[@]}" >"${COORDINATOR_VLLM_LOG}" 2>&1 &
    COORDINATOR_LOCAL_LLM_PID=$!
    echo "Coordinator vLLM pid: ${COORDINATOR_LOCAL_LLM_PID}"
    echo "Coordinator vLLM log: ${COORDINATOR_VLLM_LOG}"
    echo "Coordinator vLLM base URL: ${COORDINATOR_VLLM_BASE_URL}"
    echo "Coordinator vLLM model: ${COORDINATOR_VLLM_MODEL}"

    ready=0
    ready_url="${COORDINATOR_VLLM_BASE_URL%/v1}/health"
    max_attempts=$((COORDINATOR_LLM_STARTUP_TIMEOUT / COORDINATOR_LLM_STARTUP_POLL_INTERVAL))
    if [[ "${max_attempts}" -lt 1 ]]; then
      max_attempts=1
    fi
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
      if curl -fsS --max-time 5 "${ready_url}" >/dev/null 2>&1; then
        ready=1
        break
      fi
      if ! kill -0 "${COORDINATOR_LOCAL_LLM_PID}" >/dev/null 2>&1; then
        break
      fi
      sleep "${COORDINATOR_LLM_STARTUP_POLL_INTERVAL}"
    done

    if [[ "${ready}" != "1" ]]; then
      echo "WARN: coordinator vLLM server did not become ready at ${COORDINATOR_VLLM_BASE_URL}; falling back to remote OpenAI."
      if [[ -f "${COORDINATOR_VLLM_LOG}" ]]; then
        tail -n 100 "${COORDINATOR_VLLM_LOG}" || true
      fi
      cleanup_coordinator_llm
      return 1
    fi

    echo "[$(date '+%F %T')] READY: coordinator local vLLM server"
    export OPENAI_BASE_URL="${COORDINATOR_VLLM_BASE_URL}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
    export OPENAI_MODEL="${COORDINATOR_VLLM_MODEL}"
    export LLM_PROVIDER_ORDER="openai-first"
    if [[ "${CRITIC_ENABLED}" == "1" && "${CRITIC_USE_LOCAL_LLM}" == "1" ]]; then
      export CRITIC_PROVIDER_ORDER="openai-first"
      export CRITIC_OPENAI_MODEL="${CRITIC_OPENAI_MODEL:-${COORDINATOR_VLLM_MODEL}}"
    fi
    return 0
  fi

  if ! ollama_exec="$(resolve_coordinator_ollama_bin)"; then
    echo "WARN: Ollama binary not found for coordinator; falling back to remote OpenAI."
    return 1
  fi

  local ollama_visible_devices
  if [[ "${COORDINATOR_OLLAMA_USE_GPU}" == "1" ]]; then
    ollama_visible_devices="${OLLAMA_GPU_ID}"
    echo "[$(date '+%F %T')] START: coordinator local Ollama server (GPU ${OLLAMA_GPU_ID})"
  else
    ollama_visible_devices=""
    echo "[$(date '+%F %T')] START: coordinator local Ollama server (CPU only)"
  fi
  CUDA_VISIBLE_DEVICES="${ollama_visible_devices}" \
  OLLAMA_HOST="${COORDINATOR_OLLAMA_HOST}" \
  OLLAMA_KEEP_ALIVE="${COORDINATOR_OLLAMA_KEEP_ALIVE}" \
  OLLAMA_MAX_LOADED_MODELS="${COORDINATOR_OLLAMA_MAX_LOADED_MODELS}" \
  OLLAMA_NUM_PARALLEL="${COORDINATOR_OLLAMA_NUM_PARALLEL}" \
  OLLAMA_LOAD_TIMEOUT="${COORDINATOR_OLLAMA_LOAD_TIMEOUT}" \
  "${ollama_exec}" serve >"${COORDINATOR_OLLAMA_LOG}" 2>&1 &
  COORDINATOR_LOCAL_LLM_PID=$!
  echo "Coordinator Ollama pid: ${COORDINATOR_LOCAL_LLM_PID}"
  echo "Coordinator Ollama log: ${COORDINATOR_OLLAMA_LOG}"
  echo "Coordinator Ollama base URL: ${COORDINATOR_OLLAMA_BASE_URL}"
  echo "Coordinator Ollama model: ${COORDINATOR_OLLAMA_MODEL}"

  ready=0
  ready_url="${COORDINATOR_OLLAMA_BASE_URL%/v1}/api/tags"
  max_attempts=$((COORDINATOR_LLM_STARTUP_TIMEOUT / COORDINATOR_LLM_STARTUP_POLL_INTERVAL))
  if [[ "${max_attempts}" -lt 1 ]]; then
    max_attempts=1
  fi
  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    if curl -fsS --max-time 5 "${ready_url}" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${COORDINATOR_LOCAL_LLM_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep "${COORDINATOR_LLM_STARTUP_POLL_INTERVAL}"
  done

  if [[ "${ready}" != "1" ]]; then
    echo "WARN: coordinator Ollama server did not become ready at ${COORDINATOR_OLLAMA_BASE_URL}; falling back to remote OpenAI."
    if [[ -f "${COORDINATOR_OLLAMA_LOG}" ]]; then
      tail -n 100 "${COORDINATOR_OLLAMA_LOG}" || true
    fi
    cleanup_coordinator_llm
    return 1
  fi

  if ! ensure_coordinator_ollama_model "${COORDINATOR_OLLAMA_MODEL}" "${CRITIC_OLLAMA_MODEL}"; then
    echo "WARN: coordinator Ollama model could not be prepared; falling back to remote OpenAI."
    cleanup_coordinator_llm
    return 1
  fi

  if ! prewarm_coordinator_ollama_model "${COORDINATOR_OLLAMA_MODEL}" "${CRITIC_OLLAMA_MODEL}"; then
    echo "WARN: coordinator Ollama prewarm failed; falling back to remote OpenAI."
    cleanup_coordinator_llm
    return 1
  fi

  echo "[$(date '+%F %T')] READY: coordinator local Ollama server"
  export OLLAMA_BASE_URL="${COORDINATOR_OLLAMA_BASE_URL}"
  export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
  export CRITIC_OLLAMA_BASE_URL="${CRITIC_OLLAMA_BASE_URL}"
  if [[ "${COORDINATOR_USE_LOCAL_LLM}" == "1" ]]; then
    export OLLAMA_MODEL="${COORDINATOR_OLLAMA_MODEL}"
    export LLM_PROVIDER_ORDER="ollama-first"
  fi
  return 0
}

if start_coordinator_local_llm; then
  if [[ "${COORDINATOR_USE_LOCAL_LLM}" == "1" ]]; then
    echo "Coordinator provider order: ollama-first with remote ${OPENAI_MODEL} fallback"
  else
    export LLM_PROVIDER_ORDER="openai-first"
    echo "Coordinator planner: openai-first (${OPENAI_MODEL}), critic local model: ${CRITIC_OLLAMA_MODEL}"
  fi
else
  export LLM_PROVIDER_ORDER="openai-first"
  if [[ "${CRITIC_ENABLED}" == "1" && "${CRITIC_USE_LOCAL_LLM}" == "1" ]]; then
    export CRITIC_USE_LOCAL_LLM=0
    export CRITIC_PROVIDER_ORDER="openai-first"
    echo "Critic local LLM disabled because coordinator local LLM startup failed."
  fi
  if [[ -z "${OPENAI_API_KEY:-}" && -z "${OPENAI_BASE_URL:-}" ]]; then
    echo "ERROR: No coordinator LLM is available."
    echo "Set OPENAI_API_KEY for remote fallback or install/configure local Ollama for the CPU coordinator."
    exit 1
  fi
  echo "Coordinator provider order: openai-first (${OPENAI_MODEL})"
fi

if [[ "${RESET_CHECKPOINT:-0}" == "1" ]]; then
  rm -f "${CHECKPOINT_FILE}"
fi
touch "${CHECKPOINT_FILE}"

echo "Experiment file: ${EXPERIMENT_FILE}"
echo "Baseline experiment file: ${BASELINE_EXPERIMENT_FILE}"
echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo "State dir: ${STATE_DIR}"
echo "Repository root: ${REPO_ROOT}"
echo "PACE venv root: ${PACE_VENV_ROOT:-<repo-local>}"
echo "Virtualenv dir: ${VENV_DIR}"
echo "Python executable: ${VENV_PY}"
echo "Coordinator remote fallback model: ${OPENAI_MODEL}"
echo "Coordinator local LLM enabled: ${COORDINATOR_USE_LOCAL_LLM}"
echo "Coordinator local backend: ${COORDINATOR_LOCAL_LLM_BACKEND}"
echo "Install vLLM during bootstrap: ${INSTALL_VLLM}"
echo "Coordinator local model: ${COORDINATOR_OLLAMA_MODEL}"
echo "Coordinator shared local model across planner/critic: ${COORDINATOR_SHARED_LOCAL_MODEL}"
echo "Coordinator vLLM model: ${COORDINATOR_VLLM_MODEL:-<disabled>}"
echo "Coordinator vLLM isolated venv: ${COORDINATOR_VLLM_ISOLATED_VENV}"
echo "Coordinator vLLM venv dir: ${COORDINATOR_VLLM_VENV_DIR}"
echo "Coordinator vLLM base URL: ${COORDINATOR_VLLM_BASE_URL}"
echo "Coordinator vLLM visible devices: ${COORDINATOR_VLLM_VISIBLE_DEVICES}"
echo "Coordinator vLLM tensor parallel size: ${COORDINATOR_VLLM_TENSOR_PARALLEL_SIZE}"
echo "Critic enabled: ${CRITIC_ENABLED}"
echo "Critic local LLM enabled: ${CRITIC_USE_LOCAL_LLM}"
echo "Critic provider order: ${CRITIC_PROVIDER_ORDER}"
echo "Critic local model: ${CRITIC_OLLAMA_MODEL}"
echo "Critic remote fallback model: ${CRITIC_OPENAI_MODEL}"
echo "Max workers: ${MAX_WORKERS}"
echo "GPU count for worker assignment: ${ORCHESTRATOR_GPU_COUNT}"
echo "GPU slots for worker assignment: ${ORCHESTRATOR_GPU_SLOTS:-<round-robin>}"
echo "Worker GPU offset: ${WORKER_GPU_OFFSET}"
echo "Adaptive GPU throttle enabled: ${ORCHESTRATOR_ADAPTIVE_GPU_THROTTLE}"
echo "GPU overload util threshold: ${ORCHESTRATOR_GPU_OVERLOAD_UTIL_THRESHOLD}"
echo "GPU overload consecutive samples: ${ORCHESTRATOR_GPU_OVERLOAD_CONSECUTIVE_SAMPLES}"
echo "Max queue: ${MAX_QUEUE}"
echo "Coordinator Ollama use GPU: ${COORDINATOR_OLLAMA_USE_GPU}"
echo "Coordinator Ollama GPU id: ${OLLAMA_GPU_ID}"
echo "Coordinator Ollama keep alive: ${COORDINATOR_OLLAMA_KEEP_ALIVE}"
echo "Coordinator Ollama max loaded models: ${COORDINATOR_OLLAMA_MAX_LOADED_MODELS}"
echo "Coordinator Ollama num parallel: ${COORDINATOR_OLLAMA_NUM_PARALLEL}"
echo "Current default solver stack: ${SMB_CURRENT_DEFAULT_SOLVER_STACK}"
echo "Available solver options: ${SMB_AVAILABLE_SOLVER_OPTIONS}"
echo "Available acceleration options: ${SMB_AVAILABLE_ACCELERATION_OPTIONS}"
echo "Resource decision policy: ${SMB_RESOURCE_DECISION_POLICY}"
echo "IPOPT resource file: ${SMB_IPOPT_RESOURCE_FILE}"
echo "IPOPT profile menu: ${SMB_IPOPT_PROFILE_MENU}"
echo "IPOPT baseline fallback tree: ${SMB_IPOPT_BASELINE_FALLBACK_TREE}"
echo "IPOPT screening fallback tree: ${SMB_IPOPT_SCREENING_FALLBACK_TREE}"
echo "IPOPT hard-problem fallback tree: ${SMB_IPOPT_HARD_PROBLEM_FALLBACK_TREE}"
echo "IPOPT high-performance fallback tree: ${SMB_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE}"
echo "IPOPT precheck summary: ${SMB_IPOPT_PRECHECK_SUMMARY}"
echo "Verified IPOPT executables: ${SMB_VERIFIED_IPOPT_EXECUTABLES:-<none>}"
echo "Verified IPOPT linear solvers: ${SMB_VERIFIED_IPOPT_LINEAR_SOLVERS:-<none>}"
echo "Verified IPOPT profile menu: ${SMB_VERIFIED_IPOPT_PROFILE_MENU:-<none>}"
echo "Verified IPOPT baseline fallback tree: ${SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE:-<none>}"
echo "Verified IPOPT screening fallback tree: ${SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE:-<none>}"
echo "Verified IPOPT hard-problem fallback tree: ${SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE:-<none>}"
echo "Verified IPOPT high-performance fallback tree: ${SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE:-<none>}"
echo "Target experiments: ${NUM_EXPERIMENTS}"

prepare_key="${EXPERIMENT_FILE}:prepare-data"
if grep -Fxq "${prepare_key}" "${CHECKPOINT_FILE}"; then
  echo "[$(date '+%F %T')] SKIP (already done): ${prepare_key}"
else
  echo "[$(date '+%F %T')] START: ${prepare_key}"
  "${VENV_PY}" -m "${CLI_MODULE}" prepare-data --experiment "${EXPERIMENT_FILE}"
  echo "${prepare_key}" >> "${CHECKPOINT_FILE}"
  echo "[$(date '+%F %T')] DONE: ${prepare_key}"
fi

baseline_key="${BASELINE_EXPERIMENT_FILE}:train-baselines"
if [[ "${ENSURE_BASELINES}" == "1" ]] || [[ ! -f "${BASELINE_LEADERBOARD}" ]]; then
  if grep -Fxq "${baseline_key}" "${CHECKPOINT_FILE}"; then
    echo "[$(date '+%F %T')] SKIP (already done): ${baseline_key}"
  else
    echo "[$(date '+%F %T')] START: ${baseline_key}"
    "${VENV_PY}" -m "${CLI_MODULE}" train-baselines --experiment "${BASELINE_EXPERIMENT_FILE}"
    echo "${baseline_key}" >> "${CHECKPOINT_FILE}"
    echo "[$(date '+%F %T')] DONE: ${baseline_key}"
  fi
fi

orchestrator_key="${EXPERIMENT_FILE}:run-orchestrated-graph-dev"
if grep -Fxq "${orchestrator_key}" "${CHECKPOINT_FILE}"; then
  echo "[$(date '+%F %T')] SKIP (already done): ${orchestrator_key}"
else
  echo "[$(date '+%F %T')] START: ${orchestrator_key}"
  gpu_slot_args=()
  if [[ -n "${ORCHESTRATOR_GPU_SLOTS}" ]]; then
    gpu_slot_args+=(--gpu-slots "${ORCHESTRATOR_GPU_SLOTS}")
  fi
  "${VENV_PY}" -m "${CLI_MODULE}" run-orchestrated-graph-dev \
    --experiment "${EXPERIMENT_FILE}" \
    --baseline-experiment "${BASELINE_EXPERIMENT_FILE}" \
    --state-dir "${STATE_DIR}" \
    --max-workers "${MAX_WORKERS}" \
    --gpu-count "${ORCHESTRATOR_GPU_COUNT}" \
    --worker-gpu-offset "${WORKER_GPU_OFFSET}" \
    --max-queue "${MAX_QUEUE}" \
    --max-experiments "${NUM_EXPERIMENTS}" \
    "${gpu_slot_args[@]}"
  echo "${orchestrator_key}" >> "${CHECKPOINT_FILE}"
  echo "[$(date '+%F %T')] DONE: ${orchestrator_key}"
fi

echo "[$(date '+%F %T')] graph orchestrator workflow complete."
