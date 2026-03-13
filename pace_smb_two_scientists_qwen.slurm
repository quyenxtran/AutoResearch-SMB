#!/bin/bash
# Two-scientist SMB optimization job for the local SembaSMB reference model.
#
# Suggested submissions:
# sbatch AutoResearch-SMB/pace_smb_two_scientists_qwen.slurm
# sbatch --export=ALL,START_LOCAL_LLM=1 AutoResearch-SMB/pace_smb_two_scientists_qwen.slurm
# sbatch --export=ALL,START_LOCAL_LLM=1,RUN_BASELINE_NOTEBOOK=1 AutoResearch-SMB/pace_smb_two_scientists_qwen.slurm
# sbatch --export=ALL,START_LOCAL_LLM=1,AGENT_ENTRYPOINT="python -m your_package.your_runner" AutoResearch-SMB/pace_smb_two_scientists_qwen.slurm
#
# This file is based on the existing PACE sample Slurm file, but it is narrowed to the
# SembaSMB optimization workflow and exports an explicit compute budget so the two
# scientists can plan around the real resources of the job.

#SBATCH --job-name=smb-two-scientists
#SBATCH -A gts-sn73
#SBATCH --qos=inferno
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
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
if [[ -z "${AUTORESEARCH_ROOT:-}" ]]; then
  if [[ -f "${REPO_ROOT}/LLM_SOUL.md" && -d "${REPO_ROOT}/SembaSMB" ]]; then
    AUTORESEARCH_ROOT="${REPO_ROOT}"
  else
    AUTORESEARCH_ROOT="${REPO_ROOT}/AutoResearch-SMB"
  fi
fi
if [[ -z "${SMB_ROOT:-}" ]]; then
  if [[ -d "${AUTORESEARCH_ROOT}/SembaSMB" ]]; then
    SMB_ROOT="${AUTORESEARCH_ROOT}/SembaSMB"
  else
    SMB_ROOT="${REPO_ROOT}/SembaSMB"
  fi
fi
OBJECTIVES_FILE="${OBJECTIVES_FILE:-$AUTORESEARCH_ROOT/Objectives.md}"
LLM_SOUL_FILE="${LLM_SOUL_FILE:-$AUTORESEARCH_ROOT/LLM_SOUL.md}"
IPOPT_RESOURCE_FILE="${IPOPT_RESOURCE_FILE:-$AUTORESEARCH_ROOT/IPOPT_SOLVER_RESOURCES.md}"
NOTEBOOK_REFERENCE="${NOTEBOOK_REFERENCE:-$SMB_ROOT/SMB_KratonFeed_updated_config copy 6.ipynb}"

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
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PACE_VENV_ROOT/AutoResearch-SMB/.pip-cache}"
fi
if [[ -z "${VENV_PY:-}" ]]; then
  venv_candidates=()
  if [[ -n "${VENV_DIR:-}" ]]; then
    venv_candidates+=("${VENV_DIR}/bin/python")
  fi
  if [[ -n "${PACE_VENV_ROOT}" ]]; then
    venv_candidates+=("${PACE_VENV_ROOT}/AutoResearch-SMB/.venv/bin/python")
  fi
  venv_candidates+=("${REPO_ROOT}/.venv/bin/python")
  venv_candidates+=("${AUTORESEARCH_ROOT}/.venv/bin/python")
  venv_candidates+=("${SMB_ROOT}/.venv/bin/python")
  if [[ -n "${HOME:-}" ]]; then
    venv_candidates+=("${HOME}/AutoResearch-SMB/.venv/bin/python")
    venv_candidates+=("${HOME}/SembaSMB/.venv/bin/python")
  fi

  for candidate in "${venv_candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      VENV_PY="${candidate}"
      VENV_DIR="$(cd "$(dirname "${candidate}")/.." && pwd)"
      break
    fi
  done
fi

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/artifacts/two_scientist_smb}"
mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"

export PYTHONUNBUFFERED=1

if [[ ! -x "${VENV_PY:-}" ]]; then
  echo "ERROR: expected virtualenv python not found at ${VENV_PY}"
  echo "Checked common candidates:"
  if [[ -n "${PACE_VENV_ROOT}" ]]; then
    echo "  - ${PACE_VENV_ROOT}/AutoResearch-SMB/.venv/bin/python"
  fi
  echo "  - ${REPO_ROOT}/.venv/bin/python"
  echo "  - ${AUTORESEARCH_ROOT}/.venv/bin/python"
  echo "  - ${SMB_ROOT}/.venv/bin/python"
  if [[ -n "${HOME:-}" ]]; then
    echo "  - ${HOME}/AutoResearch-SMB/.venv/bin/python"
    echo "  - ${HOME}/SembaSMB/.venv/bin/python"
  fi
  echo "Override with --export=ALL,VENV_DIR=/path/to/.venv or VENV_PY=/path/to/python"
  exit 1
fi

export PATH="${VENV_DIR}/bin:${PATH}"
if [[ -d "${HOME:-}/.local/ollama/bin" ]]; then
  export PATH="${HOME}/.local/ollama/bin:${PATH}"
fi

if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE}" != "0" ]]; then
  SMB_MEMORY_GB_COMPUTED="$(( (SLURM_MEM_PER_NODE + 1023) / 1024 ))"
else
  SMB_MEMORY_GB_COMPUTED="32"
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  SMB_GPU_COUNT_COMPUTED="${SLURM_GPUS_ON_NODE}"
else
  SMB_GPU_COUNT_COMPUTED="1"
fi

export SMB_EXPERIMENT_NAME="${SMB_EXPERIMENT_NAME:-qwen_smb_two_scientists}"
export SMB_AGENT_COUNT="${SMB_AGENT_COUNT:-2}"
export SMB_AGENT_1_NAME="${SMB_AGENT_1_NAME:-Scientist_A}"
export SMB_AGENT_1_ROLE="${SMB_AGENT_1_ROLE:-process_optimization_scientist}"
export SMB_AGENT_2_NAME="${SMB_AGENT_2_NAME:-Scientist_B}"
export SMB_AGENT_2_ROLE="${SMB_AGENT_2_ROLE:-skeptical_process_and_numerics_scientist}"
export SMB_LOCAL_LLM_MODEL="${SMB_LOCAL_LLM_MODEL:-qwen3.5:9b}"
export SMB_CPU_TASKS="${SMB_CPU_TASKS:-${SLURM_CPUS_PER_TASK:-12}}"
export SMB_GPU_COUNT="${SMB_GPU_COUNT:-${SMB_GPU_COUNT_COMPUTED}}"
export SMB_GPU_MODEL="${SMB_GPU_MODEL:-rtx_6000}"
export SMB_MEMORY_GB="${SMB_MEMORY_GB:-${SMB_MEMORY_GB_COMPUTED}}"
export SMB_WALLTIME_HOURS="${SMB_WALLTIME_HOURS:-12}"
export SMB_NODE_COUNT="${SMB_NODE_COUNT:-${SLURM_JOB_NUM_NODES:-1}}"
export SMB_ALLOW_CPU_MODE="${SMB_ALLOW_CPU_MODE:-1}"
export SMB_ALLOW_GPU_MODE="${SMB_ALLOW_GPU_MODE:-1}"
export SMB_PREFERRED_SIMULATION_MODE="${SMB_PREFERRED_SIMULATION_MODE:-cpu}"
export SMB_PREFERRED_LLM_MODE="${SMB_PREFERRED_LLM_MODE:-gpu}"
export SMB_LOW_FIDELITY_HINT="${SMB_LOW_FIDELITY_HINT:-nfex=4,nfet=2,ncp=1}"
export SMB_MEDIUM_FIDELITY_HINT="${SMB_MEDIUM_FIDELITY_HINT:-nfex=6,nfet=3,ncp=2}"
export SMB_HIGH_FIDELITY_HINT="${SMB_HIGH_FIDELITY_HINT:-nfex=10,nfet=5,ncp=2,nc=(1,2,3,2)}"
export SMB_FFEED_BOUNDS_ML_MIN="${SMB_FFEED_BOUNDS_ML_MIN:-(0.0,2.5]}"
export SMB_MAX_PUMP_FLOW_ML_MIN="${SMB_MAX_PUMP_FLOW_ML_MIN:-2.5}"
export SMB_TARGET_PURITY_EX_MEOH_FREE="${SMB_TARGET_PURITY_EX_MEOH_FREE:-0.90}"
export SMB_TARGET_RECOVERY_GA="${SMB_TARGET_RECOVERY_GA:-0.90}"
export SMB_TARGET_RECOVERY_MA="${SMB_TARGET_RECOVERY_MA:-0.90}"
export SMB_OBJECTIVES_FILE="${SMB_OBJECTIVES_FILE:-${OBJECTIVES_FILE}}"
export SMB_LLM_SOUL_FILE="${SMB_LLM_SOUL_FILE:-${LLM_SOUL_FILE}}"
export SMB_IPOPT_RESOURCE_FILE="${SMB_IPOPT_RESOURCE_FILE:-${IPOPT_RESOURCE_FILE}}"
export SMB_REFERENCE_NOTEBOOK="${SMB_REFERENCE_NOTEBOOK:-${NOTEBOOK_REFERENCE}}"
export SMB_REFERENCE_MODEL_ROOT="${SMB_REFERENCE_MODEL_ROOT:-${SMB_ROOT}}"
export SMB_CURRENT_DEFAULT_SOLVER_STACK="${SMB_CURRENT_DEFAULT_SOLVER_STACK:-pyomo+ipopt_sens+ma57}"
export SMB_AVAILABLE_SOLVER_OPTIONS="${SMB_AVAILABLE_SOLVER_OPTIONS:-pyomo+ipopt_sens+ma57(current_default_cpu); pyomo+ipopt_sens+mumps(cpu_if_installed); pyomo+ipopt_sens+pardiso(cpu_if_installed); pyomo+ipopt_sens+ma97(cpu_if_installed); madnlp+cudss(experimental_gpu_if_installed)}"
export SMB_AVAILABLE_ACCELERATION_OPTIONS="${SMB_AVAILABLE_ACCELERATION_OPTIONS:-optimized_blas_lapack_if_available; warm_starts; cpu_parallel_screening; fidelity_ladder; gpu_for_qwen_inference; experimental_gpu_nlp_solver_if_installed}"
export SMB_RESOURCE_DECISION_POLICY="${SMB_RESOURCE_DECISION_POLICY:-model_may_choose_best_available_option_after_checking_installation_runtime_and_solution_quality}"
export SMB_IPOPT_PROFILE_MENU="${SMB_IPOPT_PROFILE_MENU:-baseline_reproduction; screening_fast; balanced_cpu_mumps; robust_cpu_ma97; robust_cpu_mumps; high_end_pardiso; validation_strict}"
export SMB_IPOPT_BASELINE_FALLBACK_TREE="${SMB_IPOPT_BASELINE_FALLBACK_TREE:-baseline_reproduction -> robust_cpu_mumps -> validation_strict}"
export SMB_IPOPT_SCREENING_FALLBACK_TREE="${SMB_IPOPT_SCREENING_FALLBACK_TREE:-screening_fast -> balanced_cpu_mumps -> baseline_reproduction}"
export SMB_IPOPT_HARD_PROBLEM_FALLBACK_TREE="${SMB_IPOPT_HARD_PROBLEM_FALLBACK_TREE:-robust_cpu_ma97 -> robust_cpu_mumps -> validation_strict}"
export SMB_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE="${SMB_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE:-high_end_pardiso -> robust_cpu_ma97 -> robust_cpu_mumps}"
export SMB_COMPUTE_SUMMARY="nodes=${SMB_NODE_COUNT}; cpus=${SMB_CPU_TASKS}; gpus=${SMB_GPU_COUNT}x${SMB_GPU_MODEL}; mem_gb=${SMB_MEMORY_GB}; walltime_h=${SMB_WALLTIME_HOURS}; preferred_sim=${SMB_PREFERRED_SIMULATION_MODE}; preferred_llm=${SMB_PREFERRED_LLM_MODE}"

START_LOCAL_LLM="${START_LOCAL_LLM:-0}"
LOCAL_LLM_BACKEND="${LOCAL_LLM_BACKEND:-ollama}"
LOCAL_LLM_USE_GPU="${LOCAL_LLM_USE_GPU:-1}"
OLLAMA_MODEL="${OLLAMA_MODEL:-${SMB_LOCAL_LLM_MODEL}}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://${OLLAMA_HOST}/v1}"
OLLAMA_LOG="${OLLAMA_LOG:-$LOG_DIR/ollama-smb-${SLURM_JOB_ID:-local}.log}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"
OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-10m}"
OLLAMA_GPU_ID="${OLLAMA_GPU_ID:-0}"
LOCAL_LLM_PID=""
SMB_IPOPT_PRECHECK_SUMMARY=""
SMB_VERIFIED_IPOPT_EXECUTABLES=""
SMB_VERIFIED_IPOPT_LINEAR_SOLVERS=""
SMB_VERIFIED_IPOPT_PROFILE_MENU=""
SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE=""

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

resolve_ollama_bin() {
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

cleanup_local_llm() {
  if [[ -n "${LOCAL_LLM_PID:-}" ]] && kill -0 "${LOCAL_LLM_PID}" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] STOP: local LLM server pid=${LOCAL_LLM_PID}"
    kill "${LOCAL_LLM_PID}" >/dev/null 2>&1 || true
    wait "${LOCAL_LLM_PID}" >/dev/null 2>&1 || true
  fi
  LOCAL_LLM_PID=""
}

trap cleanup_local_llm EXIT

start_local_qwen() {
  local ollama_exec ready ready_url max_attempts prompt_payload response_file http_code
  if [[ "${START_LOCAL_LLM}" != "1" ]]; then
    return 0
  fi
  if [[ "${LOCAL_LLM_BACKEND}" != "ollama" ]]; then
    echo "WARN: only LOCAL_LLM_BACKEND=ollama is wired in this SMB Slurm template."
    return 1
  fi
  if ! ollama_exec="$(resolve_ollama_bin)"; then
    echo "WARN: ollama not found on PATH; local Qwen will not be started."
    return 1
  fi

  local visible_devices
  if [[ "${LOCAL_LLM_USE_GPU}" == "1" ]]; then
    visible_devices="${OLLAMA_GPU_ID}"
    echo "[$(date '+%F %T')] START: local Ollama server for Qwen on GPU ${OLLAMA_GPU_ID}"
  else
    visible_devices=""
    echo "[$(date '+%F %T')] START: local Ollama server for Qwen on CPU"
  fi

  CUDA_VISIBLE_DEVICES="${visible_devices}" \
  OLLAMA_HOST="${OLLAMA_HOST}" \
  OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}" \
  OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS}" \
  OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" \
  OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT}" \
  "${ollama_exec}" serve >"${OLLAMA_LOG}" 2>&1 &
  LOCAL_LLM_PID=$!

  ready=0
  ready_url="${OLLAMA_BASE_URL%/v1}/api/tags"
  max_attempts=30
  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    if curl -fsS --max-time 5 "${ready_url}" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${LOCAL_LLM_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 3
  done

  if [[ "${ready}" != "1" ]]; then
    echo "WARN: local Ollama server did not become ready."
    if [[ -f "${OLLAMA_LOG}" ]]; then
      tail -n 100 "${OLLAMA_LOG}" || true
    fi
    cleanup_local_llm
    return 1
  fi

  if ! OLLAMA_HOST="${OLLAMA_HOST}" "${ollama_exec}" list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${OLLAMA_MODEL}"; then
    echo "[$(date '+%F %T')] PULL: ${OLLAMA_MODEL}"
    OLLAMA_HOST="${OLLAMA_HOST}" "${ollama_exec}" pull "${OLLAMA_MODEL}"
  fi

  prompt_payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"stream":false}' "${OLLAMA_MODEL}")
  response_file="$(mktemp)"
  http_code="$(curl -sS \
    --max-time 60 \
    -o "${response_file}" \
    -w '%{http_code}' \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${OLLAMA_API_KEY:-ollama}" \
    -d "${prompt_payload}" \
    "${OLLAMA_BASE_URL%/}/chat/completions" || true)"
  rm -f "${response_file}"

  if [[ ! "${http_code}" =~ ^2 ]]; then
    echo "WARN: local Qwen prewarm failed with status=${http_code:-<curl-error>}"
    cleanup_local_llm
    return 1
  fi

  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL}"
  export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
  export OLLAMA_MODEL="${OLLAMA_MODEL}"
  echo "[$(date '+%F %T')] READY: local Qwen available at ${OLLAMA_BASE_URL}"
  return 0
}

probe_ipopt_resources

cd "${REPO_ROOT}"

cat > "${ARTIFACT_DIR}/compute_manifest.${SLURM_JOB_ID:-local}.md" <<EOF
# Two Scientist SMB Compute Manifest

- job_id: ${SLURM_JOB_ID:-local}
- experiment: ${SMB_EXPERIMENT_NAME}
- scientist_count: ${SMB_AGENT_COUNT}
- scientist_1: ${SMB_AGENT_1_NAME} (${SMB_AGENT_1_ROLE})
- scientist_2: ${SMB_AGENT_2_NAME} (${SMB_AGENT_2_ROLE})
- compute_summary: ${SMB_COMPUTE_SUMMARY}
- objectives_file: ${SMB_OBJECTIVES_FILE}
- llm_soul_file: ${SMB_LLM_SOUL_FILE}
- ipopt_resource_file: ${SMB_IPOPT_RESOURCE_FILE}
- reference_model_root: ${SMB_REFERENCE_MODEL_ROOT}
- reference_notebook: ${SMB_REFERENCE_NOTEBOOK}
- ffeed_bounds_ml_min: ${SMB_FFEED_BOUNDS_ML_MIN}
- max_pump_flow_ml_min: ${SMB_MAX_PUMP_FLOW_ML_MIN}
- target_purity_ex_meoh_free: ${SMB_TARGET_PURITY_EX_MEOH_FREE}
- target_recovery_ga: ${SMB_TARGET_RECOVERY_GA}
- target_recovery_ma: ${SMB_TARGET_RECOVERY_MA}
- current_default_solver_stack: ${SMB_CURRENT_DEFAULT_SOLVER_STACK}
- available_solver_options: ${SMB_AVAILABLE_SOLVER_OPTIONS}
- available_acceleration_options: ${SMB_AVAILABLE_ACCELERATION_OPTIONS}
- resource_decision_policy: ${SMB_RESOURCE_DECISION_POLICY}
- ipopt_profile_menu: ${SMB_IPOPT_PROFILE_MENU}
- ipopt_baseline_fallback_tree: ${SMB_IPOPT_BASELINE_FALLBACK_TREE}
- ipopt_screening_fallback_tree: ${SMB_IPOPT_SCREENING_FALLBACK_TREE}
- ipopt_hard_problem_fallback_tree: ${SMB_IPOPT_HARD_PROBLEM_FALLBACK_TREE}
- ipopt_high_performance_fallback_tree: ${SMB_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE}
- ipopt_precheck_summary: ${SMB_IPOPT_PRECHECK_SUMMARY}
- verified_ipopt_executables: ${SMB_VERIFIED_IPOPT_EXECUTABLES}
- verified_ipopt_linear_solvers: ${SMB_VERIFIED_IPOPT_LINEAR_SOLVERS}
- verified_ipopt_profile_menu: ${SMB_VERIFIED_IPOPT_PROFILE_MENU}
- verified_ipopt_baseline_fallback_tree: ${SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE}
- verified_ipopt_screening_fallback_tree: ${SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE}
- verified_ipopt_hard_problem_fallback_tree: ${SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE}
- verified_ipopt_high_performance_fallback_tree: ${SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE}
- low_fidelity_hint: ${SMB_LOW_FIDELITY_HINT}
- medium_fidelity_hint: ${SMB_MEDIUM_FIDELITY_HINT}
- high_fidelity_hint: ${SMB_HIGH_FIDELITY_HINT}
EOF

echo "Repository root: ${REPO_ROOT}"
echo "SMB root: ${SMB_ROOT}"
echo "Objectives file: ${OBJECTIVES_FILE}"
echo "LLM soul file: ${LLM_SOUL_FILE}"
echo "IPOPT resource file: ${IPOPT_RESOURCE_FILE}"
echo "Reference notebook: ${NOTEBOOK_REFERENCE}"
echo "Python executable: ${VENV_PY}"
echo "Compute summary: ${SMB_COMPUTE_SUMMARY}"
echo "Current default solver stack: ${SMB_CURRENT_DEFAULT_SOLVER_STACK}"
echo "Available solver options: ${SMB_AVAILABLE_SOLVER_OPTIONS}"
echo "Available acceleration options: ${SMB_AVAILABLE_ACCELERATION_OPTIONS}"
echo "Resource decision policy: ${SMB_RESOURCE_DECISION_POLICY}"
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
echo "Compute manifest: ${ARTIFACT_DIR}/compute_manifest.${SLURM_JOB_ID:-local}.md"

start_local_qwen || true

RUN_BASELINE_NOTEBOOK="${RUN_BASELINE_NOTEBOOK:-0}"
if [[ "${RUN_BASELINE_NOTEBOOK}" == "1" ]]; then
  echo "[$(date '+%F %T')] START: execute reference notebook baseline"
  "${VENV_PY}" -m jupyter nbconvert --execute --to notebook --inplace "${NOTEBOOK_REFERENCE}"
  echo "[$(date '+%F %T')] DONE: execute reference notebook baseline"
fi

AGENT_ENTRYPOINT="${AGENT_ENTRYPOINT:-}"
if [[ -n "${AGENT_ENTRYPOINT}" ]]; then
  echo "[$(date '+%F %T')] START: AGENT_ENTRYPOINT=${AGENT_ENTRYPOINT}"
  bash -lc "${AGENT_ENTRYPOINT}"
  echo "[$(date '+%F %T')] DONE: agent entrypoint"
else
  echo "No AGENT_ENTRYPOINT provided."
  echo "This job exported the two-scientist SMB compute budget and optional local Qwen environment."
  echo "Set AGENT_ENTRYPOINT when the autonomous runner is ready."
fi

echo "[$(date '+%F %T')] SMB two-scientist job setup complete."
