#!/bin/bash
# Two-scientist SMB optimization job for the local SembaSMB reference model.
#
# Suggested submissions:
# sbatch AutoResearch-SMB/slurm/pace_smb_two_scientists_qwen.slurm
# sbatch --export=ALL,START_LOCAL_LLM=1 AutoResearch-SMB/slurm/pace_smb_two_scientists_qwen.slurm
# sbatch --export=ALL,START_LOCAL_LLM=1,RUN_BASELINE_NOTEBOOK=1 AutoResearch-SMB/slurm/pace_smb_two_scientists_qwen.slurm
# sbatch --export=ALL,START_LOCAL_LLM=1,AGENT_ENTRYPOINT="python -m benchmarks.agent_runner" AutoResearch-SMB/slurm/pace_smb_two_scientists_qwen.slurm
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
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
  if [[ -f "${REPO_ROOT}/benchmarks/agent_runner.py" && -d "${REPO_ROOT}/src/sembasmb" ]]; then
    AUTORESEARCH_ROOT="${REPO_ROOT}"
  else
    AUTORESEARCH_ROOT="${REPO_ROOT}/AutoResearch-SMB"
  fi
fi
if [[ -z "${SMB_ROOT:-}" ]]; then
  SMB_ROOT="${AUTORESEARCH_ROOT}/src"
fi
OBJECTIVES_FILE="${OBJECTIVES_FILE:-$AUTORESEARCH_ROOT/agents/Objectives.md}"
LLM_SOUL_FILE="${LLM_SOUL_FILE:-$AUTORESEARCH_ROOT/agents/LLM_SOUL.md}"
IPOPT_RESOURCE_FILE="${IPOPT_RESOURCE_FILE:-$AUTORESEARCH_ROOT/agents/IPOPT_SOLVER_RESOURCES.md}"
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
echo "[$(date '+%F %T')] BOOT: SMB two-scientists script started on $(hostname)"
echo "[$(date '+%F %T')] BOOT: REPO_ROOT=${REPO_ROOT}"
echo "[$(date '+%F %T')] BOOT: LOG_DIR=${LOG_DIR} ARTIFACT_DIR=${ARTIFACT_DIR}"

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
  fi
  echo "Override with --export=ALL,VENV_DIR=/path/to/.venv or VENV_PY=/path/to/python"
  exit 1
fi

export PATH="${VENV_DIR}/bin:${PATH}"
if [[ -d "${HOME:-}/.local/ollama/bin" ]]; then
  export PATH="${HOME}/.local/ollama/bin:${PATH}"
fi

# Ensure scratch-installed IPOPT is visible in batch jobs (same pattern as stage runner).
IPOPT_BIN_DIR="${IPOPT_BIN_DIR:-}"
if [[ -z "${IPOPT_BIN_DIR}" && -n "${DEFAULT_SCRATCH_ROOT}" ]]; then
  IPOPT_BIN_DIR="${DEFAULT_SCRATCH_ROOT}/opt/ipopt/bin"
fi
IPOPT_LIB_DIR="${IPOPT_LIB_DIR:-}"
if [[ -z "${IPOPT_LIB_DIR}" && -n "${DEFAULT_SCRATCH_ROOT}" ]]; then
  IPOPT_LIB_DIR="${DEFAULT_SCRATCH_ROOT}/opt/ipopt/lib"
fi
OPENBLAS_LIBDIR="${OPENBLAS_LIBDIR:-/usr/local/pace-apps/manual/packages/fenics/2018/linux-rhel9-cascadelake/gcc-9.4.0/openblas-0.3.28-cg7h4gbbmohnlrdesf4ykgipgza67why/lib}"

if [[ -n "${IPOPT_BIN_DIR}" && -d "${IPOPT_BIN_DIR}" ]]; then
  export PATH="${IPOPT_BIN_DIR}:${PATH}"
fi

runtime_libs=()
if [[ -n "${IPOPT_LIB_DIR}" && -d "${IPOPT_LIB_DIR}" ]]; then
  runtime_libs+=("${IPOPT_LIB_DIR}")
fi
if [[ -n "${OPENBLAS_LIBDIR}" && -d "${OPENBLAS_LIBDIR}" ]]; then
  runtime_libs+=("${OPENBLAS_LIBDIR}")
fi
if [[ ${#runtime_libs[@]} -gt 0 ]]; then
  runtime_libs_joined="$(IFS=:; echo "${runtime_libs[*]}")"
  export LD_LIBRARY_PATH="${runtime_libs_joined}:${LD_LIBRARY_PATH:-}"
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
export SMB_EXECUTIVE_LLM_MODEL="${SMB_EXECUTIVE_LLM_MODEL:-deepseek-r1:7b}"
export SMB_CPU_TASKS="${SMB_CPU_TASKS:-${SLURM_CPUS_PER_TASK:-6}}"
export SMB_IPOPT_WORKERS="${SMB_IPOPT_WORKERS:-2}"
export SMB_IPOPT_THREADS_PER_WORKER="${SMB_IPOPT_THREADS_PER_WORKER:-2}"
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
export SMB_MAX_PUMP_FLOW_RAF_ML_MIN="${SMB_MAX_PUMP_FLOW_RAF_ML_MIN:-5.0}"
export SMB_FRAF_BOUNDS="${SMB_FRAF_BOUNDS:-0.5,5.0}"
export SMB_TSTEP_BOUNDS="${SMB_TSTEP_BOUNDS:-8.0,12.0}"
# Exploratory search targets (used in search stage)
export SMB_TARGET_PURITY_EX_MEOH_FREE="${SMB_TARGET_PURITY_EX_MEOH_FREE:-0.60}"
export SMB_TARGET_RECOVERY_GA="${SMB_TARGET_RECOVERY_GA:-0.75}"
export SMB_TARGET_RECOVERY_MA="${SMB_TARGET_RECOVERY_MA:-0.75}"
# Project objective targets (used for strict validation/final acceptance)
export SMB_PROJECT_TARGET_PURITY_EX_MEOH_FREE="${SMB_PROJECT_TARGET_PURITY_EX_MEOH_FREE:-0.60}"
export SMB_PROJECT_TARGET_RECOVERY_GA="${SMB_PROJECT_TARGET_RECOVERY_GA:-0.75}"
export SMB_PROJECT_TARGET_RECOVERY_MA="${SMB_PROJECT_TARGET_RECOVERY_MA:-0.75}"
export SMB_BENCHMARK_HOURS="${SMB_BENCHMARK_HOURS:-12.0}"
export SMB_SEARCH_BUDGET_HOURS="${SMB_SEARCH_BUDGET_HOURS:-10.0}"
export SMB_VALIDATION_BUDGET_HOURS="${SMB_VALIDATION_BUDGET_HOURS:-2.0}"
export SMB_MIN_PROBE_REFERENCE_RUNS="${SMB_MIN_PROBE_REFERENCE_RUNS:-3}"
export SMB_SINGLE_SCIENTIST_MODE="${SMB_SINGLE_SCIENTIST_MODE:-1}"
export SMB_PROBE_LOW_FIDELITY_ENABLED="${SMB_PROBE_LOW_FIDELITY_ENABLED:-1}"
export SMB_PROBE_NFEX="${SMB_PROBE_NFEX:-5}"
export SMB_PROBE_NFET="${SMB_PROBE_NFET:-2}"
export SMB_PROBE_NCP="${SMB_PROBE_NCP:-1}"
export SMB_FINALIZATION_HARD_GATE_ENABLED="${SMB_FINALIZATION_HARD_GATE_ENABLED:-1}"
export SMB_FINALIZATION_LOW_FIDELITY_NFEX="${SMB_FINALIZATION_LOW_FIDELITY_NFEX:-${SMB_PROBE_NFEX}}"
export SMB_FINALIZATION_LOW_FIDELITY_NFET="${SMB_FINALIZATION_LOW_FIDELITY_NFET:-${SMB_PROBE_NFET}}"
export SMB_FINALIZATION_LOW_FIDELITY_NCP="${SMB_FINALIZATION_LOW_FIDELITY_NCP:-${SMB_PROBE_NCP}}"
export SMB_IPOPT_MAX_ITER="${SMB_IPOPT_MAX_ITER:-1000}"
export SMB_IPOPT_TOL="${SMB_IPOPT_TOL:-1e-5}"
export SMB_IPOPT_ACCEPTABLE_TOL="${SMB_IPOPT_ACCEPTABLE_TOL:-1e-4}"
export SMB_EXECUTIVE_CONTROLLER_ENABLED="${SMB_EXECUTIVE_CONTROLLER_ENABLED:-1}"
export SMB_EXECUTIVE_TRIGGER_REJECTS="${SMB_EXECUTIVE_TRIGGER_REJECTS:-1}"
export SMB_EXECUTIVE_FORCE_AFTER_REJECTS="${SMB_EXECUTIVE_FORCE_AFTER_REJECTS:-2}"
export SMB_EXECUTIVE_TOP_K_LOCK="${SMB_EXECUTIVE_TOP_K_LOCK:-3}"
export SMB_EXECUTIVE_LIVE_MONITOR="${SMB_EXECUTIVE_LIVE_MONITOR:-1}"
export SMB_EXECUTIVE_MONITOR_POLL_SECONDS="${SMB_EXECUTIVE_MONITOR_POLL_SECONDS:-1.0}"
export SMB_EXECUTIVE_MONITOR_WINDOW_ITERS="${SMB_EXECUTIVE_MONITOR_WINDOW_ITERS:-12}"
export SMB_EXECUTIVE_MONITOR_STALL_EPS="${SMB_EXECUTIVE_MONITOR_STALL_EPS:-0.01}"
export SMB_EXECUTIVE_WATCHDOG_ENABLED="${SMB_EXECUTIVE_WATCHDOG_ENABLED:-1}"
export SMB_EXECUTIVE_WATCHDOG_MIN_ITERS="${SMB_EXECUTIVE_WATCHDOG_MIN_ITERS:-80}"
export SMB_EXECUTIVE_WATCHDOG_STALL_WINDOWS="${SMB_EXECUTIVE_WATCHDOG_STALL_WINDOWS:-2}"
export SMB_EXECUTIVE_WATCHDOG_MAX_INF_DU="${SMB_EXECUTIVE_WATCHDOG_MAX_INF_DU:-0}"
export SMB_EXECUTIVE_WATCHDOG_MAX_MUMPS_REALLOC="${SMB_EXECUTIVE_WATCHDOG_MAX_MUMPS_REALLOC:-4}"
export SMB_MAX_SOLVE_SECONDS="${SMB_MAX_SOLVE_SECONDS:-1800}"
export SMB_AGENT_TEE="${SMB_AGENT_TEE:-1}"
export SMB_OBJECTIVES_FILE="${SMB_OBJECTIVES_FILE:-${OBJECTIVES_FILE}}"
export SMB_LLM_SOUL_FILE="${SMB_LLM_SOUL_FILE:-${LLM_SOUL_FILE}}"
export SMB_IPOPT_RESOURCE_FILE="${SMB_IPOPT_RESOURCE_FILE:-${IPOPT_RESOURCE_FILE}}"
export SMB_RESEARCH_MD="${SMB_RESEARCH_MD:-${REPO_ROOT}/research.md}"
export SMB_REFERENCE_NOTEBOOK="${SMB_REFERENCE_NOTEBOOK:-${NOTEBOOK_REFERENCE}}"
export SMB_REFERENCE_MODEL_ROOT="${SMB_REFERENCE_MODEL_ROOT:-${SMB_ROOT}}"
export SMB_SKIP_INITIAL_PLAN_LLM="${SMB_SKIP_INITIAL_PLAN_LLM:-1}"
export SMB_RESEARCH_TAIL_CHARS="${SMB_RESEARCH_TAIL_CHARS:-1200}"
export SMB_OBJECTIVES_MAX_CHARS="${SMB_OBJECTIVES_MAX_CHARS:-3000}"
export SMB_LLM_SOUL_MAX_CHARS="${SMB_LLM_SOUL_MAX_CHARS:-1800}"
export SMB_PROBLEM_DEFINITION_MAX_CHARS="${SMB_PROBLEM_DEFINITION_MAX_CHARS:-1200}"
export SMB_SKILLS_MAX_CHARS="${SMB_SKILLS_MAX_CHARS:-1200}"
export SMB_IPOPT_RESOURCE_MAX_CHARS="${SMB_IPOPT_RESOURCE_MAX_CHARS:-900}"
export SMB_LLM_TIMEOUT_SECONDS="${SMB_LLM_TIMEOUT_SECONDS:-120}"
export SMB_LLM_MAX_TOKENS="${SMB_LLM_MAX_TOKENS:-320}"
export SMB_LLM_MAX_RETRIES="${SMB_LLM_MAX_RETRIES:-1}"
export SMB_LLM_RETRY_BACKOFF_SECONDS="${SMB_LLM_RETRY_BACKOFF_SECONDS:-0}"
export SMB_FORCE_MUMPS_ONLY="${SMB_FORCE_MUMPS_ONLY:-1}"
export SMB_SOLVER_NAME="${SMB_SOLVER_NAME:-auto}"
export SMB_LINEAR_SOLVER="${SMB_LINEAR_SOLVER:-mumps}"
if [[ "${SMB_FORCE_MUMPS_ONLY}" == "1" && "${SMB_LINEAR_SOLVER}" != "mumps" ]]; then
  echo "WARN: SMB_FORCE_MUMPS_ONLY=1 overrides SMB_LINEAR_SOLVER='${SMB_LINEAR_SOLVER}' -> 'mumps'" >&2
  SMB_LINEAR_SOLVER="mumps"
fi
export SMB_CURRENT_DEFAULT_SOLVER_STACK="${SMB_CURRENT_DEFAULT_SOLVER_STACK:-pyomo+ipopt_sens+mumps}"
export SMB_AVAILABLE_SOLVER_OPTIONS="${SMB_AVAILABLE_SOLVER_OPTIONS:-pyomo+ipopt_sens+mumps(forced_default_cpu)}"
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
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-5m}"
OLLAMA_GPU_ID="${OLLAMA_GPU_ID:-}"
SMB_OLLAMA_PULL_IF_MISSING="${SMB_OLLAMA_PULL_IF_MISSING:-0}"
SMB_OLLAMA_PREWARM_ENABLED="${SMB_OLLAMA_PREWARM_ENABLED:-1}"
SMB_OLLAMA_PREWARM_MAX_SECONDS="${SMB_OLLAMA_PREWARM_MAX_SECONDS:-45}"
export SMB_FALLBACK_LLM_ENABLED="${SMB_FALLBACK_LLM_ENABLED:-1}"
export SMB_FALLBACK_LLM_BASE_URL="${SMB_FALLBACK_LLM_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
export SMB_FALLBACK_LLM_MODEL="${SMB_FALLBACK_LLM_MODEL:-gpt-5-nano}"
export SMB_FALLBACK_LLM_API_KEY="${SMB_FALLBACK_LLM_API_KEY:-${OPENAI_API_KEY:-}}"
if [[ "${SMB_TSTEP_BOUNDS}" != *,* ]]; then
  echo "ERROR: SMB_TSTEP_BOUNDS='${SMB_TSTEP_BOUNDS}' is invalid. Expected 'min,max' (for example: 8.0,12.0)." >&2
  echo "Hint: when using sbatch --export, set SMB_TSTEP_BOUNDS in the shell first: export SMB_TSTEP_BOUNDS=\"8.0,12.0\"" >&2
  exit 2
fi
if [[ -n "${SMB_FALLBACK_LLM_API_KEY}" ]]; then
  SMB_FALLBACK_LLM_KEY_PRESENT="yes"
else
  SMB_FALLBACK_LLM_KEY_PRESENT="no"
fi
LOCAL_LLM_PID=""
SMB_IPOPT_PRECHECK_SUMMARY=""
SMB_VERIFIED_IPOPT_EXECUTABLES=""
SMB_VERIFIED_IPOPT_LINEAR_SOLVERS=""
SMB_VERIFIED_IPOPT_PROFILE_MENU=""
SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE=""
SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE=""
SMB_IPOPT_PRECHECK_TIMEOUT_SECONDS="${SMB_IPOPT_PRECHECK_TIMEOUT_SECONDS:-20}"

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
  local timeout_cmd=""
  if command -v timeout >/dev/null 2>&1; then
    timeout_cmd="timeout ${SMB_IPOPT_PRECHECK_TIMEOUT_SECONDS}"
  fi

  if command -v ipopt_sens >/dev/null 2>&1; then
    append_resource_item SMB_VERIFIED_IPOPT_EXECUTABLES "ipopt_sens"
    if [[ -n "${timeout_cmd}" ]]; then
      probe_output="$(${timeout_cmd} ipopt_sens --print-options 2>/dev/null || true)"
    else
      probe_output="$(ipopt_sens --print-options 2>/dev/null || true)"
    fi
    if [[ -n "${probe_output}" ]]; then
      combined_options="${combined_options}"$'\n'"${probe_output}"
    fi
  fi

  if command -v ipopt >/dev/null 2>&1; then
    append_resource_item SMB_VERIFIED_IPOPT_EXECUTABLES "ipopt"
    if [[ -n "${timeout_cmd}" ]]; then
      probe_output="$(${timeout_cmd} ipopt --print-options 2>/dev/null || true)"
    else
      probe_output="$(ipopt --print-options 2>/dev/null || true)"
    fi
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
  local ollama_exec ready ready_url max_attempts prompt_payload response_file http_code prewarm_url
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
    # Preserve Slurm-assigned CUDA_VISIBLE_DEVICES by default; only fall back to OLLAMA_GPU_ID when needed.
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      visible_devices="${CUDA_VISIBLE_DEVICES}"
    else
      visible_devices="${OLLAMA_GPU_ID}"
    fi
    echo "[$(date '+%F %T')] START: local Ollama server for Qwen on GPU map '${visible_devices:-<auto>}'"
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
    if [[ "${SMB_OLLAMA_PULL_IF_MISSING}" == "1" ]]; then
      echo "[$(date '+%F %T')] PULL: ${OLLAMA_MODEL}"
      OLLAMA_HOST="${OLLAMA_HOST}" "${ollama_exec}" pull "${OLLAMA_MODEL}"
    else
      echo "WARN: local model '${OLLAMA_MODEL}' not found and SMB_OLLAMA_PULL_IF_MISSING=0"
      cleanup_local_llm
      return 1
    fi
  fi

  if [[ "${SMB_OLLAMA_PREWARM_ENABLED}" == "1" ]]; then
    prewarm_url="${OLLAMA_BASE_URL%/v1}/api/generate"
    prompt_payload=$(printf '{"model":"%s","prompt":"Reply with exactly: pong","stream":false,"options":{"num_predict":8,"temperature":0}}' "${OLLAMA_MODEL}")
    response_file="$(mktemp)"
    http_code="$(curl -sS \
      --connect-timeout 2 \
      --max-time "${SMB_OLLAMA_PREWARM_MAX_SECONDS}" \
      -o "${response_file}" \
      -w '%{http_code}' \
      -H "Content-Type: application/json" \
      -d "${prompt_payload}" \
      "${prewarm_url}" || true)"
    rm -f "${response_file}"

    if [[ ! "${http_code}" =~ ^2 ]]; then
      echo "WARN: local Qwen prewarm failed with status=${http_code:-<curl-error>} (continuing without prewarm gate)"
    fi
  fi

  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL}"
  export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
  export OLLAMA_MODEL="${OLLAMA_MODEL}"
  echo "[$(date '+%F %T')] READY: local Qwen available at ${OLLAMA_BASE_URL}"
  return 0
}

echo "[$(date '+%F %T')] START: IPOPT precheck (timeout=${SMB_IPOPT_PRECHECK_TIMEOUT_SECONDS}s)"
probe_ipopt_resources
echo "[$(date '+%F %T')] DONE: IPOPT precheck"

if [[ -z "${SMB_VERIFIED_IPOPT_EXECUTABLES}" ]]; then
  echo "ERROR: IPOPT precheck found no usable executable on PATH." >&2
  echo "Hint: set IPOPT_BIN_DIR/IPOPT_LIB_DIR (or load your IPOPT module) before submission." >&2
  echo "Current PATH: ${PATH}" >&2
  echo "Expected examples:" >&2
  echo "  export IPOPT_BIN_DIR=\"/storage/scratch1/<group>/<user>/opt/ipopt/bin\"" >&2
  echo "  export IPOPT_LIB_DIR=\"/storage/scratch1/<group>/<user>/opt/ipopt/lib\"" >&2
  exit 2
fi

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
- research_md: ${SMB_RESEARCH_MD}
- fallback_llm_enabled: ${SMB_FALLBACK_LLM_ENABLED}
- fallback_llm_base_url: ${SMB_FALLBACK_LLM_BASE_URL}
- fallback_llm_model: ${SMB_FALLBACK_LLM_MODEL}
- fallback_llm_api_key_present: ${SMB_FALLBACK_LLM_KEY_PRESENT}
- reference_model_root: ${SMB_REFERENCE_MODEL_ROOT}
- reference_notebook: ${SMB_REFERENCE_NOTEBOOK}
- force_mumps_only: ${SMB_FORCE_MUMPS_ONLY}
- solver_name: ${SMB_SOLVER_NAME}
- linear_solver: ${SMB_LINEAR_SOLVER}
- ffeed_bounds_ml_min: ${SMB_FFEED_BOUNDS_ML_MIN}
- max_pump_flow_ml_min: ${SMB_MAX_PUMP_FLOW_ML_MIN}
- tstep_bounds: ${SMB_TSTEP_BOUNDS}
- exploratory_target_purity_ex_meoh_free: ${SMB_TARGET_PURITY_EX_MEOH_FREE}
- exploratory_target_recovery_ga: ${SMB_TARGET_RECOVERY_GA}
- exploratory_target_recovery_ma: ${SMB_TARGET_RECOVERY_MA}
- project_objective_purity_ex_meoh_free: ${SMB_PROJECT_TARGET_PURITY_EX_MEOH_FREE}
- project_objective_recovery_ga: ${SMB_PROJECT_TARGET_RECOVERY_GA}
- project_objective_recovery_ma: ${SMB_PROJECT_TARGET_RECOVERY_MA}
- benchmark_hours: ${SMB_BENCHMARK_HOURS}
- search_budget_hours: ${SMB_SEARCH_BUDGET_HOURS}
- validation_budget_hours: ${SMB_VALIDATION_BUDGET_HOURS}
- min_probe_reference_runs: ${SMB_MIN_PROBE_REFERENCE_RUNS}
- single_scientist_mode: ${SMB_SINGLE_SCIENTIST_MODE}
- probe_low_fidelity_enabled: ${SMB_PROBE_LOW_FIDELITY_ENABLED}
- probe_fidelity_nfex: ${SMB_PROBE_NFEX}
- probe_fidelity_nfet: ${SMB_PROBE_NFET}
- probe_fidelity_ncp: ${SMB_PROBE_NCP}
- finalization_hard_gate_enabled: ${SMB_FINALIZATION_HARD_GATE_ENABLED}
- finalization_low_fidelity_nfex: ${SMB_FINALIZATION_LOW_FIDELITY_NFEX}
- finalization_low_fidelity_nfet: ${SMB_FINALIZATION_LOW_FIDELITY_NFET}
- finalization_low_fidelity_ncp: ${SMB_FINALIZATION_LOW_FIDELITY_NCP}
- ipopt_max_iter: ${SMB_IPOPT_MAX_ITER}
- ipopt_tol: ${SMB_IPOPT_TOL}
- ipopt_acceptable_tol: ${SMB_IPOPT_ACCEPTABLE_TOL}
- executive_controller_enabled: ${SMB_EXECUTIVE_CONTROLLER_ENABLED}
- executive_trigger_rejects: ${SMB_EXECUTIVE_TRIGGER_REJECTS}
- executive_force_after_rejects: ${SMB_EXECUTIVE_FORCE_AFTER_REJECTS}
- executive_top_k_lock: ${SMB_EXECUTIVE_TOP_K_LOCK}
- executive_live_monitor: ${SMB_EXECUTIVE_LIVE_MONITOR}
- executive_monitor_poll_seconds: ${SMB_EXECUTIVE_MONITOR_POLL_SECONDS}
- executive_monitor_window_iters: ${SMB_EXECUTIVE_MONITOR_WINDOW_ITERS}
- executive_monitor_stall_eps: ${SMB_EXECUTIVE_MONITOR_STALL_EPS}
- executive_watchdog_enabled: ${SMB_EXECUTIVE_WATCHDOG_ENABLED}
- executive_watchdog_min_iters: ${SMB_EXECUTIVE_WATCHDOG_MIN_ITERS}
- executive_watchdog_stall_windows: ${SMB_EXECUTIVE_WATCHDOG_STALL_WINDOWS}
- executive_watchdog_max_inf_du: ${SMB_EXECUTIVE_WATCHDOG_MAX_INF_DU}
- executive_watchdog_max_mumps_realloc: ${SMB_EXECUTIVE_WATCHDOG_MAX_MUMPS_REALLOC}
- max_solve_seconds: ${SMB_MAX_SOLVE_SECONDS}
- agent_tee: ${SMB_AGENT_TEE}
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
echo "Research log file: ${SMB_RESEARCH_MD}"
echo "Fallback LLM enabled: ${SMB_FALLBACK_LLM_ENABLED}"
echo "Fallback LLM endpoint/model: ${SMB_FALLBACK_LLM_BASE_URL} | ${SMB_FALLBACK_LLM_MODEL}"
echo "Fallback LLM API key present: ${SMB_FALLBACK_LLM_KEY_PRESENT}"
echo "Reference notebook: ${NOTEBOOK_REFERENCE}"
echo "Python executable: ${VENV_PY}"
echo "Compute summary: ${SMB_COMPUTE_SUMMARY}"
echo "Force MUMPS only: ${SMB_FORCE_MUMPS_ONLY}"
echo "Solver selection: solver_name=${SMB_SOLVER_NAME}, linear_solver=${SMB_LINEAR_SOLVER}"
echo "Agent budget: benchmark=${SMB_BENCHMARK_HOURS}h search=${SMB_SEARCH_BUDGET_HOURS}h validation=${SMB_VALIDATION_BUDGET_HOURS}h min_probe_reference_runs=${SMB_MIN_PROBE_REFERENCE_RUNS}"
echo "Single-scientist mode: ${SMB_SINGLE_SCIENTIST_MODE}"
echo "Probe low-fidelity policy: enabled=${SMB_PROBE_LOW_FIDELITY_ENABLED} nfex=${SMB_PROBE_NFEX} nfet=${SMB_PROBE_NFET} ncp=${SMB_PROBE_NCP}"
echo "Finalization hard gate: enabled=${SMB_FINALIZATION_HARD_GATE_ENABLED} low_fidelity_nfex<=${SMB_FINALIZATION_LOW_FIDELITY_NFEX} low_fidelity_nfet<=${SMB_FINALIZATION_LOW_FIDELITY_NFET} low_fidelity_ncp<=${SMB_FINALIZATION_LOW_FIDELITY_NCP}"
echo "IPOPT defaults: max_iter=${SMB_IPOPT_MAX_ITER} tol=${SMB_IPOPT_TOL} acceptable_tol=${SMB_IPOPT_ACCEPTABLE_TOL}"
echo "Executive live monitor: enabled=${SMB_EXECUTIVE_LIVE_MONITOR} poll_s=${SMB_EXECUTIVE_MONITOR_POLL_SECONDS} window_iters=${SMB_EXECUTIVE_MONITOR_WINDOW_ITERS} stall_eps=${SMB_EXECUTIVE_MONITOR_STALL_EPS}"
echo "Executive watchdog: enabled=${SMB_EXECUTIVE_WATCHDOG_ENABLED} min_iters=${SMB_EXECUTIVE_WATCHDOG_MIN_ITERS} stall_windows=${SMB_EXECUTIVE_WATCHDOG_STALL_WINDOWS} max_inf_du=${SMB_EXECUTIVE_WATCHDOG_MAX_INF_DU} max_mumps_realloc=${SMB_EXECUTIVE_WATCHDOG_MAX_MUMPS_REALLOC}"
echo "Per-solve walltime cap (seconds): ${SMB_MAX_SOLVE_SECONDS}"
echo "Solver tee enabled: ${SMB_AGENT_TEE}"
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
  if [[ "${AGENT_ENTRYPOINT}" == python\ * ]]; then
    AGENT_ENTRYPOINT="${VENV_PY} ${AGENT_ENTRYPOINT#python }"
  fi
  hash -r
  echo "Agent launch python: ${VENV_PY}"
  echo "Agent launch which python: $(command -v python || true)"
  echo "[$(date '+%F %T')] START: AGENT_ENTRYPOINT=${AGENT_ENTRYPOINT}"
  bash -c "${AGENT_ENTRYPOINT}"
  echo "[$(date '+%F %T')] DONE: agent entrypoint"
else
  echo "No AGENT_ENTRYPOINT provided."
  echo "This job exported the two-scientist SMB compute budget and optional local Qwen environment."
  echo "Set AGENT_ENTRYPOINT when the autonomous runner is ready."
fi

echo "[$(date '+%F %T')] SMB two-scientist job setup complete."
