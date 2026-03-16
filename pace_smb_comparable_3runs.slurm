#!/bin/bash
# Comparable SMB suite: MINLP + single-scientist + two-scientists.
#
# Submit once (spawns 3 array tasks, one per mode):
#   sbatch AutoResearch-SMB/pace_smb_comparable_3runs.slurm
#
# Optional explicit suite tag shared by all 3 tasks:
#   sbatch --export=ALL,SMB_SUITE_TAG=mycmp01 AutoResearch-SMB/pace_smb_comparable_3runs.slurm
#
# Modes by array index:
#   0 = MINLP (run_stage optimize-layouts, all 35 NC, reference seed)
#   1 = Single-scientist agent
#   2 = Two-scientists agent

#SBATCH --job-name=smb-compare3
#SBATCH -A gts-sn73
#SBATCH --qos=inferno
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --mail-user=qtran47@gatech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Force base script to run from AutoResearch-SMB root.
export REPO_ROOT="${SCRIPT_DIR}"

# Shared comparison tag across all array children.
SMB_SUITE_TAG="${SMB_SUITE_TAG:-cmp_${SLURM_ARRAY_JOB_ID:-local}}"

# Shared thresholds and IPOPT settings for apples-to-apples comparison.
export SMB_TARGET_PURITY_EX_MEOH_FREE="${SMB_TARGET_PURITY_EX_MEOH_FREE:-0.80}"
export SMB_TARGET_RECOVERY_GA="${SMB_TARGET_RECOVERY_GA:-0.75}"
export SMB_TARGET_RECOVERY_MA="${SMB_TARGET_RECOVERY_MA:-0.80}"
export SMB_PROJECT_TARGET_PURITY_EX_MEOH_FREE="${SMB_PROJECT_TARGET_PURITY_EX_MEOH_FREE:-0.80}"
export SMB_PROJECT_TARGET_RECOVERY_GA="${SMB_PROJECT_TARGET_RECOVERY_GA:-0.75}"
export SMB_PROJECT_TARGET_RECOVERY_MA="${SMB_PROJECT_TARGET_RECOVERY_MA:-0.80}"
export SMB_IPOPT_MAX_ITER="${SMB_IPOPT_MAX_ITER:-1000}"
export SMB_IPOPT_TOL="${SMB_IPOPT_TOL:-1e-5}"
export SMB_IPOPT_ACCEPTABLE_TOL="${SMB_IPOPT_ACCEPTABLE_TOL:-1e-4}"
export SMB_MAX_SOLVE_SECONDS="${SMB_MAX_SOLVE_SECONDS:-1800}"
export SMB_MIN_PROBE_REFERENCE_RUNS="${SMB_MIN_PROBE_REFERENCE_RUNS:-35}"
export SMB_PROBE_LOW_FIDELITY_ENABLED="${SMB_PROBE_LOW_FIDELITY_ENABLED:-1}"
export SMB_PROBE_NFEX="${SMB_PROBE_NFEX:-5}"
export SMB_PROBE_NFET="${SMB_PROBE_NFET:-2}"
export SMB_PROBE_NCP="${SMB_PROBE_NCP:-1}"
export SMB_FORCE_MUMPS_ONLY="${SMB_FORCE_MUMPS_ONLY:-1}"
export SMB_LINEAR_SOLVER="${SMB_LINEAR_SOLVER:-mumps}"
export SMB_SOLVER_NAME="${SMB_SOLVER_NAME:-auto}"

# Keep each run isolated.
ART_DIR="${SCRIPT_DIR}/artifacts"
RUNS_DIR="${ART_DIR}/agent_runs"
mkdir -p "${RUNS_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODE="minlp"
    RUN_NAME="minlp_all35_${SMB_SUITE_TAG}"
    export START_LOCAL_LLM=0
    export SMB_SINGLE_SCIENTIST_MODE=1
    export AGENT_ENTRYPOINT="python -m benchmarks.run_stage \
      --stage optimize-layouts \
      --run-name ${RUN_NAME} \
      --artifact-dir ${ART_DIR}/smb_stage_runs \
      --solver-name ipopt \
      --linear-solver mumps \
      --nc-library all \
      --seed-library 2.2,1.2,0.9,1.3,9.4 \
      --nfex ${SMB_PROBE_NFEX} --nfet ${SMB_PROBE_NFET} --ncp ${SMB_PROBE_NCP} \
      --purity-min ${SMB_TARGET_PURITY_EX_MEOH_FREE} \
      --recovery-ga-min ${SMB_TARGET_RECOVERY_GA} \
      --recovery-ma-min ${SMB_TARGET_RECOVERY_MA} \
      --max-iter ${SMB_IPOPT_MAX_ITER} \
      --tol ${SMB_IPOPT_TOL} \
      --acceptable-tol ${SMB_IPOPT_ACCEPTABLE_TOL} \
      --max-solve-seconds ${SMB_MAX_SOLVE_SECONDS} \
      --tstep-bounds 8.0,12.0 \
      --ffeed-bounds 0.5,2.5 \
      --fdes-bounds 0.5,2.5 \
      --fex-bounds 0.5,2.5 \
      --fraf-bounds 0.5,2.5 \
      --f1-bounds 0.5,5.0 \
      --tee"
    ;;
  1)
    MODE="single"
    RUN_NAME="single_scientist_all35_${SMB_SUITE_TAG}"
    SQLITE_DB="${RUNS_DIR}/smb_agent_context_${RUN_NAME}.sqlite"
    RESEARCH_MD="${RUNS_DIR}/research_${RUN_NAME}.md"
    export START_LOCAL_LLM=1
    export SMB_SINGLE_SCIENTIST_MODE=1
    export AGENT_ENTRYPOINT="python -m benchmarks.agent_runner \
      --tee \
      --single-scientist-mode 1 \
      --run-name ${RUN_NAME} \
      --research-md ${RESEARCH_MD} \
      --sqlite-db ${SQLITE_DB} \
      --reset-research-section"
    ;;
  2)
    MODE="two"
    RUN_NAME="two_scientists_all35_${SMB_SUITE_TAG}"
    SQLITE_DB="${RUNS_DIR}/smb_agent_context_${RUN_NAME}.sqlite"
    RESEARCH_MD="${RUNS_DIR}/research_${RUN_NAME}.md"
    export START_LOCAL_LLM=1
    export SMB_SINGLE_SCIENTIST_MODE=0
    export AGENT_ENTRYPOINT="python -m benchmarks.agent_runner \
      --tee \
      --single-scientist-mode 0 \
      --run-name ${RUN_NAME} \
      --research-md ${RESEARCH_MD} \
      --sqlite-db ${SQLITE_DB} \
      --reset-research-section"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID='${SLURM_ARRAY_TASK_ID}' (expected 0,1,2)" >&2
    exit 2
    ;;
esac

echo "Comparable suite task start"
echo "  mode: ${MODE}"
echo "  run_name: ${RUN_NAME}"
echo "  suite_tag: ${SMB_SUITE_TAG}"
echo "  thresholds: purity>=${SMB_TARGET_PURITY_EX_MEOH_FREE}, rGA>=${SMB_TARGET_RECOVERY_GA}, rMA>=${SMB_TARGET_RECOVERY_MA}"
echo "  ipopt: max_iter=${SMB_IPOPT_MAX_ITER}, tol=${SMB_IPOPT_TOL}, acceptable_tol=${SMB_IPOPT_ACCEPTABLE_TOL}, max_solve_s=${SMB_MAX_SOLVE_SECONDS}"
echo "  entrypoint: ${AGENT_ENTRYPOINT}"

# Reuse the hardened base launcher for env detection, IPOPT probing, and optional local LLM.
bash "${SCRIPT_DIR}/pace_smb_two_scientists_qwen.slurm"

echo "Comparable suite task done: mode=${MODE}, run_name=${RUN_NAME}"
