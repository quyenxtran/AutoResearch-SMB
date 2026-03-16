#!/bin/bash
# Single-scientist SMB run (24h, GPU for local LLM), comparable thresholds.
#
# Submit:
#   sbatch AutoResearch-SMB/pace_smb_single_scientist_24h.slurm
#
# Optional shared tag:
#   sbatch --export=ALL,SMB_SUITE_TAG=cmp01 AutoResearch-SMB/pace_smb_single_scientist_24h.slurm

#SBATCH --job-name=smb-single-24h
#SBATCH -A gts-sn73
#SBATCH --qos=inferno
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-user=qtran47@gatech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${SCRIPT_DIR}"

SUITE_TAG="${SMB_SUITE_TAG:-cmp_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${SMB_RUN_NAME:-single_scientist_all35_${SUITE_TAG}}"

ART_DIR="${SCRIPT_DIR}/artifacts/agent_runs"
mkdir -p "${ART_DIR}"

# Matched thresholds / solver settings.
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
export SMB_WALLTIME_HOURS="${SMB_WALLTIME_HOURS:-24}"

export START_LOCAL_LLM="${START_LOCAL_LLM:-1}"
export SMB_AGENT_TEE="${SMB_AGENT_TEE:-1}"
export SMB_SINGLE_SCIENTIST_MODE=1
export AGENT_ENTRYPOINT="python -m benchmarks.agent_runner --tee --single-scientist-mode 1 --run-name ${RUN_NAME} --research-md ${ART_DIR}/research_${RUN_NAME}.md --sqlite-db ${ART_DIR}/smb_agent_context_${RUN_NAME}.sqlite --reset-research-section"

echo "Starting single-scientist run"
echo "  run_name=${RUN_NAME}"
echo "  suite_tag=${SUITE_TAG}"
echo "  thresholds: purity>=${SMB_TARGET_PURITY_EX_MEOH_FREE}, rGA>=${SMB_TARGET_RECOVERY_GA}, rMA>=${SMB_TARGET_RECOVERY_MA}"

bash "${SCRIPT_DIR}/pace_smb_two_scientists_qwen.slurm"

