#!/bin/bash
# CPU-only MINLP run (24h), intended for comparison against agent-driven runs.
#
# Submit:
#   sbatch AutoResearch-SMB/pace_smb_minlp_cpu_24h.slurm
#
# Optional shared tag:
#   sbatch --export=ALL,SMB_SUITE_TAG=cmp01 AutoResearch-SMB/pace_smb_minlp_cpu_24h.slurm

#SBATCH --job-name=smb-minlp-24h
#SBATCH -A gts-sn73
#SBATCH --qos=inferno
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

# Prefer submission directory on Slurm because BASH_SOURCE can resolve to
# /var/lib/slurm/slurmd/job*/ for copied scripts that do not include siblings.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pace_smb_stage_runner.slurm" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
fi
export REPO_ROOT="${SCRIPT_DIR}"

SUITE_TAG="${SMB_SUITE_TAG:-cmp_$(date +%Y%m%d_%H%M%S)}"
export SMB_SUITE_TAG="${SUITE_TAG}"

# Matched thresholds / solver settings for comparability.
export SMB_PURITY_MIN="${SMB_PURITY_MIN:-0.80}"
export SMB_RECOVERY_GA_MIN="${SMB_RECOVERY_GA_MIN:-0.75}"
export SMB_RECOVERY_MA_MIN="${SMB_RECOVERY_MA_MIN:-0.80}"
export SMB_IPOPT_MAX_ITER="${SMB_IPOPT_MAX_ITER:-1000}"
export SMB_IPOPT_TOL="${SMB_IPOPT_TOL:-1e-5}"
export SMB_IPOPT_ACCEPTABLE_TOL="${SMB_IPOPT_ACCEPTABLE_TOL:-1e-4}"
export SMB_MAX_SOLVE_SECONDS="${SMB_MAX_SOLVE_SECONDS:-1800}"
export SMB_LINEAR_SOLVER="${SMB_LINEAR_SOLVER:-mumps}"
export SMB_SOLVER_NAME="${SMB_SOLVER_NAME:-ipopt}"
export SMB_STAGE="${SMB_STAGE:-optimize-layouts}"
export SMB_STAGE_TEE="${SMB_STAGE_TEE:-1}"

# MINLP scope.
export SMB_RUN_NAME="${SMB_RUN_NAME:-minlp_all35_${SUITE_TAG}}"
export SMB_NC_LIBRARY="${SMB_NC_LIBRARY:-all}"
export SMB_SEED_LIBRARY="${SMB_SEED_LIBRARY:-notebook}"

# Keep fidelity configurable; default to lower fidelity for broad search.
export SMB_NFEX="${SMB_NFEX:-5}"
export SMB_NFET="${SMB_NFET:-2}"
export SMB_NCP="${SMB_NCP:-1}"

echo "Starting MINLP CPU run"
echo "  run_name=${SMB_RUN_NAME}"
echo "  suite_tag=${SUITE_TAG}"
echo "  thresholds: purity>=${SMB_PURITY_MIN}, rGA>=${SMB_RECOVERY_GA_MIN}, rMA>=${SMB_RECOVERY_MA_MIN}"
echo "  ipopt: max_iter=${SMB_IPOPT_MAX_ITER}, tol=${SMB_IPOPT_TOL}, acceptable_tol=${SMB_IPOPT_ACCEPTABLE_TOL}, max_solve_s=${SMB_MAX_SOLVE_SECONDS}"

bash "${SCRIPT_DIR}/pace_smb_stage_runner.slurm"

