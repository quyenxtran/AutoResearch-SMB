#!/bin/bash
# Generic SMB stage runner for PACE.
#
# Submit one stage at a time:
# sbatch --export=ALL,SMB_STAGE=solver-check AutoResearch-SMB/pace_smb_stage_runner.slurm
# sbatch --export=ALL,SMB_STAGE=reference-eval,SMB_RUN_NAME=reference_hi_fi AutoResearch-SMB/pace_smb_stage_runner.slurm
# sbatch --export=ALL,SMB_STAGE=reference-eval,SMB_NFEX=6,SMB_NFET=3,SMB_NCP=2,SMB_RUN_NAME=reference_med_fi AutoResearch-SMB/pace_smb_stage_runner.slurm
# sbatch --export=ALL,SMB_STAGE=nc-screen,SMB_NC_LIBRARY="1,2,3,2;2,2,2,2;1,3,2,2",SMB_RUN_NAME=nc_smoke AutoResearch-SMB/pace_smb_stage_runner.slurm
#
# Override account, partition, or time at submit time if needed:
# sbatch --account=gts-sn73 --qos=inferno --partition=cpu-amd --time=08:00:00 --export=ALL,SMB_STAGE=reference-eval AutoResearch-SMB/pace_smb_stage_runner.slurm

#SBATCH --job-name=smb-stage
#SBATCH -A gts-sn73
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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
  if [[ -f "${REPO_ROOT}/benchmarks/run_stage.py" && -d "${REPO_ROOT}/SembaSMB" ]]; then
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
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/artifacts/smb_stage_runs}"
mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"

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

export PYTHONUNBUFFERED=1
export PATH="${VENV_DIR}/bin:${PATH}"

SMB_STAGE="${SMB_STAGE:-solver-check}"
SMB_RUN_NAME="${SMB_RUN_NAME:-${SMB_STAGE}}"
SMB_SOLVER_NAME="${SMB_SOLVER_NAME:-auto}"
SMB_LINEAR_SOLVER="${SMB_LINEAR_SOLVER:-}"
SMB_NC="${SMB_NC:-1,2,3,2}"
SMB_NC_LIBRARY="${SMB_NC_LIBRARY:-1,2,3,2;2,2,2,2}"
SMB_F1="${SMB_F1:-2.2}"
SMB_FDES="${SMB_FDES:-1.2}"
SMB_FEX="${SMB_FEX:-0.9}"
SMB_FFEED="${SMB_FFEED:-1.3}"
SMB_FRAF="${SMB_FRAF:-}"
SMB_TSTEP="${SMB_TSTEP:-9.4}"
SMB_NFEX="${SMB_NFEX:-10}"
SMB_NFET="${SMB_NFET:-5}"
SMB_NCP="${SMB_NCP:-2}"
SMB_PURITY_MIN="${SMB_PURITY_MIN:-0.90}"
SMB_RECOVERY_GA_MIN="${SMB_RECOVERY_GA_MIN:-0.90}"
SMB_RECOVERY_MA_MIN="${SMB_RECOVERY_MA_MIN:-0.90}"
SMB_MAX_PUMP_FLOW="${SMB_MAX_PUMP_FLOW:-2.5}"
SMB_STAGE_TEE="${SMB_STAGE_TEE:-0}"

STAGE_ARGS=(
  --stage "${SMB_STAGE}"
  --run-name "${SMB_RUN_NAME}"
  --artifact-dir "${ARTIFACT_DIR}"
  --solver-name "${SMB_SOLVER_NAME}"
  --nc "${SMB_NC}"
  --nc-library "${SMB_NC_LIBRARY}"
  --f1 "${SMB_F1}"
  --fdes "${SMB_FDES}"
  --fex "${SMB_FEX}"
  --ffeed "${SMB_FFEED}"
  --tstep "${SMB_TSTEP}"
  --nfex "${SMB_NFEX}"
  --nfet "${SMB_NFET}"
  --ncp "${SMB_NCP}"
  --purity-min "${SMB_PURITY_MIN}"
  --recovery-ga-min "${SMB_RECOVERY_GA_MIN}"
  --recovery-ma-min "${SMB_RECOVERY_MA_MIN}"
  --max-pump-flow "${SMB_MAX_PUMP_FLOW}"
)

if [[ -n "${SMB_LINEAR_SOLVER}" ]]; then
  STAGE_ARGS+=(--linear-solver "${SMB_LINEAR_SOLVER}")
fi
if [[ -n "${SMB_FRAF}" ]]; then
  STAGE_ARGS+=(--fraf "${SMB_FRAF}")
fi
if [[ "${SMB_STAGE_TEE}" == "1" ]]; then
  STAGE_ARGS+=(--tee)
fi

echo "Repository root: ${REPO_ROOT}"
echo "AutoResearch-SMB root: ${AUTORESEARCH_ROOT}"
echo "SembaSMB root: ${SMB_ROOT}"
echo "Python executable: ${VENV_PY}"
echo "Stage: ${SMB_STAGE}"
echo "Run name: ${SMB_RUN_NAME}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-12}"
echo "Artifact dir: ${ARTIFACT_DIR}"
echo "nc: ${SMB_NC}"
echo "nc library: ${SMB_NC_LIBRARY}"
echo "flows: Ffeed=${SMB_FFEED}, F1=${SMB_F1}, Fdes=${SMB_FDES}, Fex=${SMB_FEX}, tstep=${SMB_TSTEP}"
echo "fidelity: nfex=${SMB_NFEX}, nfet=${SMB_NFET}, ncp=${SMB_NCP}"

cd "${REPO_ROOT}"
"${VENV_PY}" "${AUTORESEARCH_ROOT}/benchmarks/run_stage.py" "${STAGE_ARGS[@]}"

echo "[$(date '+%F %T')] SMB stage runner complete."
