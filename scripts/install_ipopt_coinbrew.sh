#!/usr/bin/env bash
set -euo pipefail

# Build and install a scratch-local Ipopt using COIN-OR coinbrew.
# Official references:
# - https://coin-or.github.io/Ipopt/INSTALL.html
# - https://coin-or.github.io/coinbrew/
#
# Default install locations:
#   source root:  $SCRATCH/src/coin-or
#   install root: $SCRATCH/opt/ipopt
#
# You can override:
#   IPOPT_PREFIX=/path/to/install
#   COIN_OR_ROOT=/path/to/sources
#   COINBREW_TEST=0
#   IPOPT_LAPACK_LFLAGS="-L/path/to/lib -lopenblas"
#   IPOPT_BLAS_LFLAGS="-L/path/to/lib -lopenblas"
#   COINBREW_EXTRA_ARGS="--parallel-jobs 8"

if [[ -z "${SCRATCH:-}" ]]; then
  echo "ERROR: SCRATCH is not set."
  echo "On PACE, set SCRATCH or export IPOPT_PREFIX and COIN_OR_ROOT explicitly."
  exit 1
fi

IPOPT_PREFIX="${IPOPT_PREFIX:-$SCRATCH/opt/ipopt}"
COIN_OR_ROOT="${COIN_OR_ROOT:-$SCRATCH/src/coin-or}"
COINBREW="${COIN_OR_ROOT}/coinbrew"
COINBREW_TEST="${COINBREW_TEST:-1}"
IPOPT_LAPACK_LFLAGS="${IPOPT_LAPACK_LFLAGS:-}"
IPOPT_BLAS_LFLAGS="${IPOPT_BLAS_LFLAGS:-}"
COINBREW_EXTRA_ARGS="${COINBREW_EXTRA_ARGS:-}"

required_cmds=(bash git make pkg-config gcc g++ gfortran)
download_cmd=""
if command -v wget >/dev/null 2>&1; then
  download_cmd="wget -O"
elif command -v curl >/dev/null 2>&1; then
  download_cmd="curl -L -o"
else
  echo "ERROR: neither wget nor curl is available."
  exit 1
fi

missing_cmds=()
for cmd in "${required_cmds[@]}"; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    missing_cmds+=("${cmd}")
  fi
done

if (( ${#missing_cmds[@]} > 0 )); then
  echo "ERROR: missing required build tools: ${missing_cmds[*]}"
  echo "Load the necessary compiler/build modules first, then rerun."
  exit 1
fi

mkdir -p "${COIN_OR_ROOT}" "${IPOPT_PREFIX}"
cd "${COIN_OR_ROOT}"

if [[ -n "${OPENBLAS_ROOT:-}" ]]; then
  export PKG_CONFIG_PATH="${OPENBLAS_ROOT}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  if [[ -z "${IPOPT_LAPACK_LFLAGS}" ]]; then
    IPOPT_LAPACK_LFLAGS="-L${OPENBLAS_ROOT}/lib -lopenblas"
  fi
  if [[ -z "${IPOPT_BLAS_LFLAGS}" ]]; then
    IPOPT_BLAS_LFLAGS="${IPOPT_LAPACK_LFLAGS}"
  fi
fi

if [[ ! -x "${COINBREW}" ]]; then
  echo "Downloading coinbrew into ${COINBREW}"
  ${download_cmd} "${COINBREW}" https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
  chmod u+x "${COINBREW}"
fi

echo "Source root: ${COIN_OR_ROOT}"
echo "Install prefix: ${IPOPT_PREFIX}"
echo "coinbrew: ${COINBREW}"
if [[ -n "${IPOPT_LAPACK_LFLAGS}" ]]; then
  echo "Lapack/BLAS linker flags: ${IPOPT_LAPACK_LFLAGS}"
fi
if [[ -n "${IPOPT_BLAS_LFLAGS}" ]]; then
  echo "BLAS linker flags: ${IPOPT_BLAS_LFLAGS}"
fi
if [[ -n "${PKG_CONFIG_PATH:-}" ]]; then
  echo "PKG_CONFIG_PATH: ${PKG_CONFIG_PATH}"
fi

"${COINBREW}" fetch Ipopt --no-prompt

build_args=(
  Ipopt
  --prefix="${IPOPT_PREFIX}"
  --no-prompt
  --verbosity=3
)

if [[ -n "${IPOPT_LAPACK_LFLAGS}" ]]; then
  build_args+=("--with-lapack-lflags=${IPOPT_LAPACK_LFLAGS}")
fi
if [[ -n "${IPOPT_BLAS_LFLAGS}" ]]; then
  build_args+=("--with-blas-lflags=${IPOPT_BLAS_LFLAGS}")
fi

if [[ -n "${COINBREW_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${COINBREW_EXTRA_ARGS} )
  build_args+=("${extra_args[@]}")
fi

if [[ "${COINBREW_TEST}" == "1" ]]; then
  build_args+=(--test)
fi

echo "Building Ipopt with coinbrew..."
"${COINBREW}" build "${build_args[@]}"

if [[ ! -x "${IPOPT_PREFIX}/bin/ipopt" ]]; then
  echo "coinbrew build finished, but ${IPOPT_PREFIX}/bin/ipopt was not found."
  echo "Trying explicit install step..."
  "${COINBREW}" install Ipopt --no-prompt || true
fi

if [[ ! -x "${IPOPT_PREFIX}/bin/ipopt" ]]; then
  echo "ERROR: install completed but ipopt executable was not found in ${IPOPT_PREFIX}/bin."
  echo "Check the coinbrew build output above for BLAS/LAPACK, MUMPS, or ASL issues."
  echo "If LAPACK was not found, rerun with something like:"
  echo "  export IPOPT_LAPACK_LFLAGS='-L/path/to/lib -lopenblas'"
  echo "  export IPOPT_BLAS_LFLAGS='-L/path/to/lib -lopenblas'"
  echo "  bash scripts/install_ipopt_coinbrew.sh"
  exit 1
fi

echo
echo "Ipopt installation complete."
echo "To use it in this shell:"
echo "  export PATH=\"${IPOPT_PREFIX}/bin:\$PATH\""
echo "  export LD_LIBRARY_PATH=\"${IPOPT_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\""
echo
echo "Quick checks:"
echo "  which ipopt"
echo "  ipopt --print-options | head -n 40"
echo "  python benchmarks/run_stage.py --stage solver-check --run-name after_ipopt_install"
