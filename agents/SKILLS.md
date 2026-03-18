# SMB Physics Essentials

## Zone Functions

For 4-zone SMB:

- `Z1`: desorb strongly adsorbed species
- `Z2`: extract-side cleanup
- `Z3`: feed-side adsorption/capture split
- `Z4`: regeneration

Changing `nc=(n1,n2,n3,n4)` changes residence distribution across these functions.

## Flow Mass Balance (Critical)

Must hold within tolerance:

- `F1 = Fdes + Fex`
- `F1 = Ffeed + Fraf`

Derived:

- `F2 = F1 - Fex`
- `F3 = F2 + Ffeed`
- `F4 = F1 - Fdes`

## Switching Time

`tstep` interacts with flows and layout:

- too short: under-developed separation
- too long: front overshoot/cross-contamination

## MLL Isotherm Notes

- nonlinear, competitive coupling
- `H`, `qm`, `K`, `kapp` jointly shape selectivity/capacity/transfer
- single-component intuition can fail in multicomponent runs

## Multi-Fidelity Policy

1. low/medium for screening
2. medium for local refinement
3. high for final claims

## Solver Status Interpretation

- `optimal`/`ok`: candidate converged; still check constraints
- `infeasible`: no feasible point found at setup/start
- `solver_error`: numerical failure, not automatic physical impossibility
- `acceptable`: loose convergence; re-check
- `maxIterations`: iteration cap hit

## Purity and Recovery Definitions

Use exact code metrics in `metrics.py`. Do not substitute custom formulas in claims.

## Physical Hardware Constraints

- external streams within pump limits
- `F1` within internal circulation limit
- total columns fixed at 8
- zone column counts physically meaningful

## What Agents Must Learn From Data

- best feasible `nc`
- best flow/`tstep` region
- local sensitivity around top candidates
- robustness margins/failure boundaries
