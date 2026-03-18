# SKILLS.md - SMB Physics and Operational Fundamentals

This file contains physics and numerical fundamentals. It does not encode "best known" operating points.

## 1. SMB Zone Functions

In 4-zone SMB:

- `Z1` (desorbent in -> extract out): desorption of strongly adsorbed species.
- `Z2` (extract out -> feed in): purification / impurity cleanup on extract side.
- `Z3` (feed in -> raffinate out): adsorption/capture behavior for target split.
- `Z4` (raffinate out -> desorbent in): regeneration before returning to Z1.

Changing `nc = (n1, n2, n3, n4)` redistributes residence volume across these functions.

## 2. Flow Mass Balance (Critical Invariant)

Must hold within tolerance:

- `F1 = Fdes + Fex`
- `F1 = Ffeed + Fraf`

Derived zone flows:

- `F2 = F1 - Fex`
- `F3 = F2 + Ffeed`
- `F4 = F1 - Fdes`

Reject candidates that violate balance materially.

## 3. Equilibrium Theory and Feasibility Conditions

MLL isotherm is nonlinear and competitive; analytic triangle guarantees from linear systems do not directly apply.
Practical feasibility still depends on zone functions being preserved under chosen flow/switch settings.

## 4. MLL Isotherm - What the Parameters Mean

- `H`: affinity at dilute limit (selectivity driver).
- `qm`: saturation capacity.
- `K`: nonlinearity/curvature.
- `kapp`: mass-transfer approach to equilibrium.

Competition means component behavior is coupled; isolated intuition can fail.

## 5. Switching Time and Cyclic Steady State

`tstep` controls front travel per switch:

- too short -> insufficient separation development
- too long -> front overshoot and cross-contamination

`tstep` must be interpreted jointly with flows and layout.

## 6. Multi-Fidelity Discretization

Lower fidelity is faster but less accurate near active constraints.
Use ladder:

1. low/medium for screening
2. medium for local refinement
3. high for final claims

## 7. Solver Status Interpretation

- `optimal` / `ok`: converged candidate; still check metrics.
- `infeasible`: no feasible point from that start/setting.
- `solver_error`: numerical failure; not proof of physical impossibility.
- `acceptable`: loose convergence; re-check with stricter settings.
- `maxIterations`: iteration budget exhausted.

## 8. Purity and Recovery - Exact Definitions

Use code definitions in `metrics.py`:

- purity is MeOH-free extract purity metric used by project.
- recovery metrics are component-specific extract recoveries.
- productivity is extract acid throughput metric used for objective.

Do not substitute alternative formulas in comparative claims.

## 9. Physical Hardware Constraints

Hardware-consistent rules:

- external pump-limited streams stay within external pump capacity.
- internal circulation (`F1`) obeys internal limit.
- total columns fixed at 8.
- all zone column counts remain physically meaningful.

## 10. What the Agents Should Discover, Not Be Told

Must be learned from runs, not assumed:

- best `nc` for current feed/constraints
- best flow/tstep region
- local sensitivity around high-performing candidates
- robustness margins and failure boundaries
