# SMB Optimization Objective

## Mission

Use local `sembasmb` as source of truth. Optimize Kraton-feed SMB by selecting:

- operating variables: `F1`, `Fdes`, `Fex`, `Ffeed`, `tstep`
- layout: `nc` over fixed 8 columns
- fidelity/solver strategy for reproducible search

No chemistry/hardware redesign unless explicitly requested.

## Optimization Goal

Maximize:

- `productivity_ex_ga_ma = (CE_GA + CE_MA) * UE * area * eb`

Subject to runtime bounds and hard constraints.

## Components and Basis

Fixed component order:

1. `GA`
2. `MA`
3. `Water`
4. `MeOH`

Kraton feed defaults:

- `wt0 = (0.003, 0.004, 0.990, 0.003)`
- `rho = (1.5, 1.6, 1.0, 0.79)`
- `CF = wt0 / sum(wt0 / rho)`

## Desorbent Composition

Default desorbent is pure MeOH:

- mass fraction `(0,0,0,1)`

## SMB Configuration

Reference:

- total columns: `8`
- reference `nc = (1,2,3,2)`
- high-fidelity mesh: `nfex=10, nfet=5, ncp=2`

Layout rule:

- `sum(nc)=8`
- no zero-column zones

## Decision Variables vs Derived Quantities

Optimize:

- `nc`, `F1`, `Fdes`, `Fex`, `Ffeed`, `tstep`

Derived unless reformulated:

- `Fraf`

Flow invariants:

- `F1 = Fdes + Fex`
- `F1 = Ffeed + Fraf`

## Hard Operating Constraints

Use runtime-exported bounds; defaults:

- `0.5 <= Ffeed <= 2.5` mL/min
- external streams `<= 2.5` mL/min
- `F1 <= 5.0` mL/min
- all flows positive

Project defaults:

- `purity_ex_meoh_free >= 0.60`
- `recovery_ex_GA >= 0.75`
- `recovery_ex_MA >= 0.75`

## Required Workflow

1. Solver/model check and baseline feasibility.
2. Screen all NC with reference seed.
3. Expand seeds/flows only after layout evidence.
4. Refine top feasible candidates.
5. Final high-fidelity validation for claims.

## Mandatory NC-Coverage and Comparative Reasoning

Before deep exploitation:

- cover active NC library with reference seed probes
- compare against current best and recent failure
- include numeric evidence: productivity/purity/recovery/violation
- include flow deltas: `dFfeed,dF1,dFdes,dFex,dFraf,dtstep`
- include topology deltas: `dZ1,dZ2,dZ3,dZ4`

## Insights and Trends Ledger

Each iteration logs:

- chosen candidate and why
- metric comparison vs prior runs
- physics interpretation
- next hypothesis

## Simulation Priority Policy

Priority order:

1. feasibility and violation reduction
2. quality constraints
3. productivity inside feasible region

## LLM Runtime and Fallback Policy

- local model first when enabled
- concise evidence-first prompts
- deterministic fallback when LLM unavailable

## Final Deliverables

- best validated candidate (`nc`, flows, fidelity, solver profile)
- objective/constraint metrics
- comparison vs alternatives/failures
- reproducibility metadata (run name, sqlite/artifacts, solver settings)
