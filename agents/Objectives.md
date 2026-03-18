# SMB Optimization Objective (Agent-Driven NLP Optimizer)

## Mission

Use the local `sembasmb` implementation as the physics source of truth. Optimize the SMB process for the Kraton-feed case by choosing:

- operating variables (`F1`, `Fdes`, `Fex`, `Ffeed`, `tstep`)
- section layout (`nc`) over fixed 8-column hardware
- fidelity and solver strategy for efficient, reproducible search

Do not redesign chemistry or hardware unless explicitly requested.

Reference code modules:

- `src/sembasmb/config.py`
- `src/sembasmb/model.py`
- `src/sembasmb/discretization.py`
- `src/sembasmb/optimization.py`
- `src/sembasmb/metrics.py`
- `src/sembasmb/solver.py`

## Optimization Goal

Primary objective:

- maximize `productivity_ex_ga_ma = (CE_GA + CE_MA) * UE * area * eb`

Subject to hard constraints in this file and runtime bounds from the active Slurm/job environment.

## Components and Basis

Component order must be fixed everywhere:

1. `GA`
2. `MA`
3. `Water`
4. `MeOH`

Kraton-feed mass fractions (`wt0`) and densities (`rho`):

- `wt0 = (0.003, 0.004, 0.990, 0.003)`
- `rho = (1.5, 1.6, 1.0, 0.79)`

Internal feed concentration basis is computed exactly as:

- `CF = wt0 / sum(wt0 / rho)`

## Desorbent Composition

Use pure methanol desorbent unless explicitly changed:

- desorbent mass fractions: `(0.0, 0.0, 0.0, 1.0)`
- internal `CD` in current code: `(0.0, 0.0, 0.0, 0.79)`

## SMB Configuration

Default reference configuration:

- total columns fixed: `8`
- reference layout: `nc = (1, 2, 3, 2)`
- default high-fidelity mesh: `nfex = 10`, `nfet = 5`, `ncp = 2`
- fixed geometry: `L = 20.0`, `d = 1.0`, `eb = 0.44`, `ep = 0.66`
- transport model: `Pe = 1000.0`, `isoth = "MLL"`, `xscheme = "CENTRAL"`

Fixed isotherm/transport parameters unless a separate sensitivity study is requested:

- `kapp = (0.8, 1.22, 1.0, 0.69)`
- `qm = (0.084, 0.117, 0.02, 0.05)`
- `K = (254.0, 1208.0, 1e-3, 79.0)`
- `H = (0.61, 0.52, 1e-3, 0.06)`

Admissible layout rule:

- `sum(nc) = 8`
- each zone must remain physically meaningful (no zero-column zones)

## Decision Variables vs Derived Quantities

Optimize directly:

- `F1`, `Fdes`, `Fex`, `Ffeed`, `tstep`, `nc`

Treat as derived unless explicitly reformulated:

- `Fraf`

Flow invariants must hold:

- `F1 = Fdes + Fex`
- `F1 = Ffeed + Fraf`

## Hard Operating Constraints

Use runtime-exported bounds when provided. Otherwise use these defaults:

- `0.5 <= Ffeed <= 2.5` mL/min
- external pump streams (`Fdes`, `Fex`, `Ffeed`, `Fraf`) `<= 2.5` mL/min
- internal circulation `F1 <= 5.0` mL/min
- all flows positive

Project-level quality targets are controlled by runtime env/CLI, with current defaults:

- `purity_ex_meoh_free >= 0.60`
- `recovery_ex_GA >= 0.75`
- `recovery_ex_MA >= 0.75`

Exploratory thresholds may differ from project thresholds. Final reporting uses project targets.

## Required Workflow

Use this order:

1. Verify solver/model availability and baseline feasibility.
2. Screen NC layouts with reference seed at low/medium fidelity.
3. Expand flows/seeds only after layout evidence exists.
4. Refine top candidates.
5. Perform final high-fidelity validation for reporting.

Do not spend high-fidelity budget on broad blind screening.

## Mandatory NC-Coverage and Comparative Reasoning

Before deep exploitation:

- cover all admissible NC layouts in the active library at least once with reference seed
- compare proposed run against:
  - current best run
  - at least one recent failed or near-failed run

Comparison blocks must be quantitative:

- productivity, purity, recovery, violation
- explicit flow deltas: `dFfeed`, `dF1`, `dFdes`, `dFex`, `dFraf`, `dtstep`
- explicit topology deltas: `dZ1`, `dZ2`, `dZ3`, `dZ4`

## Insights and Trends Ledger (Required in `research.md`)

Every iteration must append:

- selected candidate and reason
- metrics vs prior evidence
- key physics interpretation
- next hypothesis or decision branch

## Simulation Priority Policy

Prioritize in this order:

1. Feasibility and violation reduction
2. Meeting project purity/recovery thresholds
3. Productivity maximization inside feasible region

## LLM Runtime and Fallback Policy

- Use local model first when enabled.
- Keep prompts compact and evidence-first.
- If local LLM is unavailable, use deterministic fallback policy.
- Do not claim improvement without cited run evidence.

## Final Deliverables

At run end, provide:

- best validated candidate (`nc`, flows, fidelity, solver profile)
- objective and constraint metrics
- evidence of robustness (comparison to alternatives and recent failures)
- reproducibility metadata (run name, artifacts, sqlite path, solver settings)
