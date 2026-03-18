# Problem Definition

## Core Question

Under fixed compute budget, find a high-productivity feasible SMB operating point for Kraton feed, and do it with a reliable search policy.

## Optimization Problem

Objective:

- maximize `productivity_ex_ga_ma`

Constraints:

- quality thresholds (runtime project targets)
- flow bounds and pump limits
- flow consistency: `F1 = Fdes + Fex = Ffeed + Fraf`
- fixed hardware: 8 columns, `nc` must be admissible

Main variables:

- `nc`, `Ffeed`, `F1`, `Fdes`, `Fex`, `tstep`

Derived:

- `Fraf` (unless explicitly reformulated)

## What Kind of Optimization Problem?

- fixed `nc`: large nonconvex NLP
- variable `nc`: mixed discrete+continuous search

Global optimum is not guaranteed with local NLP solvers; target is best validated solution under finite budget.

## Practical Benchmark Framing

Hold constant across methods:

- model/feed assumptions
- NC library
- bounds/constraints
- final high-fidelity validator

Compare methods by:

- feasibility rate
- best validated productivity
- time/evals to best feasible
- robustness of final point

## Fixed-Budget Rule

Respect exported job budget. Reserve final validation budget; do not spend it early without justification.

## Five-Hour Benchmark Protocol

If using 5h mode:

- 4h search/refinement
- 1h final validation

## Recommended Success Criteria

Strong result includes:

- feasible high-fidelity candidate meeting targets
- strong productivity
- evidence-backed competitor comparisons
- reproducibility metadata (solver/profile/run IDs)

## Minimal Evidence For Final Claim

- final high-fidelity metrics
- at least two meaningful competitor comparisons
- explicit nearby failure-mode discussion
- local perturbation/sensitivity evidence

## Bottom Line

Feasibility first, optimization second, validation last, claims only from evidence.
