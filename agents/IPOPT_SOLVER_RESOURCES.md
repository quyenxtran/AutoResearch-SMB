# IPOPT Solver Resources

Use this file as a compact menu for profile selection and fallback logic.

## Intended Use

- Select solver/profile based on phase: screening, refinement, validation.
- Prefer verified runtime exports from Slurm when available.
- Do not assume a linear solver is available until verified.

## General Selection Rules

1. Start with robust profile for current phase.
2. If failure is numerical, switch linear solver/profile before changing physics.
3. If failure persists across profiles and starts, treat as likely infeasible region.
4. Revalidate final candidate with strict validation profile.

## Profile Menu

Recommended profile intents:

- `baseline_reproduction`: reproduce reference behavior.
- `screening_fast`: faster exploratory runs.
- `balanced_cpu_mumps`: stable screening/refinement default.
- `robust_cpu_mumps`: higher robustness when screening fails.
- `robust_cpu_ma97`: hard-problem rescue profile when available.
- `high_end_pardiso`: performance-oriented profile when available.
- `validation_strict`: final report-quality validation.

## Fallback Trees

Use as ordered fallback chains:

- baseline: `baseline_reproduction -> robust_cpu_mumps -> validation_strict`
- screening: `screening_fast -> balanced_cpu_mumps -> baseline_reproduction`
- hard-problem: `robust_cpu_ma97 -> robust_cpu_mumps -> validation_strict`
- high-performance: `high_end_pardiso -> robust_cpu_ma97 -> robust_cpu_mumps`

If a solver in chain is unavailable, skip to next verified option.

## Reporting Requirement

For each run, log:

- solver executable
- linear solver
- profile name
- termination condition
- key tolerances (`tol`, `acceptable_tol`, `max_iter`)

Final claims must include at least one strict-validation run.
