# IPOPT Solver Resources

Compact solver/profile guide for screening, refinement, and validation.

## Selection Rules

1. Start with phase-appropriate verified profile.
2. If numerical failure, switch profile/linear solver before changing physics.
3. If repeated failure across profiles/starts, treat as likely infeasible region.
4. Revalidate final claim with strict profile.

## Profile Menu

- `baseline_reproduction`
- `screening_fast`
- `balanced_cpu_mumps`
- `robust_cpu_mumps`
- `robust_cpu_ma97`
- `high_end_pardiso`
- `validation_strict`

## Fallback Trees

- baseline: `baseline_reproduction -> robust_cpu_mumps -> validation_strict`
- screening: `screening_fast -> balanced_cpu_mumps -> baseline_reproduction`
- hard-problem: `robust_cpu_ma97 -> robust_cpu_mumps -> validation_strict`
- high-performance: `high_end_pardiso -> robust_cpu_ma97 -> robust_cpu_mumps`

Skip unavailable options in chain order.

## Per-Run Logging Requirement

Log:

- executable
- linear solver
- profile
- termination condition
- key tolerances (`tol`, `acceptable_tol`, `max_iter`)

Final claim must include at least one strict-validation run.
