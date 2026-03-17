# IPOPT Solver Resources

This file defines a menu of available Ipopt-style solver configurations for the SMB optimization agents.

These are resources only.

They are not instructions to always use a particular solver or profile. The agents must:

- verify installation and availability first
- choose the simplest profile that matches the task
- record which profile was selected and why
- fall back if the preferred profile is unavailable or unstable

If the Slurm job exports verified Ipopt resource variables, those verified variables take precedence over this static menu.

Preferred verified variables:

- `SMB_VERIFIED_IPOPT_EXECUTABLES`
- `SMB_VERIFIED_IPOPT_LINEAR_SOLVERS`
- `SMB_VERIFIED_IPOPT_PROFILE_MENU`
- `SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE`
- `SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE`
- `SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE`
- `SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE`
- `SMB_IPOPT_PRECHECK_SUMMARY`

## Intended Use

Use these profiles for four different purposes:

- baseline reproduction
- low-fidelity screening
- difficult or unstable solves
- final high-confidence validation

## General Selection Rules

- Prefer `ipopt_sens` when reproducing the existing notebook or local SembaSMB behavior.
- Prefer tighter tolerances for final validation than for screening.
- Prefer looser tolerances only for coarse exploration, never for the final reported answer.
- If a faster profile gives a materially different answer, re-check the candidate with a stricter profile.
- If a linear solver is not installed, skip that profile and move to its fallback.

## Profile Menu

### `baseline_reproduction`

- Goal: match current notebook behavior as closely as possible
- solver executable: `ipopt_sens`
- linear solver: `ma57`
- options:
  - `linear_solver = ma57`
  - `mu_init = 1e-3`
  - `max_iter = 5000`
  - `tol = 1e-6`
  - `acceptable_tol = 1e-5`
  - `halt_on_ampl_error = yes`
- Use when:
  - reproducing the existing SembaSMB notebook
  - establishing baseline trust
  - validating the final answer against the reference stack
- Fallback:
  - `robust_cpu_mumps`

### `screening_fast`

- Goal: cheap coarse sweeps and feasibility scouting
- solver executable: `ipopt_sens` if available, otherwise `ipopt`
- preferred linear solver: `mumps`
- options:
  - `linear_solver = mumps`
  - `mu_init = 1e-2`
  - `max_iter = 1200`
  - `tol = 1e-5`
  - `acceptable_tol = 1e-4`
  - `halt_on_ampl_error = yes`
  - `print_level = 0`
- Use when:
  - scanning many points
  - low-fidelity model screening
  - searching for warm-start candidates
- Fallback:
  - `baseline_reproduction`

### `balanced_cpu_mumps`

- Goal: general-purpose CPU solve with moderate robustness
- solver executable: `ipopt_sens`
- linear solver: `mumps`
- options:
  - `linear_solver = mumps`
  - `mu_init = 1e-3`
  - `max_iter = 4000`
  - `tol = 1e-6`
  - `acceptable_tol = 1e-5`
  - `halt_on_ampl_error = yes`
- Use when:
  - `ma57` is unavailable
  - a portable CPU option is needed
  - the model is medium difficulty but not extremely unstable
- Fallback:
  - `robust_cpu_ma97`

### `robust_cpu_ma97`

- Goal: harder nonlinear solves if MA97 exists
- solver executable: `ipopt_sens`
- linear solver: `ma97`
- options:
  - `linear_solver = ma97`
  - `mu_init = 1e-4`
  - `max_iter = 6000`
  - `tol = 1e-7`
  - `acceptable_tol = 1e-6`
  - `halt_on_ampl_error = yes`
- Use when:
  - the baseline profile struggles
  - the problem appears numerically sensitive
  - a stricter validation solve is needed
- Fallback:
  - `robust_cpu_mumps`

### `robust_cpu_mumps`

- Goal: portable robust fallback
- solver executable: `ipopt_sens`
- linear solver: `mumps`
- options:
  - `linear_solver = mumps`
  - `mu_init = 1e-4`
  - `max_iter = 6000`
  - `tol = 1e-7`
  - `acceptable_tol = 1e-6`
  - `halt_on_ampl_error = yes`
- Use when:
  - another linear solver is unavailable
  - a more conservative solve is needed
  - final candidate needs an alternative confirmation on CPU
- Fallback:
  - `validation_strict`

### `high_end_pardiso`

- Goal: high-performance CPU solve if Pardiso is available
- solver executable: `ipopt_sens`
- linear solver: `pardiso`
- options:
  - `linear_solver = pardiso`
  - `mu_init = 1e-3`
  - `max_iter = 5000`
  - `tol = 1e-7`
  - `acceptable_tol = 1e-6`
  - `halt_on_ampl_error = yes`
- Use when:
  - Pardiso is installed and licensed if required
  - a high-end CPU option is available
  - runtime is important but correctness still matters
- Fallback:
  - `robust_cpu_ma97`

### `validation_strict`

- Goal: final confirmation of a near-final or final candidate
- solver executable: `ipopt_sens`
- preferred linear solver: `ma57`
- options:
  - `linear_solver = ma57`
  - `mu_init = 1e-4`
  - `max_iter = 8000`
  - `tol = 1e-8`
  - `acceptable_tol = 1e-7`
  - `halt_on_ampl_error = yes`
- Use when:
  - confirming the final point
  - checking whether a candidate remains feasible under stricter settings
  - ranking a small number of finalists
- Fallback:
  - `robust_cpu_ma97`
  - then `robust_cpu_mumps`

## Fallback Trees

### Baseline tree

- `baseline_reproduction`
- then `robust_cpu_mumps`
- then `validation_strict`

### Screening tree

- `screening_fast`
- then `balanced_cpu_mumps`
- then `baseline_reproduction`

### Hard-problem tree

- `robust_cpu_ma97`
- then `robust_cpu_mumps`
- then `validation_strict`

### High-performance CPU tree

- `high_end_pardiso`
- then `robust_cpu_ma97`
- then `robust_cpu_mumps`

## Reporting Requirement

Whenever a scientist chooses an Ipopt profile, record:

- profile name
- solver executable
- linear solver
- whether it was available immediately or chosen as fallback
- why it was selected
- whether the final accepted answer was rechecked with a stricter profile
