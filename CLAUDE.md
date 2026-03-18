# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -r requirements.txt
pip install -e ".[dev]"          # includes pytest, ruff, mypy

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_config.py::TestBuildInputs::test_velocities_are_positive  # single test

# Lint / type check
ruff check src/ benchmarks/ tests/
mypy src/sembasmb/

# Manual verification (no IPOPT needed)
python -c "from sembasmb import SMBConfig, build_inputs, build_model; print('imports OK')"
python -m benchmarks.agent_runner --help
python -m benchmarks.run_stage --help
```

> `pyproject.toml` sets `pythonpath = ["src"]` so pytest finds `sembasmb` automatically. For scripts run outside pytest, `benchmarks/run_stage.py` manually inserts `REPO_ROOT/src` into `sys.path`.

## Architecture

### Core simulation pipeline (`src/sembasmb/`)

The pipeline is strictly sequential — each step feeds the next:

```
SMBConfig + FlowRates
    └─► build_inputs()        → SMBInputs   (config.py)
            └─► build_model()  → Pyomo model (model.py)
                    └─► apply_discretization()          (discretization.py)
                            ├─► [add_optimization()]    (optimization.py)  ← only for NLP optimization runs
                            └─► solve_model()           (solver.py)
                                    └─► compute_outlet_averages()
                                        compute_purity_recovery()          (metrics.py)
                                        plot_profiles()                    (plotting.py)
```

**`config.py`** — `SMBConfig` is a frozen dataclass holding all physical/chemical constants. `build_inputs()` converts volumetric flows (mL/min) into interstitial velocities (cm/min), computes column cross-sectional area, resolves isotherm parameters from `isotherm.py`, and packs everything into an `SMBInputs` frozen dataclass. All downstream code works from `SMBInputs` — never from `SMBConfig` directly.

**`model.py`** — Builds a Pyomo `ConcreteModel` for a single SMB cycle. Key design: isotherm parameters (`kapp`, `qm`, `K`, `H`) are declared as `Var` but immediately fixed — they become free only when add_optimization() unfixes the decision variables. The spatial dimension `x` and time `t` are both normalized to `[0, 1]`. Column index `col` runs 1…ncols (total columns across all zones).

**`discretization.py`** — Applies Pyomo DAE transformations: Lagrange-Radau collocation in time (`nfet` finite elements, `ncp` collocation points) and finite differences in space (`nfex` elements, `CENTRAL` or `BACKWARD` scheme). Also adds the cyclic steady-state constraints (CSSC/CSSCp/CSSQ) that connect the end of one column's period to the start of the next — this is the key SMB periodicity constraint.

**`optimization.py`** — `add_optimization()` unfixes the flow velocity variables and switching time, sets their bounds, adds time-averaged outlet concentration variables (`ce_acid`, `cr_acid`, etc.), then adds hard constraints for purity/recovery/solvent limits and an objective to maximize extract acid productivity. After this call, the model is ready for `solve_model()`.

**`metrics.py`** — Post-solve: time-averages outlet concentrations at `x=1.0` of the extract and raffinate columns. Returns purity (MeOH-free basis), per-component recovery, and productivity. All dict keys use 1-based integer component indices, not component name strings.

**`isotherm.py`** — Single function `get_isotherm_params(isoth)`. Three types: `'MLL'` (4-component Modified Langmuir, main production model), `'MLLE'` (extended version with tighter H bounds), `'L'` (linear, 3-component). The isotherm type controls H variable bounds in `model.py`.

### Fidelity ladder

All optimization runs use one of three discretization levels:

| Level  | nfex | nfet | ncp | Use case |
|--------|------|------|-----|----------|
| Low    | 4–5  | 2    | 1   | Layout screening, fast feasibility check |
| Medium | 6    | 3    | 2   | Candidate refinement |
| High   | 10   | 5    | 2   | Final validation (reference/production) |

Never jump from low to high directly — numerical instability risk. Always use low → medium → high.

### Flow mass balance (critical invariant)

```
F1 = Ffeed + Fraf      (feed zone balance)
F1 = Fdes + Fex        (desorbent zone balance)
```

`Fraf` is derived (never optimized independently). `optimization.py` enforces this as `RaffinateConsistency`. Any run that violates this by >1% will produce solver errors or unphysical results.

### Benchmark / agent layer (`benchmarks/`)

**`run_stage.py`** — Standalone CLI that runs a single optimization stage (`solver-check`, `reference-eval`, `nc-screen`, `flow-screen`, `optimize-layouts`). Contains `NOTEBOOK_SEEDS` — 8 reference operating points from the Kraton feed notebook that serve as warm-start initializations. `IpoptLiveMonitor` runs in a background thread, tailing the IPOPT log and killing the solver if it stalls (watchdog).

**`agent_runner.py`** — The LLM orchestration layer. Implements the three-scientist loop:
- **Scientist_A** (proposer): reads `agents/` knowledge files + SQLite history + heuristics (hypotheses.json/failures.json) + convergence tracker, proposes next candidate with acquisition reasoning (EXPLORE/EXPLOIT/VERIFY)
- **Scientist_B** (reviewer): independently validates A's proposal against physics and prior data
- **Scientist_Executive** (moderator): rules on every A/B disagreement with 5-ruling taxonomy (IMPLEMENT_A, IMPLEMENT_B_COUNTER, IMPLEMENT_HYBRID, RETURN_FOR_REVISION, FORCE_DIAGNOSTIC)

All experiment results are persisted to a SQLite DB (`smb_agent_context.sqlite`). The `convergence_tracker` table records best-feasible-J after each simulation for sample efficiency comparison against the MINLP baseline. The LLM context is built from: objectives, soul, IPOPT resources, heuristics files, convergence tracker, `research.md` tail, and the full SQLite history.

### Knowledge files (`agents/`)

| File | Format | Purpose |
|------|--------|---------|
| `hypotheses.json` | JSON | 11 structured hypotheses with `simulation_results[]` array — agents append evidence after each run |
| `failures.json` | JSON | 11 failure modes with `occurrences[]` array — agents log errors here |
| `SKILLS.md` | Markdown | Proven physical intuition and optimization patterns (durable) |
| `Objectives.md` | Markdown | Project targets: purity ≥ 0.60, recovery GA/MA ≥ 0.75 (exploratory targets) |
| `LLM_SOUL.md` | Markdown | Agent operating principles |
| `IPOPT_SOLVER_RESOURCES.md` | Markdown | Solver configuration guide |

When updating hypotheses or failures after a run, edit `hypotheses.json` and `failures.json` (not the deprecated `.md` versions). Add to `simulation_results[]` or `occurrences[]` respectively.

### SLURM jobs (`slurm/`)

Submit from `~/AutoResearch-SMB` (repo root):

```bash
sbatch slurm/pace_smb_stage_runner.slurm          # single stage
sbatch slurm/pace_smb_two_scientists_24h.slurm    # two-scientist agent run
sbatch slurm/pace_smb_minlp_cpu_24h.slurm         # direct MINLP baseline
```

See `slurm/local_command.md` for the standard 3-job comparative batch submission pattern. Key env vars: `SMB_NC_LIBRARY`, `SMB_STAGE`, `SMB_RUN_NAME`, `SMB_SOLVER_NAME`, `SMB_LINEAR_SOLVER`.

The SLURM scripts auto-detect `AUTORESEARCH_ROOT` by checking for `benchmarks/run_stage.py` + `src/sembasmb/` at `REPO_ROOT`. `SMB_ROOT` is set to `${AUTORESEARCH_ROOT}/src`.

### Import paths

The package is `sembasmb` (under `src/`). The README still shows stale `SembaSMB.src.smb_*` paths — ignore those. Correct usage:

```python
from sembasmb import SMBConfig, FlowRates, build_inputs, build_model
from sembasmb import apply_discretization, add_optimization, solve_model
from sembasmb import compute_outlet_averages, compute_purity_recovery, plot_profiles
```
