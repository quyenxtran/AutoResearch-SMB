# Agent-Driven NLP Optimizer

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Pyomo](https://img.shields.io/badge/Pyomo-6.0+-orange.svg)](https://pyomo.org)
[![IPOPT](https://img.shields.io/badge/IPOPT-3.14+-green.svg)](https://coin-or.github.io/Ipopt/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A framework for agent-driven nonlinear programming (NLP) optimization. A team of LLM agents (proposer, reviewer, executive moderator) iteratively guide an IPOPT-based solver through a multi-fidelity search, replacing the need for a human expert to manually tune the optimization strategy.

The core hypothesis: **an LLM agent that reasons about which experiment to run next can reach near-optimal solutions in fewer simulations than brute-force MINLP**, especially when data is scarce and each solver call is expensive.

**Reference use case**: SMB chromatography — maximizing organic acid (GA/MA) productivity subject to purity and recovery constraints on an 8-column Simulated Moving Bed unit with 35 admissible column layouts and a 5-dimensional continuous flow space.

## Why Agent-Driven Optimization?

Traditional approaches to NLP with discrete structure (like column layout selection + continuous flow optimization) either:

- **Exhaustive grid search**: Evaluate all NC layouts x all seeds x all flow combinations. Thorough but expensive — hundreds of solver calls with no intelligence about ordering.
- **Direct MINLP**: Hand the full mixed-integer problem to a global solver. Often intractable for complex DAE-constrained models.

The agent-driven approach treats each simulation as an **expensive experiment** and uses LLM reasoning to decide what to simulate next:

1. **Data-scarce decision-making** — With only a few completed simulations, the agent forms physics-grounded hypotheses about which regions of the design space are most promising, rather than sampling blindly.
2. **Acquisition strategy** — Every proposal is classified as EXPLORE (cover untested regions), EXPLOIT (refine near the best known point), or VERIFY (confirm optimality via perturbation). The agent tracks its own exploration/exploitation balance.
3. **Triple-grounded justification** — Every candidate must be justified by DATA (SQLite history + convergence tracker), PHYSICS (zone theory, mass balance, isotherm behavior), and HEURISTICS (accumulated hypotheses and failure patterns).
4. **Convergence tracking** — Best-feasible-J is recorded after every simulation for both the agent and the MINLP baseline, enabling direct sample-efficiency comparison.

## Installation

```bash
git clone https://github.com/your-org/Agent-Driven-NLP-Optimizer.git
cd Agent-Driven-NLP-Optimizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"          # includes pytest, ruff, mypy
```

IPOPT must be on `PATH`. Optional linear solvers: MA57, MA97, MUMPS, Pardiso (see `agents/IPOPT_SOLVER_RESOURCES.md`).

For local LLM inference, install [Ollama](https://ollama.com/) and pull a Qwen model. See `slurm/local_command.md` for PACE cluster submission commands.

## Quick Start

```bash
python -m benchmarks.run_stage --help       # single optimization stage
python -m benchmarks.agent_runner --help    # agent-driven optimization loop
```

### Python API (SMB use case)

```python
from sembasmb import SMBConfig, FlowRates, build_inputs, build_model
from sembasmb import apply_discretization, add_optimization, solve_model
from sembasmb import compute_outlet_averages, compute_purity_recovery

config = SMBConfig(nc=(1, 2, 3, 2), nfex=10, nfet=5, ncp=2)
flow   = FlowRates(F1=2.5, Fdes=1.5, Fex=1.0, Ffeed=1.2, tstep=9.4)

inputs = build_inputs(config, flow)
model  = build_model(config, inputs)
apply_discretization(model, inputs)
add_optimization(model, inputs)
results = solve_model(model, solver_name="ipopt_sens", linear_solver="ma57")
```

## Agent Architecture

Three agents share a SQLite experiment database (`smb_agent_context.sqlite`) with two tables: `simulation_results` (full run history) and `convergence_tracker` (best-so-far after each simulation):

- **Scientist_A** (proposer) — reads knowledge files, SQLite history, heuristics (`hypotheses.json`, `failures.json`), and the convergence tracker. Proposes the next experiment with acquisition reasoning: what type (EXPLORE/EXPLOIT/VERIFY), what it will teach, what alternatives were considered, and what hypothesis it tests.
- **Scientist_B** (reviewer) — independently validates A's proposal against physics fundamentals and prior data. Classifies objections as Hard Block (physics violation, contradicting data) or Soft Concern (preference without evidence).
- **Scientist_Executive** (moderator) — rules on every A/B disagreement with a 5-ruling taxonomy: IMPLEMENT_A, IMPLEMENT_B_COUNTER, IMPLEMENT_HYBRID, RETURN_FOR_REVISION, or FORCE_DIAGNOSTIC. Maintains neutrality — no layout or parameter preference without cited evidence.

### Acquisition Strategy

Each proposal is classified and validated:

| Type | Purpose | Budget guidance |
|------|---------|-----------------|
| `EXPLORE` | Cover untested NC layouts, flow regions, or hypotheses | First 40% of budget |
| `EXPLOIT` | Refine near the best-known solution | Middle 40% of budget |
| `VERIFY` | Perturbation tests, multi-start confirmation, fidelity escalation | Last 20% of budget |

The agent monitors its own convergence — if best-J hasn't improved in 5+ simulations, it shifts strategy. If acquisition balance is off (e.g., 80% EXPLOIT with minimal EXPLORE), the convergence tracker flags it.

### Multi-fidelity ladder

| Level  | nfex | nfet | ncp | Use case |
|--------|------|------|-----|----------|
| Low    | 4-5  | 2    | 1   | Fast screening |
| Medium | 6    | 3    | 2   | Candidate refinement |
| High   | 10   | 5    | 2   | Final validation |

Never skip levels — going low to high directly risks numerical instability.

### Global Optimum Confidence Protocol

IPOPT is a local solver. The framework uses a 6-tier confidence protocol instead of claiming global optimality:

| Tier | Label | Evidence |
|------|-------|----------|
| 0 | NONE | <3 feasible solutions |
| 1 | LOCAL_CANDIDATE | 1+ feasible high-fidelity solution |
| 2 | LOCALLY_STABLE | Best holds across 2+ seeds |
| 3 | COVERAGE_CONFIRMED | All NC layouts screened |
| 4 | PERTURBATION_STABLE | +/-10-15% perturbation all worse |
| 5 | HIGH_CONFIDENCE_LOCAL | Tiers 2-4 satisfied, no improvement in last 5 runs |

### Comparing Agent vs MINLP

Both methods log convergence data to enable fair comparison:

```bash
# Agent-driven (records to convergence_tracker table automatically)
python -m benchmarks.agent_runner --run-name agent_run

# MINLP baseline (records convergence_log in output JSON)
python -m benchmarks.run_stage --stage optimize-layouts --run-name minlp_baseline
```

The comparison metric is **simulations to reach X% of the best-known objective**. Plot `best_feasible_J` vs `sim_number` for both methods on the same axes.

## Knowledge Files (`agents/`)

| File | Format | Purpose |
|------|--------|---------|
| `hypotheses.json` | JSON | Structured hypotheses with `simulation_results[]`; agents consult before proposing and update after each run |
| `failures.json` | JSON | Known failure modes with `occurrences[]`; agents check risk before proposing and log incidents after |
| `SKILLS.md` | Markdown | Fundamental physics only (zone functions, mass balance, isotherm theory) — no optimal-value hints |
| `Objectives.md` | Markdown | Project targets and constraints for the SMB use case |
| `LLM_SOUL.md` | Markdown | Agent operating principles, acquisition strategy protocol, moderation rules, confidence protocol |
| `IPOPT_SOLVER_RESOURCES.md` | Markdown | Solver configuration guide |

The heuristics files (`hypotheses.json`, `failures.json`) accumulate knowledge over time. The agent reads them before every proposal and updates them after every simulation — building a growing "sense" for what works, what fails, and what remains untested.

## SMB Use Case: Flow Mass Balance

```
F1 = Ffeed + Fraf      (feed zone)
F1 = Fdes  + Fex       (desorbent zone)
```

`Fraf` is derived and never optimized independently. Violating this by >1% causes solver errors or unphysical results.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{agent_nlp_optimizer_2025,
  title  = {Agent-Driven NLP Optimizer},
  author = {Tran, Q. and collaborators},
  year   = {2025},
  url    = {https://github.com/your-org/Agent-Driven-NLP-Optimizer}
}
```
