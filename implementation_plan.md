# Repository Reorganization – AutoResearch-SMB

## Goal

Reorganize the AutoResearch-SMB repository into a clean, professional folder structure. Move all agent knowledge MD files into a dedicated `agents/` directory and update all references across Python code, SLURM scripts, and documentation.

## Current Structure (flat & cluttered)

```
AutoResearch-SMB/
├── .gitignore
├── README.md                              # ← keep at root
├── BENCHMARK_FAIRNESS.md                  # agent doc (scattered)
├── FAILURES.md                            # agent doc (scattered)
├── HYPOTHESES.md                          # agent doc (scattered)
├── IPOPT_SOLVER_RESOURCES.md              # agent doc (scattered)
├── LLM_SOUL.md                            # agent doc (scattered)
├── Objectives.md                          # agent doc (scattered)
├── Problem_definition.md                  # agent doc (scattered)
├── SKILLS.md                              # agent doc (scattered)
├── local_command.md                       # SLURM usage notes (scattered)
├── pace_graph_orchestrator_dev.slurm      # SLURM scripts (scattered)
├── pace_smb_comparable_3runs.slurm
├── pace_smb_minlp_cpu_24h.slurm
├── pace_smb_single_scientist_24h.slurm
├── pace_smb_stage_runner.slurm
├── pace_smb_two_scientists_24h.slurm
├── pace_smb_two_scientists_qwen.slurm
├── SembaSMB/
│   ├── src/                               # core library buried in sub-sub-dir
│   ├── requirement.txt
│   └── requirements-optional.txt
├── benchmarks/
│   ├── agent_runner.py
│   └── run_stage.py
├── scripts/                               # utilities (OK)
├── artifacts/                             # gitignored runtime outputs (OK)
└── _sembasmb_git_backup/                  # stale git backup
```

## Proposed Structure

```
AutoResearch-SMB/
├── .gitignore                              # updated
├── README.md                               # updated paths
├── pyproject.toml                          # NEW - proper packaging
│
├── agents/                                 # NEW - all agent knowledge docs
│   ├── Objectives.md
│   ├── LLM_SOUL.md
│   ├── IPOPT_SOLVER_RESOURCES.md
│   ├── SKILLS.md
│   ├── HYPOTHESES.md
│   ├── FAILURES.md
│   ├── BENCHMARK_FAIRNESS.md
│   └── Problem_definition.md
│
├── src/                                    # MOVED from SembaSMB/src/ → top-level
│   └── sembasmb/                           # renamed to proper Python package name
│       ├── __init__.py
│       ├── config.py                       # renamed from smb_config.py
│       ├── model.py                        # renamed from smb_model.py
│       ├── discretization.py               # renamed from smb_discretization.py
│       ├── isotherm.py                     # renamed from smb_isotherm.py
│       ├── optimization.py                 # renamed from smb_optimization.py
│       ├── solver.py                       # renamed from smb_solver.py
│       ├── metrics.py                      # renamed from smb_metrics.py
│       └── plotting.py                     # renamed from smb_plotting.py
│
├── benchmarks/                             # cleaned up
│   ├── __init__.py
│   ├── agent_runner.py                     # updated imports & paths
│   └── run_stage.py                        # updated imports & paths
│
├── scripts/                                # stays the same
│   ├── install_ipopt_coinbrew.sh
│   ├── plot_smb_3d_tradeoff.py
│   ├── plot_smb_benchmark_results.py
│   ├── plot_smb_pr_productivity.py
│   ├── plot_smb_pr_productivity_minlp.py
│   ├── summarize_all_nc_results.sh
│   └── sync_pace_results.ps1
│
├── slurm/                                  # NEW - SLURM job scripts collected
│   ├── local_command.md
│   ├── pace_graph_orchestrator_dev.slurm
│   ├── pace_smb_comparable_3runs.slurm
│   ├── pace_smb_minlp_cpu_24h.slurm
│   ├── pace_smb_single_scientist_24h.slurm
│   ├── pace_smb_stage_runner.slurm
│   ├── pace_smb_two_scientists_24h.slurm
│   └── pace_smb_two_scientists_qwen.slurm
│
├── tests/                                  # NEW - empty test scaffold
│   └── __init__.py
│
└── requirements.txt                        # MOVED from SembaSMB/requirement.txt
```

> [!IMPORTANT]
> The `_sembasmb_git_backup/` directory is a stale bare git backup. It will be deleted.
> The old `SembaSMB/` folder will be fully replaced by `src/sembasmb/`.

## Proposed Changes

### Agent Knowledge Directory

#### [NEW] `agents/` directory
Move these 8 MD files from root into `agents/`:
- [Objectives.md](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/Objectives.md), [LLM_SOUL.md](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/LLM_SOUL.md), [IPOPT_SOLVER_RESOURCES.md](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/IPOPT_SOLVER_RESOURCES.md), [SKILLS.md](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/SKILLS.md), [HYPOTHESES.md](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/HYPOTHESES.md), [FAILURES.md](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/FAILURES.md), [BENCHMARK_FAIRNESS.md](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/BENCHMARK_FAIRNESS.md), [Problem_definition.md](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/Problem_definition.md)

---

### Core Library Restructure

#### [MODIFY] `src/sembasmb/` (moved from `SembaSMB/src/`)
- Create `src/sembasmb/` as a proper Python package
- Rename files: drop `smb_` prefix (redundant within a `sembasmb` package)
  - `smb_config.py` → `config.py`
  - `smb_model.py` → `model.py`
  - `smb_discretization.py` → `discretization.py`
  - `smb_isotherm.py` → `isotherm.py`
  - `smb_optimization.py` → `optimization.py`
  - `smb_solver.py` → `solver.py`
  - `smb_metrics.py` → `metrics.py`
  - `smb_plotting.py` → `plotting.py`
- Update `__init__.py` imports to match new names

---

### Benchmark Module Updates

#### [MODIFY] [agent_runner.py](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/benchmarks/agent_runner.py)
- Update default paths for `--objectives-file`, `--llm-soul-file`, `--ipopt-resource-file` from `REPO_ROOT / "Objectives.md"` → `REPO_ROOT / "agents" / "Objectives.md"`, etc.

#### [MODIFY] [run_stage.py](file:///c:/Users/quyen/OneDrive%20-%20Georgia%20Institute%20of%20Technology/Aim%203%206ZSMB/Production/Batch%206%20Jan%2026/4ZSMB%20Simulation/Program/Program_V2/AutoResearch-SMB/benchmarks/run_stage.py)
- Update `sys.path.insert` and `from src import ...` to use the new `src/sembasmb` package path

---

### SLURM Scripts

#### [MODIFY] All `.slurm` scripts (moved to `slurm/`)
- Update paths referencing root-level MD files → `agents/` directory
- Update any references to `SembaSMB/` → `src/sembasmb/`

---

### Requirements & Packaging

#### [NEW] [pyproject.toml](file:///pyproject.toml)
- Add a proper `pyproject.toml` with package metadata, dependencies, and entry points
- This replaces `SembaSMB/requirement.txt` and `SembaSMB/requirements-optional.txt`

#### [NEW] [requirements.txt](file:///requirements.txt)
- Move content from `SembaSMB/requirement.txt` to root `requirements.txt`

---

### Cleanup

#### [DELETE] `_sembasmb_git_backup/` — stale git backup
#### [DELETE] `SembaSMB/` — replaced by `src/sembasmb/`
#### [DELETE] `__pycache__/` directories in `benchmarks/` and `scripts/`

---

### Scaffolding

#### [NEW] `tests/__init__.py` — empty test directory scaffold

---

## Verification Plan

### Manual Verification
Since this is a structural reorganization (file moves + path updates) with no existing test suite, verification will be manual:

1. **Import check**: After reorganization, run `python -c "from src.sembasmb import SMBConfig, build_model, solve_model, add_optimization"` from the repo root to verify core imports work
2. **Agent runner help**: Run `python -m benchmarks.agent_runner --help` to verify the argument parser loads with correct default paths pointing to `agents/`
3. **Stage runner help**: Run `python -m benchmarks.run_stage --help` to verify it loads with the new import paths
4. **File existence check**: Verify all 8 agent MD files exist in `agents/` and are removed from root
5. **SLURM path check**: Grep all `.slurm` files for old paths (`/Objectives.md`, `/LLM_SOUL.md`, `/IPOPT_SOLVER_RESOURCES.md`) — should find zero hits at old paths
6. **User visual inspection**: The user should review the final directory tree to confirm the structure matches expectations
