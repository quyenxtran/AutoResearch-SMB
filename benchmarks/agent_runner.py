#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import textwrap
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib import error, request

from . import run_stage as rs
from . import agent_db as split_db
from . import agent_evidence as split_evidence
from . import agent_llm_client as split_llm
from . import agent_policy as split_policy
from . import agent_results as split_results
from . import agent_scientists as split_scientists


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-scientist SMB agent benchmark.")
    parser.add_argument("--run-name", default=os.environ.get("SMB_EXPERIMENT_NAME", "qwen_smb_two_scientists"))
    parser.add_argument("--artifact-dir", default=str(REPO_ROOT / "artifacts" / "agent_runs"))
    parser.add_argument("--conversation-log", default=os.environ.get("SMB_CONVERSATION_LOG", ""))
    parser.add_argument("--conversation-stream-log", default=os.environ.get("SMB_CONVERSATION_STREAM_LOG", ""))
    parser.add_argument("--live-results-log", default=os.environ.get("SMB_LIVE_RESULTS_LOG", ""))
    parser.add_argument(
        "--conversation-log-mode",
        default=os.environ.get("SMB_CONVERSATION_LOG_MODE", "compact"),
        choices=["compact", "full"],
    )
    parser.add_argument(
        "--conversation-response-max-chars",
        type=int,
        default=int(os.environ.get("SMB_CONVERSATION_RESPONSE_MAX_CHARS", "1200")),
    )
    parser.add_argument("--sqlite-db", default=os.environ.get("SMB_SQLITE_DB", str(REPO_ROOT / "artifacts" / "agent_runs" / "smb_agent_context.sqlite")))
    parser.add_argument("--research-md", default=os.environ.get("SMB_RESEARCH_MD", str(REPO_ROOT / "research.md")))
    parser.add_argument("--research-tail-chars", type=int, default=int(os.environ.get("SMB_RESEARCH_TAIL_CHARS", "1200")))
    parser.add_argument("--reset-research-section", action="store_true", default=os.environ.get("SMB_RESEARCH_RESET_SECTION", "0") == "1")
    parser.add_argument("--nc-library", default=os.environ.get("SMB_NC_LIBRARY", "1,2,3,2;2,2,2,2;1,3,2,2"))
    parser.add_argument("--seed-library", default=os.environ.get("SMB_SEED_LIBRARY", "notebook"))
    parser.add_argument("--solver-name", default=os.environ.get("SMB_SOLVER_NAME", "auto"))
    parser.add_argument("--linear-solver", default=os.environ.get("SMB_LINEAR_SOLVER", "mumps"))
    parser.add_argument("--benchmark-hours", type=float, default=float(os.environ.get("SMB_BENCHMARK_HOURS", "12.0")))
    parser.add_argument("--search-hours", type=float, default=float(os.environ.get("SMB_SEARCH_BUDGET_HOURS", "10.0")))
    parser.add_argument("--validation-hours", type=float, default=float(os.environ.get("SMB_VALIDATION_BUDGET_HOURS", "2.0")))
    parser.add_argument(
        "--project-purity-min",
        type=float,
        default=float(
            os.environ.get(
                "SMB_PROJECT_TARGET_PURITY_EX_MEOH_FREE",
                os.environ.get("SMB_TARGET_PURITY_EX_MEOH_FREE", "0.60"),
            )
        ),
    )
    parser.add_argument(
        "--project-recovery-ga-min",
        type=float,
        default=float(
            os.environ.get(
                "SMB_PROJECT_TARGET_RECOVERY_GA",
                os.environ.get("SMB_TARGET_RECOVERY_GA", "0.75"),
            )
        ),
    )
    parser.add_argument(
        "--project-recovery-ma-min",
        type=float,
        default=float(
            os.environ.get(
                "SMB_PROJECT_TARGET_RECOVERY_MA",
                os.environ.get("SMB_TARGET_RECOVERY_MA", "0.75"),
            )
        ),
    )
    parser.add_argument("--max-search-evals", type=int, default=int(os.environ.get("SMB_AGENT_MAX_SEARCH_EVALS", "18")))
    parser.add_argument("--max-validations", type=int, default=int(os.environ.get("SMB_AGENT_MAX_VALIDATIONS", "3")))
    parser.add_argument(
        "--executive-controller-enabled",
        action="store_true",
        default=os.environ.get("SMB_EXECUTIVE_CONTROLLER_ENABLED", "1") == "1",
    )
    parser.add_argument(
        "--executive-trigger-rejects",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_TRIGGER_REJECTS", "2")),
    )
    parser.add_argument(
        "--executive-force-after-rejects",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_FORCE_AFTER_REJECTS", "3")),
    )
    parser.add_argument(
        "--executive-top-k-lock",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_TOP_K_LOCK", "5")),
    )
    parser.add_argument(
        "--executive-arbitration-enabled",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_ARBITRATION_ENABLED", "1")),
    )
    parser.add_argument(
        "--executive-max-revisions",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_MAX_REVISIONS", "1")),
    )
    parser.add_argument(
        "--executive-llm-model",
        default=os.environ.get("SMB_EXECUTIVE_LLM_MODEL", ""),
    )
    parser.add_argument(
        "--systematic-infeasibility-k",
        type=int,
        default=int(os.environ.get("SMB_SYSTEMATIC_INFEASIBILITY_K", "5")),
    )
    parser.add_argument(
        "--bootstrap-reference-runs",
        type=int,
        default=int(os.environ.get("SMB_BOOTSTRAP_REFERENCE_RUNS", "2")),
    )
    parser.add_argument(
        "--random-search-mode",
        type=int,
        default=int(os.environ.get("SMB_RANDOM_SEARCH_MODE", "0")),
    )
    parser.add_argument(
        "--method",
        choices=["agent", "agent_v2", "random"],
        default=env_or_default("SMB_METHOD", "agent"),
    )
    parser.add_argument(
        "--single-scientist-mode",
        type=int,
        default=int(os.environ.get("SMB_SINGLE_SCIENTIST_MODE", "0")),
    )
    parser.add_argument(
        "--min-probe-reference-runs",
        type=int,
        default=int(os.environ.get("SMB_MIN_PROBE_REFERENCE_RUNS", "3")),
    )
    parser.add_argument(
        "--probe-low-fidelity-enabled",
        type=int,
        default=int(os.environ.get("SMB_PROBE_LOW_FIDELITY_ENABLED", "1")),
    )
    parser.add_argument(
        "--probe-nfex",
        type=int,
        default=int(os.environ.get("SMB_PROBE_NFEX", "5")),
    )
    parser.add_argument(
        "--probe-nfet",
        type=int,
        default=int(os.environ.get("SMB_PROBE_NFET", "2")),
    )
    parser.add_argument(
        "--probe-ncp",
        type=int,
        default=int(os.environ.get("SMB_PROBE_NCP", "1")),
    )
    parser.add_argument(
        "--finalization-hard-gate-enabled",
        type=int,
        default=int(os.environ.get("SMB_FINALIZATION_HARD_GATE_ENABLED", "1")),
    )
    parser.add_argument(
        "--finalization-low-fidelity-nfex",
        type=int,
        default=int(os.environ.get("SMB_FINALIZATION_LOW_FIDELITY_NFEX", os.environ.get("SMB_PROBE_NFEX", "5"))),
    )
    parser.add_argument(
        "--finalization-low-fidelity-nfet",
        type=int,
        default=int(os.environ.get("SMB_FINALIZATION_LOW_FIDELITY_NFET", os.environ.get("SMB_PROBE_NFET", "2"))),
    )
    parser.add_argument(
        "--finalization-low-fidelity-ncp",
        type=int,
        default=int(os.environ.get("SMB_FINALIZATION_LOW_FIDELITY_NCP", os.environ.get("SMB_PROBE_NCP", "1"))),
    )
    parser.add_argument("--llm-timeout-seconds", type=float, default=float(os.environ.get("SMB_LLM_TIMEOUT_SECONDS", "300")))
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=int(os.environ.get("SMB_LLM_MAX_TOKENS", "320")),
    )
    parser.add_argument("--llm-max-retries", type=int, default=int(os.environ.get("SMB_LLM_MAX_RETRIES", "1")))
    parser.add_argument(
        "--llm-retry-backoff-seconds",
        type=float,
        default=float(os.environ.get("SMB_LLM_RETRY_BACKOFF_SECONDS", "2.0")),
    )
    parser.add_argument(
        "--objectives-max-chars",
        type=int,
        default=int(os.environ.get("SMB_OBJECTIVES_MAX_CHARS", "3000")),
    )
    parser.add_argument(
        "--llm-soul-max-chars",
        type=int,
        default=int(os.environ.get("SMB_LLM_SOUL_MAX_CHARS", "1800")),
    )
    parser.add_argument(
        "--problem-definition-max-chars",
        type=int,
        default=int(os.environ.get("SMB_PROBLEM_DEFINITION_MAX_CHARS", "1200")),
    )
    parser.add_argument(
        "--skills-max-chars",
        type=int,
        default=int(os.environ.get("SMB_SKILLS_MAX_CHARS", "1200")),
    )
    parser.add_argument(
        "--ipopt-resource-max-chars",
        type=int,
        default=int(os.environ.get("SMB_IPOPT_RESOURCE_MAX_CHARS", "900")),
    )
    parser.add_argument(
        "--skip-initial-plan-llm",
        type=int,
        default=int(os.environ.get("SMB_SKIP_INITIAL_PLAN_LLM", "1")),
    )
    parser.add_argument("--tee", action="store_true", default=os.environ.get("SMB_AGENT_TEE", "0") == "1")
    parser.add_argument("--llm-enabled", action="store_true", default=os.environ.get("SMB_AGENT_LLM_ENABLED", "1") == "1")
    parser.add_argument("--llm-base-url", default=os.environ.get("OLLAMA_BASE_URL", ""))
    parser.add_argument("--llm-model", default=os.environ.get("OLLAMA_MODEL", os.environ.get("SMB_LOCAL_LLM_MODEL", "qwen3.5:9b")))
    parser.add_argument("--llm-api-key", default=os.environ.get("OLLAMA_API_KEY", "ollama"))
    parser.add_argument(
        "--fallback-llm-enabled",
        action="store_true",
        default=os.environ.get("SMB_FALLBACK_LLM_ENABLED", "1") == "1",
    )
    parser.add_argument(
        "--fallback-llm-base-url",
        default=os.environ.get("SMB_FALLBACK_LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
    )
    parser.add_argument("--fallback-llm-model", default=os.environ.get("SMB_FALLBACK_LLM_MODEL", "gpt-5-nano"))
    parser.add_argument(
        "--fallback-llm-api-key",
        default=os.environ.get("SMB_FALLBACK_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )
    parser.add_argument("--objectives-file", default=os.environ.get("SMB_OBJECTIVES_FILE", str(REPO_ROOT / "agents" / "Objectives.md")))
    parser.add_argument("--llm-soul-file", default=os.environ.get("SMB_LLM_SOUL_FILE", str(REPO_ROOT / "agents" / "LLM_SOUL.md")))
    parser.add_argument("--llm-soul-a-file", default=os.environ.get("SMB_LLM_SOUL_A_FILE", str(REPO_ROOT / "agents" / "LLM_SOUL_A.md")))
    parser.add_argument("--llm-soul-b-file", default=os.environ.get("SMB_LLM_SOUL_B_FILE", str(REPO_ROOT / "agents" / "LLM_SOUL_B.md")))
    parser.add_argument("--llm-soul-c-file", default=os.environ.get("SMB_LLM_SOUL_C_FILE", str(REPO_ROOT / "agents" / "LLM_SOUL_C.md")))
    parser.add_argument(
        "--problem-definition-file",
        default=os.environ.get("SMB_PROBLEM_DEFINITION_FILE", str(REPO_ROOT / "agents" / "Problem_definition.md")),
    )
    parser.add_argument(
        "--skills-file",
        default=os.environ.get("SMB_SKILLS_FILE", str(REPO_ROOT / "agents" / "SKILLS.md")),
    )
    parser.add_argument("--ipopt-resource-file", default=os.environ.get("SMB_IPOPT_RESOURCE_FILE", str(REPO_ROOT / "agents" / "IPOPT_SOLVER_RESOURCES.md")))
    return parser


def make_stage_args(stage: str) -> argparse.Namespace:
    return rs.build_parser().parse_args(["--stage", stage])


def env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in {None, ""} else default














































def nc_strategy_board(conn: sqlite3.Connection, nc_library: Sequence[Sequence[int]]) -> str:
    unique_layouts: List[Tuple[int, int, int, int]] = []
    seen: set[Tuple[int, int, int, int]] = set()
    for nc in nc_library:
        key = tuple(int(v) for v in nc)
        if key not in seen:
            seen.add(key)
            unique_layouts.append(key)
    if not unique_layouts:
        return "NC strategy board unavailable: empty nc library."

    stats: Dict[str, Dict[str, float]] = {}
    placeholders = ",".join("?" for _ in unique_layouts)
    rows = conn.execute(
        f"""
        SELECT
            COALESCE(nc, '') AS nc,
            COUNT(*) AS n_total,
            COALESCE(SUM(CASE WHEN status='solver_error' THEN 1 ELSE 0 END), 0) AS n_solver_error,
            COALESCE(SUM(CASE WHEN feasible=1 THEN 1 ELSE 0 END), 0) AS n_feasible,
            MIN(normalized_total_violation) AS best_violation,
            MAX(j_validated) AS best_j_validated,
            MAX(productivity) AS best_productivity,
            AVG(wall_seconds) AS avg_wall_seconds
        FROM simulation_results
        WHERE nc IN ({placeholders})
        GROUP BY nc
        """,
        tuple(nc_key(nc) for nc in unique_layouts),
    ).fetchall()
    for row in rows:
        stats[str(row[0])] = {
            "n_total": float(row[1] or 0.0),
            "n_solver_error": float(row[2] or 0.0),
            "n_feasible": float(row[3] or 0.0),
            "best_violation": float(row[4]) if row[4] is not None else float("inf"),
            "best_j_validated": float(row[5]) if row[5] is not None else float("-inf"),
            "best_productivity": float(row[6]) if row[6] is not None else float("-inf"),
            "avg_wall_seconds": float(row[7]) if row[7] is not None else 0.0,
        }

    ranked: List[Tuple[float, Tuple[int, int, int, int], Dict[str, float]]] = []
    for nc in unique_layouts:
        key = nc_key(nc)
        s = stats.get(
            key,
            {
                "n_total": 0.0,
                "n_solver_error": 0.0,
                "n_feasible": 0.0,
                "best_violation": float("inf"),
                "best_j_validated": float("-inf"),
                "best_productivity": float("-inf"),
                "avg_wall_seconds": 0.0,
            },
        )
        attempts = s["n_total"]
        solver_error_rate = (s["n_solver_error"] / attempts) if attempts > 0 else 0.0
        feasibility_bonus = 120.0 if s["n_feasible"] > 0 else 0.0
        near_feasible_bonus = 0.0
        if s["best_violation"] != float("inf"):
            near_feasible_bonus = max(0.0, 30.0 - 20.0 * s["best_violation"])
        runtime_penalty = min(20.0, s["avg_wall_seconds"] / 600.0) if s["avg_wall_seconds"] > 0 else 0.0
        score = nc_prior_score(nc) + feasibility_bonus + near_feasible_bonus - 20.0 * solver_error_rate - runtime_penalty
        ranked.append((score, nc, s))

    ranked.sort(key=lambda item: item[0], reverse=True)
    lines = [
        f"NC strategy board ({len(unique_layouts)} layouts in current library):",
        "Scientific screening rubric:",
        "- rank by observed evidence: feasibility, J_validated, productivity, violation; no prior layout preference",
        "- penalize repeated solver_error histories and high average walltime",
        "- mild penalty for extreme zone asymmetry (one zone with many more columns than others); no zone count targets assumed",
        "Ranked layouts (score combines structural symmetry penalty + SQLite evidence):",
    ]
    for idx, (score, nc, s) in enumerate(ranked, start=1):
        best_violation = "" if s["best_violation"] == float("inf") else f"{s['best_violation']:.6g}"
        best_j = "" if s["best_j_validated"] == float("-inf") else f"{s['best_j_validated']:.6g}"
        best_prod = "" if s["best_productivity"] == float("-inf") else f"{s['best_productivity']:.6g}"
        lines.append(
            f"- rank={idx:02d} nc={list(nc)} score={score:.2f} attempts={int(s['n_total'])} "
            f"feasible={int(s['n_feasible'])} solver_error={int(s['n_solver_error'])} "
            f"best_violation={best_violation or 'n/a'} best_prod={best_prod or 'n/a'} "
            f"best_J={best_j or 'n/a'} avg_wall_s={s['avg_wall_seconds']:.1f}"
        )
    return "\n".join(lines)












def parse_constraint_names(source: str) -> List[str]:
    names = re.findall(r"m\.(\w+)\s*=\s*Constraint", source)
    return sorted(set(names))


def read_file_or_missing(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def build_codebase_context() -> Dict[str, object]:
    optimization_file = REPO_ROOT / "src" / "sembasmb" / "optimization.py"
    model_file = REPO_ROOT / "src" / "sembasmb" / "model.py"
    metrics_file = REPO_ROOT / "src" / "sembasmb" / "metrics.py"
    run_stage_file = REPO_ROOT / "benchmarks" / "run_stage.py"
    config_file = REPO_ROOT / "src" / "sembasmb" / "config.py"
    solver_file = REPO_ROOT / "src" / "sembasmb" / "solver.py"

    optimization_text = read_file_or_missing(optimization_file)
    model_text = read_file_or_missing(model_file)
    metrics_text = read_file_or_missing(metrics_file)
    run_stage_text = read_file_or_missing(run_stage_file)
    config_text = read_file_or_missing(config_file)
    solver_text = read_file_or_missing(solver_file)

    objective_match = re.search(r"m\.obj\s*=\s*Objective\((.+)\)", optimization_text)
    objective_line = objective_match.group(0).strip() if objective_match else "objective line not detected"
    stage_match = re.search(r'choices=\[(.*?)\]', run_stage_text, flags=re.DOTALL)
    stage_list = []
    if stage_match:
        stage_list = [item.strip().strip("'\"") for item in stage_match.group(1).split(",")]

    flow_linked = "RaffinateConsistency" in optimization_text
    map_solver = "solve_model" in solver_text
    metric_keys = re.findall(r"'([a-zA-Z0-9_]+)'\s*:", metrics_text)
    metric_keys = sorted({key for key in metric_keys if key.startswith(("purity", "recovery", "productivity", "Frec"))})
    config_symbols = re.findall(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", config_text, flags=re.MULTILINE)
    config_symbols = sorted(set(config_symbols))[:12]

    return {
        "optimization_file": str(optimization_file),
        "model_file": str(model_file),
        "metrics_file": str(metrics_file),
        "run_stage_file": str(run_stage_file),
        "constraint_names_optimization": parse_constraint_names(optimization_text),
        "constraint_names_model": parse_constraint_names(model_text),
        "objective_expression": objective_line,
        "flow_consistency_constraint_present": flow_linked,
        "solver_entrypoint_present": map_solver,
        "known_metric_keys": metric_keys,
        "known_config_fields": config_symbols,
        "available_stages": stage_list,
    }


def codebase_context_text(context: Dict[str, object]) -> str:
    lines = [
        f"Optimization file: {context.get('optimization_file')}",
        f"Model file: {context.get('model_file')}",
        f"Metrics file: {context.get('metrics_file')}",
        f"Benchmark stage driver: {context.get('run_stage_file')}",
        f"Optimization constraints: {context.get('constraint_names_optimization')}",
        f"Model constraints: {context.get('constraint_names_model')}",
        f"Objective expression: {context.get('objective_expression')}",
        f"Flow-consistency in optimization: {context.get('flow_consistency_constraint_present')}",
        f"Solver entrypoint present: {context.get('solver_entrypoint_present')}",
        f"Metrics available in code: {context.get('known_metric_keys')}",
        f"Key config fields: {context.get('known_config_fields')}",
        f"Benchmark stages: {context.get('available_stages')}",
    ]
    return "\n".join(lines)


def runtime_compute_context_text() -> str:
    keys = [
        "SMB_COMPUTE_SUMMARY",
        "SMB_CPU_TASKS",
        "SMB_GPU_COUNT",
        "SMB_GPU_MODEL",
        "SMB_MEMORY_GB",
        "SMB_WALLTIME_HOURS",
        "SMB_CURRENT_DEFAULT_SOLVER_STACK",
        "SMB_AVAILABLE_SOLVER_OPTIONS",
        "SMB_AVAILABLE_ACCELERATION_OPTIONS",
        "SMB_RESOURCE_DECISION_POLICY",
        "SMB_VERIFIED_IPOPT_EXECUTABLES",
        "SMB_VERIFIED_IPOPT_LINEAR_SOLVERS",
        "SMB_VERIFIED_IPOPT_PROFILE_MENU",
        "SMB_VERIFIED_IPOPT_BASELINE_FALLBACK_TREE",
        "SMB_VERIFIED_IPOPT_SCREENING_FALLBACK_TREE",
        "SMB_VERIFIED_IPOPT_HARD_PROBLEM_FALLBACK_TREE",
        "SMB_VERIFIED_IPOPT_HIGH_PERFORMANCE_FALLBACK_TREE",
    ]
    lines: List[str] = []
    for key in keys:
        value = os.environ.get(key, "")
        if value:
            lines.append(f"{key}={value}")
    if not lines:
        return "No runtime compute metadata found in environment."
    return "\n".join(lines)


def optimization_constraint_context_text(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            f"Flow bounds: F1 in {getattr(args, 'f1_bounds', '<unknown>')}",
            "Flow bounds: "
            + f"Ffeed in {getattr(args, 'ffeed_bounds', '<unknown>')}, "
            + f"Fdes in {getattr(args, 'fdes_bounds', '<unknown>')}, "
            + f"Fex in {getattr(args, 'fex_bounds', '<unknown>')}, "
            + f"Fraf in {getattr(args, 'fraf_bounds', '<unknown>')}",
            f"tstep bounds: {getattr(args, 'tstep_bounds', '<unknown>')}",
            f"max pump flow ml/min: {getattr(args, 'max_pump_flow', '<unknown>')}",
            f"F1 max flow cap ml/min: {getattr(args, 'f1_max_flow', '<unknown>')}",
            f"exploratory purity_ex_meoh_free minimum: {getattr(args, 'purity_min', '<unknown>')}",
            f"exploratory recovery_ex_GA minimum: {getattr(args, 'recovery_ga_min', '<unknown>')}",
            f"exploratory recovery_ex_MA minimum: {getattr(args, 'recovery_ma_min', '<unknown>')}",
            f"project purity_ex_meoh_free objective minimum: {getattr(args, 'project_purity_min', '<unknown>')}",
            f"project recovery_ex_GA objective minimum: {getattr(args, 'project_recovery_ga_min', '<unknown>')}",
            f"project recovery_ex_MA objective minimum: {getattr(args, 'project_recovery_ma_min', '<unknown>')}",
            f"raffinate MeOH max wt: {getattr(args, 'meoh_max_raff_wt', '<unknown>')}",
            f"extract Water max wt: {getattr(args, 'water_max_ex_wt', '<unknown>')}",
            f"zone1-entry Water max wt: {getattr(args, 'water_max_zone1_entry_wt', '<unknown>')}",
        ]
    )




def start_research_log(
    path: Path,
    args: argparse.Namespace,
    code_context_text_block: str,
    compute_context_text_block: str,
    constraint_context_text_block: str,
    initial_plan: Dict[str, object],
    sqlite_excerpt: str,
    nc_strategy_excerpt: str,
    layout_trends: str,
) -> None:
    if not path.exists():
        append_research(
            path,
            "# SMB Two-Scientist Research Log\n\n"
            "This file captures planning, priorities, findings, and proposed simulation updates.\n",
        )
    section = [
        f"\n## Run: {args.run_name}\n",
        f"- started_utc: {utc_now_text()}",
        f"- benchmark_hours: {args.benchmark_hours}",
        f"- search_hours: {args.search_hours}",
        f"- validation_hours: {args.validation_hours}",
        f"- min_probe_reference_runs: {getattr(args, 'min_probe_reference_runs', '')}",
        f"- probe_low_fidelity_enabled: {bool(int(getattr(args, 'probe_low_fidelity_enabled', 0)))}",
        f"- probe_fidelity: nfex={getattr(args, 'probe_nfex', '')}, nfet={getattr(args, 'probe_nfet', '')}, ncp={getattr(args, 'probe_ncp', '')}",
        f"- finalization_hard_gate_enabled: {bool(int(getattr(args, 'finalization_hard_gate_enabled', 0)))}",
        f"- finalization_low_fidelity_requirements: nfex<={getattr(args, 'finalization_low_fidelity_nfex', '')}, nfet<={getattr(args, 'finalization_low_fidelity_nfet', '')}, ncp<={getattr(args, 'finalization_low_fidelity_ncp', '')}",
        f"- ipopt_defaults: max_iter={int(env_or_default('SMB_IPOPT_MAX_ITER', '1000'))}, tol={float(env_or_default('SMB_IPOPT_TOL', '1e-5'))}, acceptable_tol={float(env_or_default('SMB_IPOPT_ACCEPTABLE_TOL', '1e-4'))}",
        f"- solver_name: {args.solver_name}",
        f"- linear_solver: {args.linear_solver}",
        f"- nc_library: {args.nc_library}",
        f"- seed_library: {args.seed_library}",
        f"- exploratory_targets: purity={getattr(args, 'purity_min', '')}, recovery_ga={getattr(args, 'recovery_ga_min', '')}, recovery_ma={getattr(args, 'recovery_ma_min', '')}",
        f"- project_objective_targets: purity={getattr(args, 'project_purity_min', '')}, recovery_ga={getattr(args, 'project_recovery_ga_min', '')}, recovery_ma={getattr(args, 'project_recovery_ma_min', '')}",
        f"- executive_controller: enabled={bool(getattr(args, 'executive_controller_enabled', False))}, trigger_rejects={getattr(args, 'executive_trigger_rejects', '')}, force_after={getattr(args, 'executive_force_after_rejects', '')}, top_k_lock={getattr(args, 'executive_top_k_lock', '')}",
        f"- single_scientist_mode: {bool(int(getattr(args, 'single_scientist_mode', 0)))}",
        f"- sqlite_db: {args.sqlite_db}",
        "",
        "### Codebase Context Snapshot",
        "```text",
        code_context_text_block,
        "```",
        "",
        "### Runtime Compute Snapshot",
        "```text",
        compute_context_text_block,
        "```",
        "",
        "### Simulation Constraint Snapshot",
        "```text",
        constraint_context_text_block,
        "```",
        "",
        "### Existing History Snapshot",
        "```text",
        sqlite_excerpt,
        "```",
        "",
        "### NC Strategy Board",
        "```text",
        nc_strategy_excerpt,
        "```",
        "",
        "### Initial Priorities",
    ]
    for item in normalize_text_list(initial_plan.get("priorities"), max_items=12):
        section.append(f"- {item}")
    section.append("")
    section.append("### Initial Proposed Simulations")
    for item in normalize_text_list(initial_plan.get("proposed_simulations"), max_items=12):
        section.append(f"- {item}")
    section.append("")
    section.append("### NC Screening Strategy")
    for item in normalize_text_list(initial_plan.get("nc_screening_strategy"), max_items=12):
        section.append(f"- {item}")
    section.append("")
    section.append("### Initial Risks")
    for item in normalize_text_list(initial_plan.get("risks"), max_items=12):
        section.append(f"- {item}")
    section.append("")
    section.append("### Insights and Trends (Rolling)")
    section.append(layout_trends)
    section.append("")
    append_research(path, "\n".join(section) + "\n")














def inferred_violation_from_metrics(metrics: Dict[str, object]) -> Optional[float]:
    purity = as_float(metrics.get("purity_ex_meoh_free"))
    rga = as_float(metrics.get("recovery_ex_GA"))
    rma = as_float(metrics.get("recovery_ex_MA"))
    if purity is None and rga is None and rma is None:
        return None

    purity_min = float(env_or_default("SMB_TARGET_PURITY_EX_MEOH_FREE", "0.60"))
    rga_min = float(env_or_default("SMB_TARGET_RECOVERY_GA", "0.75"))
    rma_min = float(env_or_default("SMB_TARGET_RECOVERY_MA", "0.75"))

    norm = 0.0
    if purity is not None:
        norm += max(0.0, purity_min - purity) / max(purity_min, 1e-12)
    if rga is not None:
        norm += max(0.0, rga_min - rga) / max(rga_min, 1e-12)
    if rma is not None:
        norm += max(0.0, rma_min - rma) / max(rma_min, 1e-12)
    return norm




def bootstrap_reference_select(tasks: List[Dict[str, object]], tried: set[Tuple[Tuple[int, ...], str]]) -> int:
    for idx, task in enumerate(tasks):
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key in tried:
            continue
        if is_reference_seed_name(task.get("seed_name")):
            return idx
    return deterministic_select(tasks, tried)




def low_fidelity_limits(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "nfex": max(1, int(getattr(args, "finalization_low_fidelity_nfex", getattr(args, "probe_nfex", 5)))),
        "nfet": max(1, int(getattr(args, "finalization_low_fidelity_nfet", getattr(args, "probe_nfet", 2)))),
        "ncp": max(1, int(getattr(args, "finalization_low_fidelity_ncp", getattr(args, "probe_ncp", 1)))),
    }




def is_low_fidelity_result(result: Dict[str, object], args: argparse.Namespace) -> bool:
    triplet = fidelity_triplet(result)
    if triplet is None:
        return False
    limits = low_fidelity_limits(args)
    return triplet[0] <= limits["nfex"] and triplet[1] <= limits["nfet"] and triplet[2] <= limits["ncp"]




def has_low_fidelity_reference_evidence_for_nc(
    args: argparse.Namespace,
    results: List[Dict[str, object]],
    nc: Tuple[int, ...],
) -> bool:
    for item in results:
        if tuple(item.get("nc", [])) != nc:
            continue
        if not is_reference_seed_name(item.get("seed_name")):
            continue
        if not is_low_fidelity_result(item, args):
            continue
        if has_metric_evidence(item):
            return True
    return False


def has_low_fidelity_optimization_evidence_for_nc(
    args: argparse.Namespace,
    results: List[Dict[str, object]],
    nc: Tuple[int, ...],
) -> bool:
    for item in results:
        if tuple(item.get("nc", [])) != nc:
            continue
        if is_reference_seed_name(item.get("seed_name")):
            continue
        if not is_low_fidelity_result(item, args):
            continue
        if has_metric_evidence(item):
            return True
    return False




def first_untried_reference_index(
    tasks: List[Dict[str, object]],
    tried: set[Tuple[Tuple[int, ...], str]],
) -> Optional[int]:
    for idx in ranked_reference_indices(tasks):
        task = tasks[idx]
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key not in tried:
            return idx
    return None






def executive_forced_index(
    tasks: List[Dict[str, object]],
    tried: set[Tuple[Tuple[int, ...], str]],
    top_k_lock: int,
) -> Tuple[int, str]:
    ref_idx = ranked_reference_indices(tasks)
    top_ref = ref_idx[: max(1, top_k_lock)]
    for idx in top_ref:
        task = tasks[idx]
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key not in tried:
            return idx, "first untried reference task inside executive top-k lock."
    for idx in ref_idx:
        task = tasks[idx]
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key not in tried:
            return idx, "first untried reference task after top-k lock exhausted."
    idx = deterministic_select(tasks, tried)
    return idx, "fallback to first untried task because all reference tasks are exhausted."




def execute_search_task(
    args: argparse.Namespace,
    task: Dict[str, object],
    *,
    fidelity_override: Optional[Dict[str, int]] = None,
    flow_override: Optional[Dict[str, float]] = None,
    execution_note: str = "",
) -> Dict[str, object]:
    base = configure_stage_args(make_stage_args("optimize-layouts"), args)
    tstep_bounds = rs.parse_bounds(base.tstep_bounds)
    ffeed_bounds = rs.parse_bounds(base.ffeed_bounds)
    fdes_bounds = rs.parse_bounds(base.fdes_bounds)
    fex_bounds = rs.parse_bounds(base.fex_bounds)
    fraf_bounds = rs.parse_bounds(base.fraf_bounds)
    f1_bounds = rs.parse_bounds(base.f1_bounds)
    candidate_args = rs.apply_seed_to_args(
        base,
        task["seed"],
        tstep_bounds=tstep_bounds,
        ffeed_bounds=ffeed_bounds,
        fdes_bounds=fdes_bounds,
        fex_bounds=fex_bounds,
        fraf_bounds=fraf_bounds,
        f1_bounds=f1_bounds,
    )
    if isinstance(fidelity_override, dict):
        candidate_args.nfex = max(1, int(fidelity_override.get("nfex", candidate_args.nfex)))
        candidate_args.nfet = max(1, int(fidelity_override.get("nfet", candidate_args.nfet)))
        candidate_args.ncp = max(1, int(fidelity_override.get("ncp", candidate_args.ncp)))
    if isinstance(flow_override, dict):
        flow_map = {
            "Ffeed": "ffeed",
            "F1": "f1",
            "Fdes": "fdes",
            "Fex": "fex",
            "Fraf": "fraf",
            "tstep": "tstep",
        }
        for key, attr in flow_map.items():
            value = as_float(flow_override.get(key))
            if value is None:
                continue
            setattr(candidate_args, attr, float(value))
    candidate_args.run_name = f"{args.run_name}_search_nc_{'-'.join(str(v) for v in task['nc'])}_{candidate_args.seed_name}"
    result = rs.evaluate_optimized_layout(candidate_args, tuple(task["nc"]))
    if isinstance(fidelity_override, dict) or isinstance(flow_override, dict) or execution_note:
        result["execution_policy"] = {
            "fidelity_override": fidelity_override or {},
            "flow_override": flow_override or {},
            "note": execution_note,
        }
    return result


def effective_search_task(
    args: argparse.Namespace,
    task: Dict[str, object],
    *,
    flow_override: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    base = configure_stage_args(make_stage_args("optimize-layouts"), args)
    tstep_bounds = rs.parse_bounds(base.tstep_bounds)
    ffeed_bounds = rs.parse_bounds(base.ffeed_bounds)
    fdes_bounds = rs.parse_bounds(base.fdes_bounds)
    fex_bounds = rs.parse_bounds(base.fex_bounds)
    fraf_bounds = rs.parse_bounds(base.fraf_bounds)
    f1_bounds = rs.parse_bounds(base.f1_bounds)
    candidate_args = rs.apply_seed_to_args(
        base,
        task["seed"],
        tstep_bounds=tstep_bounds,
        ffeed_bounds=ffeed_bounds,
        fdes_bounds=fdes_bounds,
        fex_bounds=fex_bounds,
        fraf_bounds=fraf_bounds,
        f1_bounds=f1_bounds,
    )
    if isinstance(flow_override, dict):
        flow_map = {
            "Ffeed": "ffeed",
            "F1": "f1",
            "Fdes": "fdes",
            "Fex": "fex",
            "Fraf": "fraf",
            "tstep": "tstep",
        }
        for key, attr in flow_map.items():
            value = as_float(flow_override.get(key))
            if value is None:
                continue
            setattr(candidate_args, attr, float(value))
    return {
        "nc": list(task["nc"]),
        "seed_name": str(candidate_args.seed_name),
        "flow": {
            "Ffeed": float(candidate_args.ffeed),
            "F1": float(candidate_args.f1),
            "Fdes": float(candidate_args.fdes),
            "Fex": float(candidate_args.fex),
            "Fraf": float(candidate_args.fraf),
            "tstep": float(candidate_args.tstep),
        },
    }


def build_validation_candidates(
    args: argparse.Namespace,
    results: List[Dict[str, object]],
    max_items: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    ranked = rank_any_results(results)
    selected: List[Dict[str, object]] = []
    gate_notes: List[str] = []
    gate_seen: set[str] = set()
    seen: set[Tuple[Tuple[int, ...], float, float, float, float, float]] = set()
    hard_gate_enabled = bool(int(getattr(args, "finalization_hard_gate_enabled", 1)))
    for item in ranked:
        flow = effective_flow(item)
        if flow is None:
            continue
        if hard_gate_enabled:
            nc = tuple(item.get("nc", []))
            if is_reference_seed_name(item.get("seed_name")):
                note = f"Skipped {item.get('run_name')}: finalization gate requires non-reference optimization candidate."
                if note not in gate_seen:
                    gate_seen.add(note)
                    gate_notes.append(note)
                continue
            if not is_low_fidelity_result(item, args):
                note = f"Skipped {item.get('run_name')}: candidate is not low-fidelity pre-final run."
                if note not in gate_seen:
                    gate_seen.add(note)
                    gate_notes.append(note)
                continue
            if str(item.get("status", "")).lower() != "ok":
                note = f"Skipped {item.get('run_name')}: candidate status is '{item.get('status')}', requires status 'ok' for finalization."
                if note not in gate_seen:
                    gate_seen.add(note)
                    gate_notes.append(note)
                continue
            if not has_low_fidelity_reference_evidence_for_nc(args, results, nc):
                note = (
                    f"Skipped {item.get('run_name')}: missing low-fidelity reference evidence for nc={list(nc)} "
                    "required before final optimization."
                )
                if note not in gate_seen:
                    gate_seen.add(note)
                    gate_notes.append(note)
                continue
            if not has_low_fidelity_optimization_evidence_for_nc(args, results, nc):
                note = (
                    f"Skipped {item.get('run_name')}: missing low-fidelity optimization evidence for nc={list(nc)} "
                    "required before final optimization."
                )
                if note not in gate_seen:
                    gate_seen.add(note)
                    gate_notes.append(note)
                continue
        key = (
            tuple(item["nc"]),
            flow["Ffeed"],
            flow["F1"],
            flow["Fdes"],
            flow["Fex"],
            flow["tstep"],
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected, gate_notes


def execute_validation(args: argparse.Namespace, result: Dict[str, object], ordinal: int) -> Dict[str, object]:
    flow = effective_flow(result)
    if flow is None:
        raise RuntimeError("Validation candidate does not expose a usable flow.")
    base = configure_stage_args(make_stage_args("reference-eval"), args)
    base.run_name = f"{args.run_name}_validate_{ordinal:02d}"
    base.nc = ",".join(str(v) for v in result["nc"])
    base.nfex = 10
    base.nfet = 5
    base.ncp = 2
    base.purity_min = float(args.project_purity_min)
    base.recovery_ga_min = float(args.project_recovery_ga_min)
    base.recovery_ma_min = float(args.project_recovery_ma_min)
    base.ffeed = flow["Ffeed"]
    base.f1 = flow["F1"]
    base.fdes = flow["Fdes"]
    base.fex = flow["Fex"]
    base.fraf = flow["Fraf"]
    base.tstep = flow["tstep"]
    return rs.evaluate_candidate(base, tuple(result["nc"]))


def artifact_path(args: argparse.Namespace) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return Path(args.artifact_dir) / f"agent-runner.{job_id}.{args.run_name}.json"


def conversation_log_path(args: argparse.Namespace) -> Path:
    if args.conversation_log:
        return Path(args.conversation_log)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return Path(args.artifact_dir) / f"agent-runner.{job_id}.{args.run_name}.conversations.json"


def conversation_stream_log_path(args: argparse.Namespace, conversation_path: Path) -> Path:
    if args.conversation_stream_log:
        return Path(args.conversation_stream_log)
    if conversation_path.suffix:
        return conversation_path.with_suffix(".jsonl")
    return Path(str(conversation_path) + ".jsonl")


def live_results_log_path(args: argparse.Namespace, artifact_path_value: Path) -> Path:
    if args.live_results_log:
        return Path(args.live_results_log)
    if artifact_path_value.suffix:
        return artifact_path_value.with_suffix(".live_results.jsonl")
    return Path(str(artifact_path_value) + ".live_results.jsonl")


def initialize_conversation_stream(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def initialize_live_results_stream(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_live_results_event(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("timestamp_utc", utc_now_text())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_artifact(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def write_conversation_log(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def progress_log(message: str) -> None:
    """Emit compact progress markers into Slurm stdout for bottleneck diagnosis."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


# Runtime delegation bridge: keep legacy implementations import-compatible while
# switching the live orchestration path to the extracted split modules.
utc_now_text = split_llm.utc_now_text
required_keys_missing = split_llm.required_keys_missing
OpenAICompatClient = split_llm.OpenAICompatClient
request_json_with_single_repair = split_llm.request_json_with_single_repair

as_float = split_results.as_float
layout_text = split_results.layout_text
extract_metrics_with_validity = split_results.extract_metrics_with_validity
safe_result_metric = split_results.safe_result_metric
effective_flow = split_results.effective_flow
stream_components_from_outlets = split_results.stream_components_from_outlets
composition_metrics_from_result = split_results.composition_metrics_from_result
composition_metrics_from_raw_json = split_results.composition_metrics_from_raw_json
linear_slope = split_results.linear_slope
effective_violation = split_results.effective_violation
search_score = split_results.search_score
summarize_result = split_results.summarize_result
recent_two_run_review_context = split_results.recent_two_run_review_context
rank_any_results = split_results.rank_any_results
deterministic_select = split_results.deterministic_select
is_reference_seed_name = split_results.is_reference_seed_name
fidelity_triplet = split_results.fidelity_triplet
has_metric_evidence = split_results.has_metric_evidence
reference_probe_runs_completed = split_results.reference_probe_runs_completed
ranked_reference_indices = split_results.ranked_reference_indices
has_any_feasible = split_results.has_any_feasible

normalize_text_list = split_evidence.normalize_text_list
bottleneck_label = split_evidence.bottleneck_label
compact_result_record = split_evidence.compact_result_record
build_evidence_pack = split_evidence.build_evidence_pack
contains_run_reference = split_evidence.contains_run_reference
normalize_evidence_refs = split_evidence.normalize_evidence_refs
build_evidence_fallback_items = split_evidence.build_evidence_fallback_items
coerce_evidence_list = split_evidence.coerce_evidence_list
coerce_grounded_evidence_refs = split_evidence.coerce_grounded_evidence_refs
evidence_refs_are_grounded = split_evidence.evidence_refs_are_grounded
text_mentions_prior_runs = split_evidence.text_mentions_prior_runs
text_mentions_metric_signals = split_evidence.text_mentions_metric_signals
text_mentions_numeric_values = split_evidence.text_mentions_numeric_values
text_mentions_delta_metric_signals = split_evidence.text_mentions_delta_metric_signals
count_flow_signal_mentions = split_evidence.count_flow_signal_mentions
text_mentions_delta_flow_signals = split_evidence.text_mentions_delta_flow_signals
text_mentions_run_name_signals = split_evidence.text_mentions_run_name_signals
text_mentions_required_labels = split_evidence.text_mentions_required_labels
text_mentions_flow_signals = split_evidence.text_mentions_flow_signals
text_mentions_topology_signals = split_evidence.text_mentions_topology_signals
text_mentions_physics_signals = split_evidence.text_mentions_physics_signals
extract_nc_mentions = split_evidence.extract_nc_mentions
review_references_candidate_nc = split_evidence.review_references_candidate_nc
read_doc_excerpt = split_evidence.read_doc_excerpt
compact_prompt_block = split_evidence.compact_prompt_block
budget_evidence_pack_json = split_evidence.budget_evidence_pack_json
markdown_focused_excerpt = split_evidence.markdown_focused_excerpt
build_heuristics_context = split_evidence.build_heuristics_context
hypothesis_matcher = split_evidence.hypothesis_matcher
failure_recovery_context = split_evidence.failure_recovery_context
apply_flow_adjustments = split_evidence.apply_flow_adjustments
build_task_from_counterproposal = split_evidence.build_task_from_counterproposal

open_sqlite_db = split_db.open_sqlite_db
persist_result_to_sqlite = split_db.persist_result_to_sqlite
record_convergence_snapshot = split_db.record_convergence_snapshot
sqlite_convergence_context = split_db.sqlite_convergence_context
sqlite_targeted_query = split_db.sqlite_targeted_query
sqlite_history_context = split_db.sqlite_history_context
sqlite_record_count = split_db.sqlite_record_count
sqlite_layout_trend_table = split_db.sqlite_layout_trend_table
read_research_tail = split_db.read_research_tail
append_research = split_db.append_research
reset_research_run_section = split_db.reset_research_run_section
append_iteration_research = split_db.append_iteration_research
append_result_research = split_db.append_result_research
append_final_research = split_db.append_final_research
merge_priority_board = split_db.merge_priority_board

env_or_default = split_policy.env_or_default
nc_key = split_policy.nc_key
nc_prior_score = split_policy.nc_prior_score
sqlite_total_records_from_excerpt = split_policy.sqlite_total_records_from_excerpt
configure_stage_args = split_policy.configure_stage_args
build_search_tasks = split_policy.build_search_tasks
apply_probe_reference_gate = split_policy.apply_probe_reference_gate
probe_reference_runs_required = split_policy.probe_reference_runs_required
search_execution_policy = split_policy.search_execution_policy
deterministic_review = split_policy.deterministic_review
single_scientist_policy_review = split_policy.single_scientist_policy_review
executive_controller_decide = split_policy.executive_controller_decide
physics_informed_select = split_policy.physics_informed_select
check_systematic_infeasibility = split_policy.check_systematic_infeasibility

default_initial_priority_plan = split_scientists.default_initial_priority_plan
initial_priority_plan = split_scientists.initial_priority_plan
scientist_a_pick = split_scientists.scientist_a_pick
scientist_b_review = split_scientists.scientist_b_review
scientist_c_arbitrate = split_scientists.scientist_c_arbitrate


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    artifact = artifact_path(args)
    conversation_artifact = conversation_log_path(args)
    conversation_stream_artifact = conversation_stream_log_path(args, conversation_artifact)
    live_results_artifact = live_results_log_path(args, artifact)
    research_path = Path(args.research_md)
    objectives_excerpt = markdown_focused_excerpt(
        args.objectives_file,
        heading_keywords=(
            "mission",
            "optimization goal",
            "components and basis",
            "desorbent composition",
            "smb configuration",
            "hard operating constraints",
            "required workflow",
            "mandatory nc-coverage",
        ),
        max_chars=args.objectives_max_chars,
        max_lines=150,
    )
    soul_excerpt = markdown_focused_excerpt(
        args.llm_soul_a_file if Path(args.llm_soul_a_file).exists() else args.llm_soul_file,
        heading_keywords=(
            "role",
            "core principle",
            "acquisition strategy protocol",
            "mandatory deep review",
            "compute and fidelity policy",
            "when to stop",
            "reporting style",
        ),
        max_chars=args.llm_soul_max_chars,
        max_lines=130,
    )
    soul_b_excerpt = markdown_focused_excerpt(
        args.llm_soul_b_file if Path(args.llm_soul_b_file).exists() else args.llm_soul_file,
        heading_keywords=(
            "role",
            "core principle",
            "mandatory deep review",
            "what scientist_b must check",
            "counterproposal standard",
            "evaluation skills",
            "reporting style",
        ),
        max_chars=args.llm_soul_max_chars,
        max_lines=130,
    )
    soul_c_excerpt = markdown_focused_excerpt(
        args.llm_soul_c_file if Path(args.llm_soul_c_file).exists() else args.llm_soul_file,
        heading_keywords=(
            "role",
            "core principle",
            "scientist_executive moderation protocol",
            "decision skills",
            "reporting style",
        ),
        max_chars=args.llm_soul_max_chars,
        max_lines=130,
    )
    problem_definition_excerpt = markdown_focused_excerpt(
        args.problem_definition_file,
        heading_keywords=(
            "core question",
            "optimization problem",
            "what kind of optimization problem",
            "fixed-budget rule",
            "five-hour benchmark protocol",
            "recommended success criteria",
        ),
        max_chars=args.problem_definition_max_chars,
        max_lines=110,
    )
    skills_excerpt = markdown_focused_excerpt(
        args.skills_file,
        heading_keywords=(
            "zone functions",
            "flow mass balance",
            "switching time",
            "multi-fidelity",
            "solver status",
            "purity and recovery",
            "physical hardware constraints",
        ),
        max_chars=args.skills_max_chars,
        max_lines=100,
    )
    ipopt_excerpt = read_doc_excerpt(args.ipopt_resource_file, max_chars=args.ipopt_resource_max_chars)
    client = OpenAICompatClient(
        args.llm_base_url,
        args.llm_model,
        args.llm_enabled,
        api_key=args.llm_api_key,
        fallback_enabled=args.fallback_llm_enabled,
        fallback_base_url=args.fallback_llm_base_url,
        fallback_model=args.fallback_llm_model,
        fallback_api_key=args.fallback_llm_api_key,
        conversation_stream_path=conversation_stream_artifact,
        timeout_seconds=args.llm_timeout_seconds,
        max_tokens=args.llm_max_tokens,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff_seconds,
        conversation_log_mode=args.conversation_log_mode,
        conversation_response_max_chars=args.conversation_response_max_chars,
    )
    sqlite_conn = open_sqlite_db(args.sqlite_db)
    optimize_stage_args = configure_stage_args(make_stage_args("optimize-layouts"), args)
    code_context = build_codebase_context()
    code_context_excerpt = codebase_context_text(code_context)
    compute_context_excerpt = runtime_compute_context_text()
    constraint_context_excerpt = optimization_constraint_context_text(optimize_stage_args)

    search_results: List[Dict[str, object]] = []
    validation_results: List[Dict[str, object]] = []
    scientist_a_log: List[Dict[str, object]] = []
    scientist_b_log: List[Dict[str, object]] = []
    executive_log: List[Dict[str, object]] = []
    ledger: List[Dict[str, object]] = []
    tried: set[Tuple[Tuple[int, ...], str]] = set()
    heuristics_excerpt = build_heuristics_context(max_chars=900)
    sim_counter = 0  # global simulation counter for convergence tracking
    method_explicit = any(arg == "--method" or arg.startswith("--method=") for arg in sys.argv[1:])
    runtime_method = str(getattr(args, "method", "agent")).strip().lower()
    if bool(int(getattr(args, "random_search_mode", 0))) and not method_explicit:
        runtime_method = "random"
    if runtime_method not in {"agent", "agent_v2", "random"}:
        runtime_method = "agent"
    force_diagnostic_next_iteration = False
    force_diagnostic_reason = ""
    revision_iterations: List[int] = []

    def emit_live_event(
        event: str,
        *,
        iteration: Optional[int] = None,
        role: str = "",
        decision: str = "",
        reason: str = "",
        task: Optional[Dict[str, object]] = None,
        result: Optional[Dict[str, object]] = None,
        note: Optional[Dict[str, object]] = None,
        acquisition_type: str = "",
        arbitration_outcome: str = "",
    ) -> None:
        selected_task: Dict[str, object] = {}
        if isinstance(task, dict):
            selected_task = {
                "nc": list(task.get("nc", [])) if isinstance(task.get("nc"), list) else [],
                "seed_name": str(task.get("seed_name", "")),
            }
            if isinstance(task.get("flow_override"), dict):
                selected_task["flow_override"] = task.get("flow_override")
        record: Dict[str, object] = {
            "event": event,
            "job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "run_name": args.run_name,
            "iteration": int(iteration) if isinstance(iteration, int) else None,
            "role": role,
            "decision": decision or (str(note.get("decision", "")) if isinstance(note, dict) else ""),
            "reason": reason or (str(note.get("reason", "")) if isinstance(note, dict) else ""),
            "selected_task": selected_task,
            "acquisition_type": acquisition_type or (str(note.get("acquisition_type", "")) if isinstance(note, dict) else ""),
            "arbitration_outcome": arbitration_outcome,
        }
        if isinstance(result, dict):
            metrics, _ = extract_metrics_with_validity(result)
            record.update(
                {
                    "result_run_name": str(result.get("run_name", "")),
                    "solver_status": str(result.get("status", "")),
                    "feasible": bool(result.get("feasible", False)),
                    "j_validated": as_float(result.get("J_validated")),
                    "productivity": as_float(metrics.get("productivity_ex_ga_ma")),
                    "purity": as_float(metrics.get("purity_ex_meoh_free")),
                    "recovery_ga": as_float(metrics.get("recovery_ex_GA")),
                    "recovery_ma": as_float(metrics.get("recovery_ex_MA")),
                    "normalized_total_violation": effective_violation(result),
                }
            )
        if isinstance(note, dict):
            evidence_refs = normalize_evidence_refs(note.get("evidence_refs"), max_items=6)
            if evidence_refs:
                record["evidence_refs"] = evidence_refs
        append_live_results_event(live_results_artifact, record)

    try:
        progress_log("AGENT: init start")
        initialize_conversation_stream(conversation_stream_artifact)
        initialize_live_results_stream(live_results_artifact)
        if args.reset_research_section:
            reset_research_run_section(research_path, args.run_name)
        nc_library_values = [list(nc) for nc in rs.parse_nc_library(args.nc_library)]
        initial_sqlite_excerpt = sqlite_history_context(sqlite_conn)
        initial_nc_strategy_excerpt = nc_strategy_board(sqlite_conn, nc_library_values)
        if int(getattr(args, "skip_initial_plan_llm", 1)) == 1:
            initial_plan = default_initial_priority_plan(args)
            initial_plan["reason"] = "LLM initial planning skipped (SMB_SKIP_INITIAL_PLAN_LLM=1) for faster startup."
            initial_plan["mode"] = "deterministic_compact_startup"
            progress_log("AGENT: initial priority plan skipped (deterministic)")
        else:
            progress_log("AGENT: initial priority plan LLM start")
            initial_plan = initial_priority_plan(
                client,
                args,
                objectives_excerpt,
                soul_excerpt,
                problem_definition_excerpt,
                skills_excerpt,
                code_context_excerpt,
                initial_sqlite_excerpt,
                initial_nc_strategy_excerpt,
                compute_context_excerpt,
                constraint_context_excerpt,
            )
            progress_log("AGENT: initial priority plan LLM done")
        current_priorities = normalize_text_list(initial_plan.get("priorities"), max_items=16)
        start_research_log(
            research_path,
            args,
            code_context_excerpt,
            compute_context_excerpt,
            constraint_context_excerpt,
            initial_plan,
            initial_sqlite_excerpt,
            initial_nc_strategy_excerpt,
            sqlite_layout_trend_table(sqlite_conn),
        )

        progress_log("AGENT: solver-check start")
        solver_check = rs.run_solver_check(configure_stage_args(make_stage_args("solver-check"), args))
        progress_log("AGENT: solver-check done")
        search_tasks = build_search_tasks(args)
        search_hours_used = 0.0
        validation_hours_used = 0.0
        search_iteration = 0
        consecutive_rejects = 0

        while (
            len(tried) < len(search_tasks)
            and len(search_results) < args.max_search_evals
            and search_hours_used < args.search_hours
        ):
            search_iteration += 1
            progress_log(
                "AGENT: iteration "
                + str(search_iteration)
                + f" start (tried={len(tried)}/{len(search_tasks)}, search_hours={search_hours_used:.3f})"
            )
            emit_live_event(
                "iteration_start",
                iteration=search_iteration,
                role="main_loop",
                decision="ITERATION_START",
                reason=f"tried={len(tried)}/{len(search_tasks)} search_hours_used={search_hours_used:.4f}",
            )
            sqlite_excerpt = sqlite_history_context(sqlite_conn)
            nc_strategy_excerpt = nc_strategy_board(sqlite_conn, nc_library_values)
            research_excerpt = read_research_tail(research_path, args.research_tail_chars)
            convergence_excerpt = sqlite_convergence_context(sqlite_conn, args.run_name)
            best_so_far = rank_any_results(search_results)[0] if search_results else None
            if force_diagnostic_next_iteration:
                diag_idx, diag_note = physics_informed_select(
                    search_tasks,
                    tried,
                    search_results,
                    best_result=best_so_far,
                    reason=force_diagnostic_reason or "Systematic infeasibility requested a diagnostic task.",
                )
                task = search_tasks[diag_idx]
                task_key = (tuple(task["nc"]), str(task["seed_name"]))
                if task_key in tried:
                    break
                effective_task = effective_search_task(args, task, flow_override=task.get("flow_override") if isinstance(task.get("flow_override"), dict) else None)
                a_note = {
                    "mode": "diagnostic_forced",
                    "decision": "FORCE_DIAGNOSTIC",
                    "reason": diag_note.get("reason", force_diagnostic_reason or "Forced diagnostic task selection."),
                    "priority_updates": [
                        "Systematic infeasibility triggered an immediate diagnostic execution."
                    ],
                    "acquisition_type": "FORCE_DIAGNOSTIC",
                }
                b_note = {
                    "mode": "diagnostic_forced",
                    "decision": "approve",
                    "reason": "Diagnostic override bypassed Scientist_B review.",
                    "priority_updates": [
                        "Diagnostic override bypassed Scientist_B so the next iteration can probe failure structure."
                    ],
                    "acquisition_type": "FORCE_DIAGNOSTIC",
                    "risk_flags": [force_diagnostic_reason] if force_diagnostic_reason else [],
                }
                scientist_a_log.append(
                    {
                        "task": task,
                        "proposed_task": task,
                        "effective_task_after_policy": effective_task,
                        "decision": a_note,
                    }
                )
                scientist_b_log.append(
                    {
                        "task": task,
                        "reviewed_task": task,
                        "effective_task_after_policy": effective_task,
                        "decision": b_note,
                    }
                )
                executive_note = {
                    "decision": "FORCE_DIAGNOSTIC",
                    "reason": diag_note.get("reason", force_diagnostic_reason or "Forced diagnostic execution."),
                    "priority_updates": [
                        "Systematic infeasibility trigger forced a diagnostic run next iteration."
                    ],
                    "acquisition_type": "FORCE_DIAGNOSTIC",
                    "diagnostic_focus": diag_note.get("reason", force_diagnostic_reason or ""),
                }
                executive_log.append({"task": task, "decision": executive_note})
                emit_live_event("scientist_a_decision", iteration=search_iteration, role="scientist_a_pick", task=task, note=a_note)
                emit_live_event("scientist_b_decision", iteration=search_iteration, role="scientist_b_review", task=task, note=b_note)
                emit_live_event("executive_decision", iteration=search_iteration, role="scientist_c_arbitrate", task=task, note=executive_note, arbitration_outcome="FORCE_DIAGNOSTIC")
                current_priorities = merge_priority_board(current_priorities, a_note, b_note, executive_note)
                append_iteration_research(
                    research_path,
                    search_iteration,
                    task,
                    a_note,
                    b_note,
                    scientist_a_proposed_task=task,
                    effective_task_after_policy=effective_task,
                    scientist_b_reviewed_task=task,
                    executive_note=executive_note,
                )
                tried.add(task_key)
                execution_policy = search_execution_policy(args, search_tasks, search_results, task)
                if execution_policy.get("reason"):
                    append_research(
                        research_path,
                        f"- execution_policy: {execution_policy.get('reason')}\n",
                    )
                progress_log(f"AGENT: execute search diagnostic_forced start task={task}")
                result = execute_search_task(
                    args,
                    task,
                    fidelity_override=(
                        execution_policy.get("fidelity_override")
                        if isinstance(execution_policy.get("fidelity_override"), dict)
                        else None
                    ),
                    flow_override=effective_task.get("flow") if isinstance(effective_task.get("flow"), dict) else None,
                    execution_note=str(execution_policy.get("reason", "")),
                )
                progress_log(
                    "AGENT: execute search diagnostic_forced done "
                    + f"run={result.get('run_name')} status={result.get('status')} "
                    + f"wall_s={float(((result.get('timing') or {}).get('wall_seconds') or 0.0)):.1f}"
                )
                result["diagnostic_forced"] = True
                result["diagnostic_forced_reason"] = force_diagnostic_reason
                emit_live_event(
                    "simulation_complete",
                    iteration=search_iteration,
                    role="simulation",
                    task=task,
                    result=result,
                    acquisition_type="FORCE_DIAGNOSTIC",
                    arbitration_outcome="FORCE_DIAGNOSTIC",
                )
                search_results.append(result)
                persist_result_to_sqlite(sqlite_conn, args.run_name, "search", result)
                sim_counter += 1
                record_convergence_snapshot(
                    sqlite_conn,
                    args.run_name,
                    runtime_method,
                    sim_counter,
                    result,
                    search_hours_used * 3600.0,
                    search_hours_used,
                    acquisition_type="FORCE_DIAGNOSTIC",
                )
                heuristics_excerpt = build_heuristics_context(max_chars=900)
                append_result_research(research_path, result, "search")
                append_research(
                    research_path,
                    "\n#### Insights and Trends Update\n"
                    f"- timestamp_utc: {utc_now_text()}\n"
                    + sqlite_layout_trend_table(sqlite_conn)
                    + "\n",
                )
                ledger.append(
                    {
                        "phase": "search_diagnostic_forced",
                        "run_name": result.get("run_name"),
                        "status": result.get("status"),
                        "timing": result.get("timing"),
                    }
                )
                timing = result.get("timing") or {}
                search_hours_used += float(timing.get("wall_seconds", 0.0)) / 3600.0
                infeasibility_state = check_systematic_infeasibility(
                    search_results,
                    int(getattr(args, "systematic_infeasibility_k", 5)),
                )
                if bool(infeasibility_state.get("triggered")):
                    force_diagnostic_next_iteration = True
                    force_diagnostic_reason = str(infeasibility_state.get("reason", ""))
                else:
                    force_diagnostic_next_iteration = False
                    force_diagnostic_reason = ""
                continue
            if runtime_method == "random":
                remaining_indices = [
                    i for i, candidate in enumerate(search_tasks) if (tuple(candidate["nc"]), str(candidate["seed_name"])) not in tried
                ]
                if not remaining_indices:
                    break
                idx = random.choice(remaining_indices)
                task = search_tasks[idx]
                task_key = (tuple(task["nc"]), str(task["seed_name"]))
                effective_task = effective_search_task(args, task, flow_override=task.get("flow_override") if isinstance(task.get("flow_override"), dict) else None)
                a_note = {
                    "mode": "random_search",
                    "decision": "random_select",
                    "reason": "Random search mode bypassed Scientist_A.",
                    "priority_updates": [
                        "Random search mode active: uniform untried task selected without LLM review."
                    ],
                    "acquisition_type": "RANDOM_SEARCH",
                }
                b_note = {
                    "mode": "random_search",
                    "decision": "approve",
                    "reason": "Random search mode bypassed Scientist_B.",
                    "priority_updates": [
                        "Random search mode active: Scientist_B review was bypassed."
                    ],
                    "acquisition_type": "RANDOM_SEARCH",
                }
                executive_note = {
                    "decision": "not_needed",
                    "reason": "Random search mode bypassed executive arbitration.",
                    "priority_updates": [],
                    "acquisition_type": "RANDOM_SEARCH",
                }
                scientist_a_log.append(
                    {
                        "task": task,
                        "proposed_task": task,
                        "effective_task_after_policy": effective_task,
                        "decision": a_note,
                    }
                )
                scientist_b_log.append(
                    {
                        "task": task,
                        "reviewed_task": task,
                        "effective_task_after_policy": effective_task,
                        "decision": b_note,
                    }
                )
                executive_log.append({"task": task, "decision": executive_note})
                emit_live_event("scientist_a_decision", iteration=search_iteration, role="scientist_a_pick", task=task, note=a_note)
                emit_live_event("scientist_b_decision", iteration=search_iteration, role="scientist_b_review", task=task, note=b_note)
                emit_live_event("executive_decision", iteration=search_iteration, role="executive_controller", task=task, note=executive_note, arbitration_outcome="RANDOM_SEARCH")
                current_priorities = merge_priority_board(current_priorities, a_note, b_note, executive_note)
                append_iteration_research(
                    research_path,
                    search_iteration,
                    task,
                    a_note,
                    b_note,
                    scientist_a_proposed_task=task,
                    effective_task_after_policy=effective_task,
                    scientist_b_reviewed_task=task,
                    executive_note=executive_note,
                )
                tried.add(task_key)
                execution_policy = search_execution_policy(args, search_tasks, search_results, task)
                if execution_policy.get("reason"):
                    append_research(
                        research_path,
                        f"- execution_policy: {execution_policy.get('reason')}\n",
                    )
                progress_log(f"AGENT: execute search_random start task={task}")
                result = execute_search_task(
                    args,
                    task,
                    fidelity_override=(
                        execution_policy.get("fidelity_override")
                        if isinstance(execution_policy.get("fidelity_override"), dict)
                        else None
                    ),
                    flow_override=effective_task.get("flow") if isinstance(effective_task.get("flow"), dict) else None,
                    execution_note=str(execution_policy.get("reason", "")),
                )
                progress_log(
                    "AGENT: execute search_random done "
                    + f"run={result.get('run_name')} status={result.get('status')} "
                    + f"wall_s={float(((result.get('timing') or {}).get('wall_seconds') or 0.0)):.1f}"
                )
                result["random_search_mode"] = True
                emit_live_event(
                    "simulation_complete",
                    iteration=search_iteration,
                    role="simulation",
                    task=task,
                    result=result,
                    acquisition_type="RANDOM_SEARCH",
                    arbitration_outcome="RANDOM_SEARCH",
                )
                search_results.append(result)
                persist_result_to_sqlite(sqlite_conn, args.run_name, "search", result)
                sim_counter += 1
                record_convergence_snapshot(
                    sqlite_conn,
                    args.run_name,
                    runtime_method,
                    sim_counter,
                    result,
                    search_hours_used * 3600.0,
                    search_hours_used,
                    acquisition_type="RANDOM_SEARCH",
                )
                heuristics_excerpt = build_heuristics_context(max_chars=900)
                append_result_research(research_path, result, "search")
                append_research(
                    research_path,
                    "\n#### Insights and Trends Update\n"
                    f"- timestamp_utc: {utc_now_text()}\n"
                    + sqlite_layout_trend_table(sqlite_conn)
                    + "\n",
                )
                ledger.append(
                    {
                        "phase": "search_random",
                        "run_name": result.get("run_name"),
                        "status": result.get("status"),
                        "timing": result.get("timing"),
                    }
                )
                timing = result.get("timing") or {}
                search_hours_used += float(timing.get("wall_seconds", 0.0)) / 3600.0
                infeasibility_state = check_systematic_infeasibility(
                    search_results,
                    int(getattr(args, "systematic_infeasibility_k", 5)),
                )
                if bool(infeasibility_state.get("triggered")):
                    force_diagnostic_next_iteration = True
                    force_diagnostic_reason = str(infeasibility_state.get("reason", ""))
                continue
            bootstrap_target = max(0, int(getattr(args, "bootstrap_reference_runs", 0)))
            bootstrap_mode = bootstrap_target > 0 and len(search_results) < bootstrap_target
            if bootstrap_mode:
                idx = bootstrap_reference_select(search_tasks, tried)
                a_note = {
                    "mode": "bootstrap_reference",
                    "decision": "bootstrap_reference",
                    "reason": (
                        "Bootstrap reference run executed to seed evidence before strict A/B/C gating "
                        f"({len(search_results) + 1}/{bootstrap_target})."
                    ),
                    "acquisition_type": "BOOTSTRAP_REFERENCE",
                    "bootstrap_reference": True,
                    "priority_updates": [
                        "Bootstrap mode active: collect baseline run evidence before relying on LLM proposal quality."
                    ],
                    "evidence": [
                        "No/limited prior evidence available; run deterministic reference probe first."
                    ],
                    "comparison_to_previous": [
                        "Bootstrap reference run to establish initial baseline for data-grounded A/B/C comparisons."
                    ],
                    "evidence_refs": [],
                }
            else:
                try:
                    progress_log(f"AGENT: Scientist_A pick start (iter={search_iteration})")
                    idx, a_note = scientist_a_pick(
                        client,
                        search_tasks,
                        search_results,
                        tried,
                        args,
                        objectives_excerpt,
                        soul_excerpt,
                        code_context_excerpt,
                        compute_context_excerpt,
                        constraint_context_excerpt,
                        nc_strategy_excerpt,
                        research_excerpt,
                        current_priorities,
                        sqlite_excerpt,
                        search_hours_used,
                        search_iteration,
                        heuristics_context=heuristics_excerpt,
                        convergence_context=convergence_excerpt,
                    )
                    progress_log(f"AGENT: Scientist_A pick done (iter={search_iteration})")
                except Exception as exc:
                    idx = deterministic_select(search_tasks, tried)
                    a_note = {
                        "mode": "deterministic_error",
                        "reason": f"Scientist_A exception fallback: {type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "priority_updates": [
                            "Scientist_A call failed; fallback to deterministic first-untried candidate to keep run alive."
                        ],
                        "evidence": [
                            "LLM Scientist_A call raised an exception.",
                            "Using deterministic fallback to avoid hard stop.",
                        ],
                        "comparison_to_previous": [
                            "No model comparison available due to Scientist_A exception."
                        ],
                        "nc_competitor_comparison": [
                            "No model NC comparison available due to Scientist_A exception."
                        ],
                        "failure_criteria": [
                            "Reject if solver status is solver_error/other with no usable primal values.",
                            "Reject if normalized_total_violation does not improve against current best evidence.",
                        ],
                    }
            a_proposed_idx = idx
            a_proposed_task = dict(search_tasks[a_proposed_idx])
            idx, probe_gate_note = apply_probe_reference_gate(
                args,
                search_tasks,
                tried,
                search_results,
                idx,
            )
            task = search_tasks[idx]
            task_key = (tuple(task["nc"]), str(task["seed_name"]))
            if task_key in tried:
                break
            if probe_gate_note is not None:
                a_note = dict(a_note)
                a_note["probe_gate"] = probe_gate_note
                priority_updates = normalize_text_list(a_note.get("priority_updates"), max_items=10)
                gate_reason = str(probe_gate_note.get("reason", "")).strip()
                if gate_reason:
                    priority_updates.append(gate_reason)
                    a_note["priority_updates"] = normalize_text_list(priority_updates, max_items=10)

            effective_task = effective_search_task(args, task)
            scientist_a_log.append(
                {
                    "task": task,
                    "proposed_task": a_proposed_task,
                    "effective_task_after_policy": effective_task,
                    "decision": a_note,
                }
            )
            emit_live_event("scientist_a_decision", iteration=search_iteration, role="scientist_a_pick", task=task, note=a_note)
            a_reason_text = str(a_note.get("reason", "")).strip()
            a_low_quality = bool(a_note.get("low_quality_recovery"))
            if (not a_low_quality) and str(a_note.get("mode", "")).strip().lower() == "deterministic":
                if a_reason_text.lower().startswith("rejected llm proposal"):
                    a_low_quality = True
            if a_low_quality:
                force_diagnostic_next_iteration = True
                force_diagnostic_reason = a_reason_text or "Scientist_A output failed quality gate."
                emit_live_event(
                    "quality_recovery",
                    iteration=search_iteration,
                    role="scientist_a_pick",
                    task=task,
                    note=a_note,
                    acquisition_type="LOW_QUALITY_RECOVERY",
                    arbitration_outcome="FORCE_DIAGNOSTIC",
                )
                append_research(
                    research_path,
                    f"- low_quality_recovery: scientist_a iteration={search_iteration} reason={force_diagnostic_reason}\n",
                )
                continue
            best_so_far = rank_any_results(search_results)[0] if search_results else None
            if bootstrap_mode:
                b_note = {
                    "mode": "bootstrap_reference",
                    "decision": "approve",
                    "reason": "Bootstrap reference run bypassed Scientist_B review to avoid startup deadlock.",
                    "acquisition_type": "BOOTSTRAP_REFERENCE",
                    "priority_updates": [
                        "Bootstrap mode active: bypass Scientist_B for initial deterministic evidence collection."
                    ],
                    "risk_flags": [],
                }
            elif bool(int(getattr(args, "single_scientist_mode", 0))):
                b_note = single_scientist_policy_review(task, best_so_far)
            else:
                try:
                    progress_log(f"AGENT: Scientist_B review start (iter={search_iteration})")
                    b_note = scientist_b_review(
                        client,
                        task,
                        effective_task,
                        best_so_far,
                        search_results,
                        args,
                        code_context_excerpt,
                        compute_context_excerpt,
                        constraint_context_excerpt,
                        nc_strategy_excerpt,
                        research_excerpt,
                        current_priorities,
                        sqlite_excerpt,
                        search_iteration,
                        soul_excerpt=soul_b_excerpt,
                        heuristics_context=heuristics_excerpt,
                    )
                    progress_log(f"AGENT: Scientist_B review done (iter={search_iteration})")
                except Exception as exc:
                    b_note = {
                        "mode": "deterministic_error",
                        "reason": f"Scientist_B exception fallback: {type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        **deterministic_review(task, best_so_far),
                    }
            scientist_b_log.append(
                {
                    "task": task,
                    "reviewed_task": task,
                    "effective_task_after_policy": effective_task,
                    "decision": b_note,
                }
            )
            emit_live_event("scientist_b_decision", iteration=search_iteration, role="scientist_b_review", task=task, note=b_note)
            b_reason_text = str(b_note.get("reason", "")).strip()
            b_low_quality = bool(b_note.get("low_quality_recovery"))
            if (not b_low_quality) and str(b_note.get("mode", "")).strip().lower() == "llm":
                if b_reason_text.lower().startswith("rejected:"):
                    b_low_quality = True
            if b_low_quality:
                force_diagnostic_next_iteration = True
                force_diagnostic_reason = b_reason_text or "Scientist_B output failed quality gate."
                emit_live_event(
                    "quality_recovery",
                    iteration=search_iteration,
                    role="scientist_b_review",
                    task=task,
                    note=b_note,
                    acquisition_type="LOW_QUALITY_RECOVERY",
                    arbitration_outcome="FORCE_DIAGNOSTIC",
                )
                append_research(
                    research_path,
                    f"- low_quality_recovery: scientist_b iteration={search_iteration} reason={force_diagnostic_reason}\n",
                )
                continue
            b_approved = str(b_note.get("decision", "approve")).lower() == "approve"
            if b_approved:
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1

            counterproposal = b_note.get("counterproposal_run") if isinstance(b_note.get("counterproposal_run"), dict) else None
            counterproposal_valid = False
            if isinstance(counterproposal, dict):
                cp_nc = counterproposal.get("nc")
                cp_flow = counterproposal.get("flow_adjustments")
                cp_effect = counterproposal.get("expected_metric_effect")
                cp_physics = str(counterproposal.get("physics_justification", "")).strip()
                valid_nc = isinstance(cp_nc, list) and len(cp_nc) == 4 and all(isinstance(v, (int, float)) for v in cp_nc)
                valid_flow = isinstance(cp_flow, dict) and any(isinstance(cp_flow.get(key), (int, float)) for key in ("Ffeed", "F1", "Fdes", "Fex", "Fraf", "tstep"))
                valid_effect = isinstance(cp_effect, dict) and any(isinstance(cp_effect.get(key), (int, float)) for key in ("delta_productivity", "delta_purity", "delta_recovery_ga", "delta_recovery_ma", "delta_violation"))
                counterproposal_valid = valid_nc and valid_flow and valid_effect and bool(cp_physics)

            if (not b_approved) and bool(int(getattr(args, "executive_arbitration_enabled", 1))) and counterproposal_valid:
                c_note = scientist_c_arbitrate(
                    client,
                    task,
                    effective_task,
                    a_note,
                    b_note,
                    search_results,
                    args,
                    heuristics_excerpt,
                    current_priorities,
                    sqlite_excerpt,
                    search_iteration,
                    revision_count_recent=sum(
                        1 for item in revision_iterations if item > search_iteration - 3
                    ),
                    force_diagnostic_reason=force_diagnostic_reason if force_diagnostic_next_iteration else "",
                    soul_excerpt=soul_c_excerpt,
                    nc_strategy_excerpt=nc_strategy_excerpt,
                )
                c_decision = str(c_note.get("decision", "")).strip().upper()
                revision_recent_count = sum(1 for item in revision_iterations if item > search_iteration - 3)
                if c_decision == "RETURN_FOR_REVISION":
                    max_revisions = max(0, int(getattr(args, "executive_max_revisions", 1)))
                    if revision_recent_count >= max_revisions:
                        c_note = dict(c_note)
                        c_note["decision"] = "FORCE_DIAGNOSTIC"
                        c_note["reason"] = (
                            "Revision request rate-limited by the rolling 3-iteration window; forcing diagnostic instead."
                        )
                        c_note["acquisition_type"] = "FORCE_DIAGNOSTIC"
                        c_decision = "FORCE_DIAGNOSTIC"
                    else:
                        revision_iterations.append(search_iteration)

                if c_decision == "RETURN_FOR_REVISION":
                    c_note = dict(c_note)
                    c_note["acquisition_type"] = "WASTED_REJECT"
                    c_note["priority_updates"] = normalize_text_list(c_note.get("priority_updates"), max_items=8)
                    c_note["priority_updates"].append("Revision returned without execution; count as a wasted reject.")
                    executive_log.append({"task": task, "decision": c_note})
                    emit_live_event(
                        "executive_decision",
                        iteration=search_iteration,
                        role="scientist_c_arbitrate",
                        task=task,
                        note=c_note,
                        arbitration_outcome="RETURN_FOR_REVISION",
                    )
                    current_priorities = merge_priority_board(current_priorities, a_note, b_note, c_note)
                    append_iteration_research(
                        research_path,
                        search_iteration,
                        task,
                        a_note,
                        b_note,
                        scientist_a_proposed_task=a_proposed_task,
                        effective_task_after_policy=effective_task,
                        scientist_b_reviewed_task=task,
                        executive_note=c_note,
                    )
                    tried.add(task_key)
                    append_research(
                        research_path,
                        f"- search_result_run: WASTED_REJECT at {utc_now_text()} for task={task}\n",
                    )
                    force_diagnostic_next_iteration = False
                    force_diagnostic_reason = ""
                    continue

                if c_decision == "FORCE_DIAGNOSTIC":
                    diag_idx, diag_note = physics_informed_select(
                        search_tasks,
                        tried,
                        search_results,
                        best_result=best_so_far,
                        preferred_nc=task.get("nc"),
                        preferred_seed_name=str(task.get("seed_name", "")),
                        reason=str(c_note.get("reason", "")) or force_diagnostic_reason or "Arbitration forced a diagnostic task.",
                    )
                    selected_task = search_tasks[diag_idx]
                    selected_key = (tuple(selected_task["nc"]), str(selected_task["seed_name"]))
                    if selected_key in tried and selected_key != task_key:
                        c_note["reason"] = "Diagnostic fallback landed on a tried task; keeping the original reject outcome."
                        c_note["acquisition_type"] = "WASTED_REJECT"
                        c_note["decision"] = "RETURN_FOR_REVISION"
                        executive_log.append({"task": task, "decision": c_note})
                        emit_live_event(
                            "executive_decision",
                            iteration=search_iteration,
                            role="scientist_c_arbitrate",
                            task=task,
                            note=c_note,
                            arbitration_outcome="RETURN_FOR_REVISION",
                        )
                        current_priorities = merge_priority_board(current_priorities, a_note, b_note, c_note)
                        append_iteration_research(
                            research_path,
                            search_iteration,
                            task,
                            a_note,
                            b_note,
                            scientist_a_proposed_task=a_proposed_task,
                            effective_task_after_policy=effective_task,
                            scientist_b_reviewed_task=task,
                            executive_note=c_note,
                        )
                        tried.add(task_key)
                        append_research(
                            research_path,
                            f"- search_result_run: WASTED_REJECT at {utc_now_text()} for task={task}\n",
                        )
                        continue
                    selected_effective_task = effective_search_task(
                        args,
                        selected_task,
                        flow_override=selected_task.get("flow_override") if isinstance(selected_task.get("flow_override"), dict) else None,
                    )
                    c_note["selected_task"] = selected_task
                    c_note["diagnostic_note"] = diag_note
                    c_note["acquisition_type"] = "FORCE_DIAGNOSTIC"
                elif c_decision == "IMPLEMENT_B_COUNTER":
                    selected_task = build_task_from_counterproposal(task, counterproposal, effective_task=effective_task, mode="counterproposal")
                    selected_key = (tuple(selected_task["nc"]), str(selected_task["seed_name"]))
                    if selected_key in tried and selected_key != task_key:
                        c_note["reason"] = "Chosen counterproposal task was already tried; switching to diagnostic fallback."
                        diag_idx, diag_note = physics_informed_select(
                            search_tasks,
                            tried,
                            search_results,
                            best_result=best_so_far,
                            preferred_nc=selected_task.get("nc"),
                            preferred_seed_name=str(selected_task.get("seed_name", "")),
                            reason=str(c_note.get("reason", "")),
                        )
                        selected_task = search_tasks[diag_idx]
                        selected_key = (tuple(selected_task["nc"]), str(selected_task["seed_name"]))
                        c_note["decision"] = "FORCE_DIAGNOSTIC"
                        c_note["acquisition_type"] = "FORCE_DIAGNOSTIC"
                        c_note["diagnostic_note"] = diag_note
                    selected_effective_task = effective_search_task(
                        args,
                        selected_task,
                        flow_override=selected_task.get("flow_override") if isinstance(selected_task.get("flow_override"), dict) else None,
                    )
                    c_note["selected_task"] = selected_task
                elif c_decision == "IMPLEMENT_HYBRID":
                    selected_task = build_task_from_counterproposal(task, counterproposal, effective_task=effective_task, mode="hybrid")
                    selected_key = (tuple(selected_task["nc"]), str(selected_task["seed_name"]))
                    if selected_key in tried and selected_key != task_key:
                        c_note["reason"] = "Hybrid arbitration target was already tried; switching to diagnostic fallback."
                        diag_idx, diag_note = physics_informed_select(
                            search_tasks,
                            tried,
                            search_results,
                            best_result=best_so_far,
                            preferred_nc=selected_task.get("nc"),
                            preferred_seed_name=str(selected_task.get("seed_name", "")),
                            reason=str(c_note.get("reason", "")),
                        )
                        selected_task = search_tasks[diag_idx]
                        selected_key = (tuple(selected_task["nc"]), str(selected_task["seed_name"]))
                        c_note["decision"] = "FORCE_DIAGNOSTIC"
                        c_note["acquisition_type"] = "FORCE_DIAGNOSTIC"
                        c_note["diagnostic_note"] = diag_note
                    selected_effective_task = effective_search_task(
                        args,
                        selected_task,
                        flow_override=selected_task.get("flow_override") if isinstance(selected_task.get("flow_override"), dict) else None,
                    )
                    c_note["selected_task"] = selected_task
                else:
                    selected_task = task
                    selected_key = task_key
                    selected_effective_task = effective_task
                    c_note["selected_task"] = selected_task

                executive_note = c_note
                executive_log.append({"task": task, "decision": executive_note})
                emit_live_event(
                    "executive_decision",
                    iteration=search_iteration,
                    role="scientist_c_arbitrate",
                    task=selected_task,
                    note=executive_note,
                    arbitration_outcome=str(executive_note.get("decision", "")).strip().upper(),
                )
                current_priorities = merge_priority_board(current_priorities, a_note, b_note, executive_note)
                append_iteration_research(
                    research_path,
                    search_iteration,
                    task,
                    a_note,
                    b_note,
                    scientist_a_proposed_task=a_proposed_task,
                    effective_task_after_policy=effective_task,
                    scientist_b_reviewed_task=task,
                    executive_note=executive_note,
                )
                tried.add(task_key)
                if selected_key != task_key:
                    tried.add(selected_key)
                execution_policy = search_execution_policy(args, search_tasks, search_results, selected_task)
                if execution_policy.get("reason"):
                    append_research(
                        research_path,
                        f"- execution_policy: {execution_policy.get('reason')}\n",
                    )
                progress_log(f"AGENT: execute search arbitration start task={selected_task}")
                result = execute_search_task(
                    args,
                    selected_task,
                    fidelity_override=(
                        execution_policy.get("fidelity_override")
                        if isinstance(execution_policy.get("fidelity_override"), dict)
                        else None
                    ),
                    flow_override=selected_effective_task.get("flow") if isinstance(selected_effective_task.get("flow"), dict) else None,
                    execution_note=str(execution_policy.get("reason", "")),
                )
                progress_log(
                    "AGENT: execute search arbitration done "
                    + f"run={result.get('run_name')} status={result.get('status')} "
                    + f"wall_s={float(((result.get('timing') or {}).get('wall_seconds') or 0.0)):.1f}"
                )
                result["arbitration_decision"] = executive_note.get("decision")
                result["arbitration_note"] = executive_note
                emit_live_event(
                    "simulation_complete",
                    iteration=search_iteration,
                    role="simulation",
                    task=selected_task,
                    result=result,
                    acquisition_type=str(executive_note.get("acquisition_type", "")).strip().upper(),
                    arbitration_outcome=str(executive_note.get("decision", "")).strip().upper(),
                )
                search_results.append(result)
                persist_result_to_sqlite(sqlite_conn, args.run_name, "search", result)
                sim_counter += 1
                acq_type = str(executive_note.get("acquisition_type", "")).strip().upper() or str(a_note.get("acquisition_type", "")).strip().upper()
                record_convergence_snapshot(
                    sqlite_conn,
                    args.run_name,
                    runtime_method,
                    sim_counter,
                    result,
                    search_hours_used * 3600.0,
                    search_hours_used,
                    acquisition_type=acq_type,
                )
                heuristics_excerpt = build_heuristics_context(max_chars=900)
                append_result_research(research_path, result, "search")
                append_research(
                    research_path,
                    "\n#### Insights and Trends Update\n"
                    f"- timestamp_utc: {utc_now_text()}\n"
                    + sqlite_layout_trend_table(sqlite_conn)
                    + "\n",
                )
                ledger.append(
                    {
                        "phase": "search_arbitrated",
                        "run_name": result.get("run_name"),
                        "status": result.get("status"),
                        "timing": result.get("timing"),
                    }
                )
                timing = result.get("timing") or {}
                search_hours_used += float(timing.get("wall_seconds", 0.0)) / 3600.0
                infeasibility_state = check_systematic_infeasibility(
                    search_results,
                    int(getattr(args, "systematic_infeasibility_k", 5)),
                )
                if bool(infeasibility_state.get("triggered")):
                    force_diagnostic_next_iteration = True
                    force_diagnostic_reason = str(infeasibility_state.get("reason", ""))
                else:
                    force_diagnostic_next_iteration = False
                    force_diagnostic_reason = ""
                continue

            executive_note = executive_controller_decide(
                args,
                search_tasks,
                tried,
                idx,
                task,
                b_note,
                search_results,
                consecutive_rejects,
            )
            executive_log.append({"task": task, "decision": executive_note})
            emit_live_event(
                "executive_decision",
                iteration=search_iteration,
                role="executive_controller",
                task=task,
                note=executive_note,
                arbitration_outcome=str(executive_note.get("decision", "")).strip().upper(),
            )
            current_priorities = merge_priority_board(current_priorities, a_note, b_note, executive_note)
            append_iteration_research(
                research_path,
                search_iteration,
                task,
                a_note,
                b_note,
                scientist_a_proposed_task=a_proposed_task,
                effective_task_after_policy=effective_task,
                scientist_b_reviewed_task=task,
                executive_note=executive_note,
            )

            if not b_approved:
                if str(executive_note.get("decision", "")).lower() != "override_execute":
                    tried.add(task_key)
                    emit_live_event(
                        "wasted_reject",
                        iteration=search_iteration,
                        role="scientist_b_review",
                        task=task,
                        note=b_note,
                        acquisition_type="WASTED_REJECT",
                        arbitration_outcome="RESPECT_REJECT",
                    )
                    append_research(
                        research_path,
                        f"- search_result_run: skipped_by_scientist_b at {utc_now_text()} for task={task}\n",
                    )
                    continue

                forced_idx = int(executive_note.get("forced_candidate_index", idx))
                forced_task = search_tasks[forced_idx]
                forced_key = (tuple(forced_task["nc"]), str(forced_task["seed_name"]))
                if forced_key != task_key and forced_key in tried:
                    tried.add(task_key)
                    append_research(
                        research_path,
                        f"- search_result_run: executive_override_skipped_duplicate at {utc_now_text()} for task={forced_task}\n",
                    )
                    continue
                tried.add(task_key)
                tried.add(forced_key)
                append_research(
                    research_path,
                    f"- search_result_run: executive_override_execute at {utc_now_text()} from task={task} to forced_task={forced_task}\n",
                )
                forced_policy = search_execution_policy(args, search_tasks, search_results, forced_task)
                if forced_policy.get("reason"):
                    append_research(
                        research_path,
                        f"- execution_policy: {forced_policy.get('reason')}\n",
                    )
                progress_log(f"AGENT: execute search_executive_forced start task={forced_task}")
                result = execute_search_task(
                    args,
                    forced_task,
                    fidelity_override=(
                        forced_policy.get("fidelity_override")
                        if isinstance(forced_policy.get("fidelity_override"), dict)
                        else None
                    ),
                    execution_note=str(forced_policy.get("reason", "")),
                )
                progress_log(
                    "AGENT: execute search_executive_forced done "
                    + f"run={result.get('run_name')} status={result.get('status')} "
                    + f"wall_s={float(((result.get('timing') or {}).get('wall_seconds') or 0.0)):.1f}"
                )
                result["executive_forced"] = True
                result["executive_forced_from_task"] = task
                emit_live_event(
                    "simulation_complete",
                    iteration=search_iteration,
                    role="simulation",
                    task=forced_task,
                    result=result,
                    acquisition_type="FORCE_DIAGNOSTIC",
                    arbitration_outcome="OVERRIDE_EXECUTE",
                )
                search_results.append(result)
                persist_result_to_sqlite(sqlite_conn, args.run_name, "search", result)
                sim_counter += 1
                record_convergence_snapshot(
                    sqlite_conn, args.run_name, runtime_method, sim_counter, result,
                    search_hours_used * 3600.0, search_hours_used,
                    acquisition_type="FORCE_DIAGNOSTIC",
                )
                append_result_research(research_path, result, "search")
                append_research(
                    research_path,
                    "\n#### Insights and Trends Update\n"
                    f"- timestamp_utc: {utc_now_text()}\n"
                    + sqlite_layout_trend_table(sqlite_conn)
                    + "\n",
                )
                ledger.append(
                    {
                        "phase": "search_executive_forced",
                        "run_name": result.get("run_name"),
                        "status": result.get("status"),
                        "timing": result.get("timing"),
                    }
                )
                timing = result.get("timing") or {}
                search_hours_used += float(timing.get("wall_seconds", 0.0)) / 3600.0
                infeasibility_state = check_systematic_infeasibility(
                    search_results,
                    int(getattr(args, "systematic_infeasibility_k", 5)),
                )
                if bool(infeasibility_state.get("triggered")):
                    force_diagnostic_next_iteration = True
                    force_diagnostic_reason = str(infeasibility_state.get("reason", ""))
                consecutive_rejects = 0
                continue

            tried.add(task_key)
            execution_policy = search_execution_policy(args, search_tasks, search_results, task)
            if execution_policy.get("reason"):
                append_research(
                    research_path,
                    f"- execution_policy: {execution_policy.get('reason')}\n",
                )
            progress_log(f"AGENT: execute search start task={task}")
            result = execute_search_task(
                args,
                task,
                fidelity_override=(
                    execution_policy.get("fidelity_override")
                    if isinstance(execution_policy.get("fidelity_override"), dict)
                    else None
                ),
                execution_note=str(execution_policy.get("reason", "")),
            )
            progress_log(
                "AGENT: execute search done "
                + f"run={result.get('run_name')} status={result.get('status')} "
                + f"wall_s={float(((result.get('timing') or {}).get('wall_seconds') or 0.0)):.1f}"
            )
            emit_live_event(
                "simulation_complete",
                iteration=search_iteration,
                role="simulation",
                task=task,
                result=result,
                acquisition_type=str(a_note.get("acquisition_type", "")).strip().upper(),
                arbitration_outcome="DIRECT_EXECUTION",
            )
            search_results.append(result)
            persist_result_to_sqlite(sqlite_conn, args.run_name, "search", result)
            sim_counter += 1
            acq_type = str(a_note.get("acquisition_type", "")).strip().upper() if isinstance(a_note, dict) else ""
            record_convergence_snapshot(
                sqlite_conn, args.run_name, runtime_method, sim_counter, result,
                search_hours_used * 3600.0, search_hours_used,
                acquisition_type=acq_type,
            )
            # Refresh heuristics after each run so next proposal uses updated knowledge
            heuristics_excerpt = build_heuristics_context(max_chars=900)
            append_result_research(research_path, result, "search")
            append_research(
                research_path,
                "\n#### Insights and Trends Update\n"
                f"- timestamp_utc: {utc_now_text()}\n"
                + sqlite_layout_trend_table(sqlite_conn)
                + "\n",
            )
            ledger.append(
                {
                    "phase": "search",
                    "run_name": result.get("run_name"),
                    "status": result.get("status"),
                    "timing": result.get("timing"),
                }
            )
            timing = result.get("timing") or {}
            search_hours_used += float(timing.get("wall_seconds", 0.0)) / 3600.0
            infeasibility_state = check_systematic_infeasibility(
                search_results,
                int(getattr(args, "systematic_infeasibility_k", 5)),
            )
            if bool(infeasibility_state.get("triggered")):
                force_diagnostic_next_iteration = True
                force_diagnostic_reason = str(infeasibility_state.get("reason", ""))

        validation_pool, finalization_gate_notes = build_validation_candidates(args, search_results, args.max_validations)
        if finalization_gate_notes:
            append_research(
                research_path,
                "\n#### Finalization Hard-Gate Notes\n"
                f"- timestamp_utc: {utc_now_text()}\n"
                + "\n".join([f"- {line}" for line in finalization_gate_notes[:20]])
                + "\n",
            )
        for ordinal, candidate in enumerate(validation_pool, start=1):
            if validation_hours_used >= args.validation_hours:
                break
            progress_log(
                f"AGENT: validation start ordinal={ordinal} source_run={candidate.get('run_name')}"
            )
            validation = execute_validation(args, candidate, ordinal)
            progress_log(
                "AGENT: validation done "
                + f"run={validation.get('run_name')} status={validation.get('status')} "
                + f"wall_s={float(((validation.get('timing') or {}).get('wall_seconds') or 0.0)):.1f}"
            )
            validation_results.append(validation)
            emit_live_event(
                "simulation_complete",
                iteration=search_iteration,
                role="validation",
                task={"nc": list(candidate.get("nc", [])), "seed_name": str(candidate.get("seed_name", ""))},
                result=validation,
                acquisition_type="VALIDATION",
                arbitration_outcome="VALIDATION",
            )
            persist_result_to_sqlite(sqlite_conn, args.run_name, "validation", validation)
            append_result_research(research_path, validation, "validation")
            append_research(
                research_path,
                "\n#### Insights and Trends Update\n"
                f"- timestamp_utc: {utc_now_text()}\n"
                + sqlite_layout_trend_table(sqlite_conn)
                + "\n",
            )
            ledger.append(
                {
                    "phase": "validation",
                    "source_run": candidate.get("run_name"),
                    "run_name": validation.get("run_name"),
                    "status": validation.get("status"),
                    "timing": validation.get("timing"),
                }
            )
            timing = validation.get("timing") or {}
            validation_hours_used += float(timing.get("wall_seconds", 0.0)) / 3600.0

        ranked_search = rank_any_results(search_results) if search_results else []
        ranked_validation = rs.rank_results([item for item in validation_results if item.get("status") == "ok"]) if validation_results else []
        final_best = ranked_validation[0] if ranked_validation else (ranked_search[0] if ranked_search else None)
        append_final_research(research_path, final_best, ranked_search, ranked_validation)

        payload = {
            "status": "ok",
            "run_name": args.run_name,
            "method": runtime_method,
            "mode": (
                "single-scientist"
                if bool(int(args.single_scientist_mode))
                else ("llm-assisted" if (client.enabled or client.fallback_enabled) else "deterministic-fallback")
            ),
            "llm": {
                "primary_enabled": client.enabled,
                "primary_base_url": args.llm_base_url,
                "primary_model": args.llm_model,
                "fallback_enabled": client.fallback_enabled,
                "fallback_base_url": args.fallback_llm_base_url if client.fallback_enabled else "",
                "fallback_model": args.fallback_llm_model if client.fallback_enabled else "",
                "last_backend_used": client.last_backend,
            },
            "single_scientist_mode": bool(int(args.single_scientist_mode)),
            "executive_controller": {
                "enabled": bool(args.executive_controller_enabled),
                "trigger_rejects": int(args.executive_trigger_rejects),
                "force_after_rejects": int(args.executive_force_after_rejects),
                "top_k_lock": int(args.executive_top_k_lock),
                "overrides_executed": sum(
                    1
                    for item in executive_log
                    if str((item.get("decision") or {}).get("decision", "")).lower() == "override_execute"
                ),
            },
            "probe_reference_policy": {
                "min_probe_reference_runs": int(args.min_probe_reference_runs),
                "available_reference_tasks": len(ranked_reference_indices(search_tasks)),
                "required_reference_runs": min(
                    int(args.min_probe_reference_runs),
                    len(ranked_reference_indices(search_tasks)),
                )
                if search_tasks
                else 0,
                "completed_reference_runs": reference_probe_runs_completed(search_results),
                "probe_low_fidelity_enabled": bool(int(args.probe_low_fidelity_enabled)),
                "probe_low_fidelity": {
                    "nfex": int(args.probe_nfex),
                    "nfet": int(args.probe_nfet),
                    "ncp": int(args.probe_ncp),
                },
            },
            "finalization_policy": {
                "hard_gate_enabled": bool(int(args.finalization_hard_gate_enabled)),
                "required_sequence": [
                    "low_fidelity_reference_seed",
                    "low_fidelity_non_reference_optimization",
                    "final_high_fidelity_validation",
                ],
                "low_fidelity_limits": low_fidelity_limits(args),
                "validation_pool_size": len(validation_pool),
                "gate_notes_count": len(finalization_gate_notes),
                "gate_notes": finalization_gate_notes[:20],
            },
            "llm_conversations": {
                "path": str(conversation_artifact.resolve()),
                "stream_path": str(conversation_stream_artifact.resolve()),
                "count": len(client.conversations),
            },
            "live_results": {
                "path": str(live_results_artifact.resolve()),
            },
            "benchmark_budget": {
                "total_hours": args.benchmark_hours,
                "search_hours": args.search_hours,
                "validation_hours": args.validation_hours,
                "search_hours_used": search_hours_used,
                "validation_hours_used": validation_hours_used,
            },
            "compute_summary": os.environ.get("SMB_COMPUTE_SUMMARY", ""),
            "solver_check": solver_check,
            "docs": {
                "objectives_file": args.objectives_file,
                "llm_soul_file": args.llm_soul_file,
                "problem_definition_file": args.problem_definition_file,
                "skills_file": args.skills_file,
                "ipopt_resource_file": args.ipopt_resource_file,
                "objectives_excerpt": objectives_excerpt,
                "llm_soul_excerpt": soul_excerpt,
                "problem_definition_excerpt": problem_definition_excerpt,
                "skills_excerpt": skills_excerpt,
                "ipopt_resource_excerpt": ipopt_excerpt,
                "compute_context_excerpt": compute_context_excerpt,
                "constraint_context_excerpt": constraint_context_excerpt,
            },
            "sqlite": {
                "db_path": str(Path(args.sqlite_db).resolve()),
                "record_count": sqlite_record_count(sqlite_conn),
            },
            "convergence": {
                "total_simulations": sim_counter,
                "method": runtime_method,
                "summary": sqlite_convergence_context(sqlite_conn, args.run_name),
            },
            "research": {
                "path": str(research_path.resolve()),
                "tail_excerpt": read_research_tail(research_path, min(args.research_tail_chars, 3000)),
                "initial_plan": initial_plan,
                "current_priorities": current_priorities,
                "layout_trends_current": sqlite_layout_trend_table(sqlite_conn),
                "nc_strategy_current": nc_strategy_board(sqlite_conn, nc_library_values),
            },
            "codebase_context": code_context,
            "scientist_a_log": scientist_a_log,
            "scientist_b_log": scientist_b_log,
            "executive_log": executive_log,
            "search_results": search_results,
            "validation_results": validation_results,
            "ranked_search_results": ranked_search,
            "ranked_validation_results": ranked_validation,
            "best_result": final_best,
            "ledger": ledger,
        }
        write_conversation_log(
            conversation_artifact,
            {
                "status": "ok",
                "run_name": args.run_name,
                "generated_at_utc": utc_now_text(),
                "llm": {
                    "primary_enabled": client.enabled,
                    "primary_base_url": args.llm_base_url,
                    "primary_model": args.llm_model,
                    "fallback_enabled": client.fallback_enabled,
                    "fallback_base_url": args.fallback_llm_base_url if client.fallback_enabled else "",
                    "fallback_model": args.fallback_llm_model if client.fallback_enabled else "",
                },
                "stream_path": str(conversation_stream_artifact.resolve()),
                "conversations": client.conversations,
            },
        )
        write_artifact(artifact, payload)
        print(json.dumps({"artifact": str(artifact), "status": "ok", "run_name": args.run_name}, indent=2))
        return 0
    except Exception as exc:
        try:
            write_conversation_log(
                conversation_artifact,
                {
                    "status": "error",
                    "run_name": args.run_name,
                    "generated_at_utc": utc_now_text(),
                    "error": str(exc),
                    "stream_path": str(conversation_stream_artifact.resolve()),
                    "conversations": client.conversations,
                },
            )
        except Exception:
            pass
        payload = {
            "status": "error",
            "run_name": args.run_name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "llm_conversations": {
                "path": str(conversation_artifact.resolve()),
                "stream_path": str(conversation_stream_artifact.resolve()),
                "count": len(client.conversations),
            },
            "live_results": {
                "path": str(live_results_artifact.resolve()),
            },
        }
        write_artifact(artifact, payload)
        print(json.dumps({"artifact": str(artifact), "status": "error", "run_name": args.run_name}, indent=2))
        return 1
    finally:
        sqlite_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
