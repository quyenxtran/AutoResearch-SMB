#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import textwrap
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib import error, request

from . import run_stage as rs


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-scientist SMB agent benchmark.")
    parser.add_argument("--run-name", default=os.environ.get("SMB_EXPERIMENT_NAME", "qwen_smb_two_scientists"))
    parser.add_argument("--artifact-dir", default=str(REPO_ROOT / "artifacts" / "agent_runs"))
    parser.add_argument("--conversation-log", default=os.environ.get("SMB_CONVERSATION_LOG", ""))
    parser.add_argument("--conversation-stream-log", default=os.environ.get("SMB_CONVERSATION_STREAM_LOG", ""))
    parser.add_argument("--sqlite-db", default=os.environ.get("SMB_SQLITE_DB", str(REPO_ROOT / "artifacts" / "agent_runs" / "smb_agent_context.sqlite")))
    parser.add_argument("--research-md", default=os.environ.get("SMB_RESEARCH_MD", str(REPO_ROOT / "research.md")))
    parser.add_argument("--research-tail-chars", type=int, default=int(os.environ.get("SMB_RESEARCH_TAIL_CHARS", "6000")))
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
    parser.add_argument("--llm-max-retries", type=int, default=int(os.environ.get("SMB_LLM_MAX_RETRIES", "2")))
    parser.add_argument(
        "--llm-retry-backoff-seconds",
        type=float,
        default=float(os.environ.get("SMB_LLM_RETRY_BACKOFF_SECONDS", "2.0")),
    )
    parser.add_argument(
        "--objectives-max-chars",
        type=int,
        default=int(os.environ.get("SMB_OBJECTIVES_MAX_CHARS", "6000")),
    )
    parser.add_argument(
        "--llm-soul-max-chars",
        type=int,
        default=int(os.environ.get("SMB_LLM_SOUL_MAX_CHARS", "3500")),
    )
    parser.add_argument(
        "--problem-definition-max-chars",
        type=int,
        default=int(os.environ.get("SMB_PROBLEM_DEFINITION_MAX_CHARS", "2500")),
    )
    parser.add_argument(
        "--skills-max-chars",
        type=int,
        default=int(os.environ.get("SMB_SKILLS_MAX_CHARS", "2200")),
    )
    parser.add_argument(
        "--ipopt-resource-max-chars",
        type=int,
        default=int(os.environ.get("SMB_IPOPT_RESOURCE_MAX_CHARS", "1600")),
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


def as_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def layout_text(nc: object) -> str:
    if isinstance(nc, (list, tuple)) and len(nc) == 4:
        return ",".join(str(int(v)) for v in nc)
    return ""


def extract_metrics_with_validity(result: Dict[str, object]) -> Tuple[Dict[str, object], Optional[bool]]:
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        return metrics, True
    provisional = result.get("provisional")
    if isinstance(provisional, dict):
        provisional_metrics = provisional.get("metrics")
        if isinstance(provisional_metrics, dict):
            return provisional_metrics, False
    return {}, None


def open_sqlite_db(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS simulation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            agent_run_name TEXT NOT NULL,
            phase TEXT NOT NULL,
            source_stage TEXT,
            candidate_run_name TEXT NOT NULL,
            nc TEXT,
            seed_name TEXT,
            status TEXT,
            feasible INTEGER,
            j_validated REAL,
            productivity REAL,
            purity REAL,
            recovery_ga REAL,
            recovery_ma REAL,
            normalized_total_violation REAL,
            metrics_validated INTEGER,
            solver_name TEXT,
            linear_solver TEXT,
            termination_condition TEXT,
            wall_seconds REAL,
            cpu_hours REAL,
            ffeed REAL,
            f1 REAL,
            fdes REAL,
            fex REAL,
            fraf REAL,
            tstep REAL,
            raw_json TEXT,
            UNIQUE(agent_run_name, phase, candidate_run_name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_created ON simulation_results(created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_feasible ON simulation_results(feasible)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_jval ON simulation_results(j_validated DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_violation ON simulation_results(normalized_total_violation ASC)")

    # Convergence tracking table: records best-so-far after each simulation for sample efficiency comparison.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS convergence_tracker (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            agent_run_name TEXT NOT NULL,
            method TEXT NOT NULL,
            sim_number INTEGER NOT NULL,
            candidate_run_name TEXT,
            best_feasible_j REAL,
            best_feasible_productivity REAL,
            best_feasible_run_name TEXT,
            cumulative_wall_seconds REAL,
            cumulative_cpu_hours REAL,
            total_feasible INTEGER,
            total_runs INTEGER,
            acquisition_type TEXT,
            nc_layouts_tested INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_method ON convergence_tracker(method, agent_run_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_sim ON convergence_tracker(sim_number)")
    conn.commit()
    return conn


def persist_result_to_sqlite(conn: sqlite3.Connection, agent_run_name: str, phase: str, result: Dict[str, object]) -> None:
    metrics, metrics_validated = extract_metrics_with_validity(result)
    flow = effective_flow(result) or {}
    slacks = result.get("constraint_slacks")
    if not isinstance(slacks, dict):
        slacks = {}
    solver = result.get("solver")
    if not isinstance(solver, dict):
        solver = {}
    solver_options = solver.get("solver_options")
    if not isinstance(solver_options, dict):
        solver_options = {}
    timing = result.get("timing")
    if not isinstance(timing, dict):
        timing = {}

    conn.execute(
        """
        INSERT INTO simulation_results (
            agent_run_name, phase, source_stage, candidate_run_name, nc, seed_name, status, feasible, j_validated,
            productivity, purity, recovery_ga, recovery_ma, normalized_total_violation, metrics_validated,
            solver_name, linear_solver, termination_condition, wall_seconds, cpu_hours, ffeed, f1, fdes, fex,
            fraf, tstep, raw_json
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(agent_run_name, phase, candidate_run_name) DO UPDATE SET
            created_at = datetime('now'),
            source_stage = excluded.source_stage,
            nc = excluded.nc,
            seed_name = excluded.seed_name,
            status = excluded.status,
            feasible = excluded.feasible,
            j_validated = excluded.j_validated,
            productivity = excluded.productivity,
            purity = excluded.purity,
            recovery_ga = excluded.recovery_ga,
            recovery_ma = excluded.recovery_ma,
            normalized_total_violation = excluded.normalized_total_violation,
            metrics_validated = excluded.metrics_validated,
            solver_name = excluded.solver_name,
            linear_solver = excluded.linear_solver,
            termination_condition = excluded.termination_condition,
            wall_seconds = excluded.wall_seconds,
            cpu_hours = excluded.cpu_hours,
            ffeed = excluded.ffeed,
            f1 = excluded.f1,
            fdes = excluded.fdes,
            fex = excluded.fex,
            fraf = excluded.fraf,
            tstep = excluded.tstep,
            raw_json = excluded.raw_json
        """,
        (
            agent_run_name,
            phase,
            str(result.get("stage", "")),
            str(result.get("run_name", "")),
            layout_text(result.get("nc")),
            str(result.get("seed_name", "")),
            str(result.get("status", "")),
            1 if bool(result.get("feasible", False)) else 0,
            as_float(result.get("J_validated")),
            as_float(metrics.get("productivity_ex_ga_ma")),
            as_float(metrics.get("purity_ex_meoh_free")),
            as_float(metrics.get("recovery_ex_GA")),
            as_float(metrics.get("recovery_ex_MA")),
            as_float(slacks.get("normalized_total_violation")),
            1 if metrics_validated is True else (0 if metrics_validated is False else None),
            str(solver.get("solver_name", "")),
            str(solver_options.get("linear_solver", "")),
            str(solver.get("termination_condition", "")),
            as_float(timing.get("wall_seconds")),
            as_float(timing.get("cpu_hours_accounted")),
            as_float(flow.get("Ffeed")),
            as_float(flow.get("F1")),
            as_float(flow.get("Fdes")),
            as_float(flow.get("Fex")),
            as_float(flow.get("Fraf")),
            as_float(flow.get("tstep")),
            json.dumps(result, separators=(",", ":"), ensure_ascii=True),
        ),
    )
    conn.commit()


def record_convergence_snapshot(
    conn: sqlite3.Connection,
    agent_run_name: str,
    method: str,
    sim_number: int,
    result: Dict[str, object],
    cumulative_wall_seconds: float,
    cumulative_cpu_hours: float,
    acquisition_type: str = "",
) -> None:
    """Record a convergence tracking point after each simulation.

    This creates the data needed to plot best-feasible-J vs simulation-count
    for agent vs MINLP comparison.
    """
    # Query the current best feasible result across all runs for this method
    best_row = conn.execute(
        """
        SELECT j_validated, productivity, candidate_run_name
        FROM simulation_results
        WHERE agent_run_name = ? AND feasible = 1 AND j_validated IS NOT NULL
        ORDER BY j_validated DESC LIMIT 1
        """,
        (agent_run_name,),
    ).fetchone()

    best_j = float(best_row[0]) if best_row else None
    best_prod = float(best_row[1]) if best_row else None
    best_run = str(best_row[2]) if best_row else None

    counts = conn.execute(
        """
        SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN feasible=1 THEN 1 ELSE 0 END), 0) AS feasible_count,
               COUNT(DISTINCT nc) AS nc_count
        FROM simulation_results
        WHERE agent_run_name = ?
        """,
        (agent_run_name,),
    ).fetchone()

    conn.execute(
        """
        INSERT INTO convergence_tracker (
            agent_run_name, method, sim_number, candidate_run_name,
            best_feasible_j, best_feasible_productivity, best_feasible_run_name,
            cumulative_wall_seconds, cumulative_cpu_hours,
            total_feasible, total_runs, acquisition_type, nc_layouts_tested
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agent_run_name,
            method,
            sim_number,
            str(result.get("run_name", "")),
            best_j,
            best_prod,
            best_run,
            cumulative_wall_seconds,
            cumulative_cpu_hours,
            int(counts[1]) if counts else 0,
            int(counts[0]) if counts else 0,
            acquisition_type or "",
            int(counts[2]) if counts else 0,
        ),
    )
    conn.commit()


def sqlite_convergence_context(conn: sqlite3.Connection, agent_run_name: str) -> str:
    """Build a convergence summary for the agent to assess its own progress."""
    rows = conn.execute(
        """
        SELECT sim_number, best_feasible_j, best_feasible_productivity,
               best_feasible_run_name, total_feasible, total_runs,
               nc_layouts_tested, acquisition_type, cumulative_wall_seconds
        FROM convergence_tracker
        WHERE agent_run_name = ?
        ORDER BY sim_number ASC
        """,
        (agent_run_name,),
    ).fetchall()
    if not rows:
        return "Convergence tracker: no data yet."

    lines = ["Convergence tracker (best feasible J after each simulation):"]
    last_j = None
    stagnation_count = 0
    for row in rows:
        sim_num, best_j, best_prod, best_run, n_feasible, n_total, n_nc, acq_type, cum_wall = row
        improved = ""
        if best_j is not None:
            if last_j is None or best_j > last_j:
                improved = " [NEW BEST]"
                stagnation_count = 0
                last_j = best_j
            else:
                stagnation_count += 1
        lines.append(
            f"- sim={sim_num} best_J={best_j} best_prod={best_prod} best_run={best_run} "
            f"feasible={n_feasible}/{n_total} nc_tested={n_nc} type={acq_type} "
            f"wall_s={cum_wall:.1f}{improved}"
        )

    # Summary statistics
    total_explore = sum(1 for r in rows if r[7] == "EXPLORE")
    total_exploit = sum(1 for r in rows if r[7] == "EXPLOIT")
    total_verify = sum(1 for r in rows if r[7] == "VERIFY")
    total_other = len(rows) - total_explore - total_exploit - total_verify
    lines.append(
        f"Acquisition balance: EXPLORE={total_explore} EXPLOIT={total_exploit} "
        f"VERIFY={total_verify} other={total_other}"
    )
    lines.append(f"Simulations since last improvement: {stagnation_count}")
    return "\n".join(lines)


def sqlite_targeted_query(conn: sqlite3.Connection, query_type: str, **kwargs: object) -> str:
    """Run targeted queries against the simulation history for deeper analysis."""
    lines: List[str] = []

    if query_type == "nc_detail":
        nc_val = str(kwargs.get("nc", ""))
        rows = conn.execute(
            """
            SELECT candidate_run_name, seed_name, status, feasible, j_validated, productivity,
                   purity, recovery_ga, recovery_ma, normalized_total_violation,
                   ffeed, f1, fdes, fex, fraf, tstep, termination_condition
            FROM simulation_results WHERE nc = ?
            ORDER BY j_validated DESC, productivity DESC, id DESC
            """,
            (nc_val,),
        ).fetchall()
        lines.append(f"All runs for nc={nc_val} ({len(rows)} total):")
        for row in rows:
            lines.append(
                f"  {row[0]} seed={row[1]} status={row[2]} feasible={bool(row[3])} J={row[4]} prod={row[5]} "
                f"purity={row[6]} rGA={row[7]} rMA={row[8]} viol={row[9]} "
                f"flow(Ffeed={row[10]},F1={row[11]},Fdes={row[12]},Fex={row[13]},Fraf={row[14]},tstep={row[15]}) "
                f"term={row[16]}"
            )

    elif query_type == "flow_region":
        min_ffeed = float(kwargs.get("min_ffeed", 0.0))
        max_ffeed = float(kwargs.get("max_ffeed", 99.0))
        rows = conn.execute(
            """
            SELECT candidate_run_name, nc, feasible, j_validated, productivity, purity,
                   recovery_ga, recovery_ma, ffeed, f1, fdes, fex, tstep
            FROM simulation_results
            WHERE ffeed >= ? AND ffeed <= ?
            ORDER BY j_validated DESC, productivity DESC
            LIMIT 20
            """,
            (min_ffeed, max_ffeed),
        ).fetchall()
        lines.append(f"Runs with Ffeed in [{min_ffeed}, {max_ffeed}] ({len(rows)} shown):")
        for row in rows:
            lines.append(
                f"  {row[0]} nc={row[1]} feasible={bool(row[2])} J={row[3]} prod={row[4]} "
                f"purity={row[5]} rGA={row[6]} rMA={row[7]} "
                f"Ffeed={row[8]} F1={row[9]} Fdes={row[10]} Fex={row[11]} tstep={row[12]}"
            )

    elif query_type == "binding_constraint":
        rows = conn.execute(
            """
            SELECT candidate_run_name, nc, purity, recovery_ga, recovery_ma,
                   normalized_total_violation, productivity, feasible
            FROM simulation_results
            WHERE feasible = 0 AND normalized_total_violation IS NOT NULL
            ORDER BY normalized_total_violation ASC
            LIMIT 20
            """,
        ).fetchall()
        lines.append(f"Near-feasible runs sorted by violation ({len(rows)} shown):")
        for row in rows:
            purity = row[2] or 0.0
            rga = row[3] or 0.0
            rma = row[4] or 0.0
            # Identify which constraint is most binding
            bottleneck = []
            if purity < 0.60:
                bottleneck.append(f"purity({purity:.4f}<0.60)")
            if rga < 0.75:
                bottleneck.append(f"rGA({rga:.4f}<0.75)")
            if rma < 0.75:
                bottleneck.append(f"rMA({rma:.4f}<0.75)")
            lines.append(
                f"  {row[0]} nc={row[1]} viol={row[5]} prod={row[6]} "
                f"binding=[{', '.join(bottleneck) if bottleneck else 'unknown'}]"
            )

    elif query_type == "improvement_history":
        rows = conn.execute(
            """
            SELECT id, candidate_run_name, nc, feasible, j_validated, productivity,
                   purity, wall_seconds
            FROM simulation_results
            ORDER BY id ASC
            """,
        ).fetchall()
        lines.append("Improvement history (cumulative best J over time):")
        best_j: Optional[float] = None
        for row in rows:
            j = row[4]
            if row[3] and j is not None:
                if best_j is None or j > best_j:
                    best_j = j
                    lines.append(
                        f"  sim={row[0]} {row[1]} nc={row[2]} J={j} prod={row[5]} purity={row[6]} [IMPROVED]"
                    )

    else:
        lines.append(f"Unknown query type: {query_type}")

    return "\n".join(lines) if lines else "No results."


def sqlite_history_context(conn: sqlite3.Connection, max_feasible: int = 5, max_near: int = 5, max_recent: int = 6) -> str:
    counts = conn.execute(
        """
        SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN feasible=1 THEN 1 ELSE 0 END), 0) AS feasible_count
        FROM simulation_results
        """
    ).fetchone()
    total = int(counts[0]) if counts else 0
    feasible_count = int(counts[1]) if counts else 0

    feasible_rows = conn.execute(
        """
        SELECT candidate_run_name, nc, seed_name, j_validated, productivity, purity, recovery_ga, recovery_ma
        FROM simulation_results
        WHERE feasible=1 AND j_validated IS NOT NULL
        ORDER BY j_validated DESC, productivity DESC, id DESC
        LIMIT ?
        """,
        (max_feasible,),
    ).fetchall()

    near_rows = conn.execute(
        """
        SELECT candidate_run_name, nc, seed_name, normalized_total_violation, productivity, purity, recovery_ga, recovery_ma, metrics_validated
        FROM simulation_results
        WHERE feasible=0 AND normalized_total_violation IS NOT NULL
        ORDER BY normalized_total_violation ASC, productivity DESC, id DESC
        LIMIT ?
        """,
        (max_near,),
    ).fetchall()

    recent_rows = conn.execute(
        """
        SELECT
            candidate_run_name, nc, status, feasible, productivity, purity, recovery_ga, recovery_ma,
            normalized_total_violation, metrics_validated, ffeed, f1, fdes, fex, fraf, tstep, raw_json
        FROM simulation_results
        ORDER BY id DESC
        LIMIT ?
        """,
        (max_recent,),
    ).fetchall()

    lines = [
        f"SQLite context: total_records={total}, feasible_records={feasible_count}",
        "Top feasible records by J_validated:",
    ]
    if feasible_rows:
        for row in feasible_rows:
            lines.append(
                f"- {row[0]} nc={row[1]} seed={row[2]} J={row[3]} prod={row[4]} purity={row[5]} rGA={row[6]} rMA={row[7]}"
            )
    else:
        lines.append("- none")

    lines.append("Top near-feasible records by normalized violation:")
    if near_rows:
        for row in near_rows:
            lines.append(
                f"- {row[0]} nc={row[1]} seed={row[2]} viol={row[3]} prod={row[4]} purity={row[5]} rGA={row[6]} rMA={row[7]} metrics_validated={row[8]}"
            )
    else:
        lines.append("- none")

    lines.append("Most recent records:")
    if recent_rows:
        for row in recent_rows:
            lines.append(
                f"- {row[0]} nc={row[1]} status={row[2]} feasible={bool(row[3])} prod={row[4]} purity={row[5]} rGA={row[6]} rMA={row[7]} viol={row[8]} metrics_validated={row[9]} "
                f"flow(Ffeed={row[10]},F1={row[11]},Fdes={row[12]},Fex={row[13]},Fraf={row[14]},tstep={row[15]})"
            )
    else:
        lines.append("- none")

    lines.append("Recent composition snapshots (outlet CE/CR):")
    comp_points: List[Dict[str, object]] = []
    if recent_rows:
        for row in recent_rows:
            comp = composition_metrics_from_raw_json(str(row[16] or ""))
            if not comp:
                continue
            point = {
                "run_name": row[0],
                "nc": row[1],
                "ffeed": as_float(row[10]),
                "tstep": as_float(row[15]),
                **comp,
            }
            comp_points.append(point)
            lines.append(
                f"- {row[0]} nc={row[1]} Ffeed={row[10]} tstep={row[15]} "
                f"CE_acid={comp['ce_acid']} CE_water={comp['ce_water']} CE_meoh={comp['ce_meoh']} "
                f"CR_acid={comp['cr_acid']} CR_water={comp['cr_water']} CR_meoh={comp['cr_meoh']} source={comp['source']}"
            )
    if not comp_points:
        lines.append("- none")
    else:
        lines.append("Flow/composition trend hints:")
        slope_ffeed_ce_acid = linear_slope(
            [as_float(item.get("ffeed")) for item in comp_points],
            [as_float(item.get("ce_acid")) for item in comp_points],
        )
        slope_tstep_ce_acid = linear_slope(
            [as_float(item.get("tstep")) for item in comp_points],
            [as_float(item.get("ce_acid")) for item in comp_points],
        )
        if slope_ffeed_ce_acid is not None:
            direction = "increases" if slope_ffeed_ce_acid > 0 else "decreases"
            lines.append(f"- As Ffeed rises, CE_acid generally {direction} (slope={slope_ffeed_ce_acid:.6g}).")
        if slope_tstep_ce_acid is not None:
            direction = "increases" if slope_tstep_ce_acid > 0 else "decreases"
            lines.append(f"- As tstep rises, CE_acid generally {direction} (slope={slope_tstep_ce_acid:.6g}).")

        by_nc: Dict[str, Dict[str, float]] = {}
        for item in comp_points:
            nc_label = str(item.get("nc", ""))
            bucket = by_nc.setdefault(nc_label, {"n": 0.0, "ce_acid_sum": 0.0, "cr_acid_sum": 0.0})
            ce_acid = as_float(item.get("ce_acid"))
            cr_acid = as_float(item.get("cr_acid"))
            if ce_acid is None or cr_acid is None:
                continue
            bucket["n"] += 1.0
            bucket["ce_acid_sum"] += ce_acid
            bucket["cr_acid_sum"] += cr_acid
        if by_nc:
            lines.append("NC-level composition means (recent rows):")
            ranked_nc = sorted(
                by_nc.items(),
                key=lambda kv: (kv[1]["ce_acid_sum"] / kv[1]["n"]) if kv[1]["n"] > 0 else float("-inf"),
                reverse=True,
            )
            for nc_label, bucket in ranked_nc[:6]:
                if bucket["n"] <= 0:
                    continue
                ce_mean = bucket["ce_acid_sum"] / bucket["n"]
                cr_mean = bucket["cr_acid_sum"] / bucket["n"]
                lines.append(f"- nc={nc_label} mean_CE_acid={ce_mean:.6g} mean_CR_acid={cr_mean:.6g} n={int(bucket['n'])}")

    return "\n".join(lines)


def sqlite_record_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM simulation_results").fetchone()
    return int(row[0]) if row else 0


def sqlite_layout_trend_table(conn: sqlite3.Connection, max_rows: int = 10) -> str:
    rows = conn.execute(
        """
        SELECT
            COALESCE(nc, '') AS nc,
            COUNT(*) AS n_total,
            COALESCE(SUM(CASE WHEN feasible=1 THEN 1 ELSE 0 END), 0) AS n_feasible,
            MIN(normalized_total_violation) AS best_violation,
            MAX(productivity) AS best_productivity,
            MAX(j_validated) AS best_j_validated
        FROM simulation_results
        GROUP BY nc
        ORDER BY
            COALESCE(MAX(j_validated), -1e99) DESC,
            COALESCE(MAX(productivity), -1e99) DESC,
            COALESCE(MIN(normalized_total_violation), 1e99) ASC,
            nc ASC
        LIMIT ?
        """,
        (max_rows,),
    ).fetchall()
    if not rows:
        return "No layout trends yet."

    lines = [
        "| nc | n_total | n_feasible | best_violation | best_productivity | best_J_validated |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for nc, n_total, n_feasible, best_violation, best_productivity, best_j_validated in rows:
        lines.append(
            "| "
            + f"{nc or '<none>'} | {int(n_total)} | {int(n_feasible)} | "
            + f"{best_violation if best_violation is not None else ''} | "
            + f"{best_productivity if best_productivity is not None else ''} | "
            + f"{best_j_validated if best_j_validated is not None else ''} |"
        )
    return "\n".join(lines)


def nc_key(nc: Sequence[int]) -> str:
    return ",".join(str(int(v)) for v in nc)


def nc_prior_score(nc: Sequence[int]) -> float:
    # Neutral structural prior: mild penalty for extreme column count asymmetry only.
    # Does NOT bias toward any specific layout (e.g., reference (1,2,3,2)).
    # The only structural preference is against layouts where one zone has all 8 columns
    # (physically degenerate) or where asymmetry is so extreme the zone functions break down.
    vals = [int(v) for v in nc]
    asymmetry = max(vals) - min(vals)
    return 100.0 - 1.5 * asymmetry


def sqlite_total_records_from_excerpt(text: str) -> int:
    match = re.search(r"total_records=(\d+)", text or "")
    return int(match.group(1)) if match else 0


def text_mentions_prior_runs(items: Sequence[str]) -> bool:
    pattern = re.compile(r"(run_name|run=|search_|validate_|reference-eval|optimize-layouts|status=|viol=|J=)")
    return any(pattern.search(str(item)) for item in items)


def text_mentions_metric_signals(items: Sequence[str]) -> bool:
    pattern = re.compile(
        r"(prod(?:uctivity)?=|productivity|purity|recovery|rga=|rma=|viol(?:ation)?=|normalized_total_violation|J=|feasible=)",
        flags=re.IGNORECASE,
    )
    return any(pattern.search(str(item)) for item in items)


def text_mentions_numeric_values(items: Sequence[str]) -> bool:
    pattern = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", flags=re.IGNORECASE)
    return any(pattern.search(str(item)) for item in items)


def text_mentions_delta_metric_signals(items: Sequence[str]) -> bool:
    blob = " ".join(str(item) for item in items)
    required = [
        r"(?:Δ|delta|d)[_\-\s]?(?:prod|productivity)",
        r"(?:Δ|delta|d)[_\-\s]?purity",
        r"(?:Δ|delta|d)[_\-\s]?rga",
        r"(?:Δ|delta|d)[_\-\s]?rma",
        r"(?:Δ|delta|d)[_\-\s]?(?:viol|violation)",
    ]
    return all(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in required)


def count_flow_signal_mentions(items: Sequence[str]) -> int:
    flow_tokens = ("ffeed", "f1", "fdes", "fex", "fraf", "tstep")
    blob = " ".join(str(item).lower() for item in items)
    return sum(1 for token in flow_tokens if token in blob)


def text_mentions_delta_flow_signals(items: Sequence[str], min_count: int = 3) -> bool:
    if count_flow_signal_mentions(items) < min_count:
        return False
    blob = " ".join(str(item) for item in items)
    pattern = re.compile(
        r"(?:Δ|delta|d)[_\-\s]?(?:ffeed|f1|fdes|fex|fraf|tstep)",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(blob))


def text_mentions_run_name_signals(items: Sequence[str]) -> bool:
    pattern = re.compile(r"(run_name=|run=|search_nc_|validate_|reference)", flags=re.IGNORECASE)
    return any(pattern.search(str(item)) for item in items)


def text_mentions_required_labels(items: Sequence[str], labels: Sequence[str]) -> bool:
    if not labels:
        return True
    blob = " ".join(str(item) for item in items)
    return all(str(label) in blob for label in labels)


def text_mentions_flow_signals(items: Sequence[str]) -> bool:
    pattern = re.compile(r"(Ffeed|F1|Fdes|Fex|Fraf|tstep|flow|F2|F4)", flags=re.IGNORECASE)
    return any(pattern.search(str(item)) for item in items)


def text_mentions_topology_signals(items: Sequence[str]) -> bool:
    pattern = re.compile(
        r"(nc=|nc\[|nc\s*\[|topology|zone|z1|z2|z3|z4|column|columns|fragmentation|symmetry|Δz1|Δz2|Δz3|Δz4)",
        flags=re.IGNORECASE,
    )
    return any(pattern.search(str(item)) for item in items)


def text_mentions_physics_signals(items: Sequence[str]) -> bool:
    pattern = re.compile(
        r"(mass\s*balance|mass\s*transfer|adsorption|desorption|zone|selectivity|isotherm|equilibrium|transport|hydrodynamic|flow\s*split|residence)",
        flags=re.IGNORECASE,
    )
    return any(pattern.search(str(item)) for item in items)


def extract_nc_mentions(text: str) -> set[Tuple[int, int, int, int]]:
    mentions: set[Tuple[int, int, int, int]] = set()
    if not text:
        return mentions
    pattern = re.compile(
        r"(?:nc\s*[:=]\s*)?[\[\(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)]",
        flags=re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        mentions.add((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return mentions


def review_references_candidate_nc(
    reason: str,
    comparisons: Sequence[str],
    nc_assessment: Sequence[str],
    candidate_nc: Sequence[int],
) -> bool:
    candidate = tuple(int(v) for v in candidate_nc)
    blob = " ".join([str(reason)] + [str(x) for x in comparisons] + [str(x) for x in nc_assessment])
    mentioned = extract_nc_mentions(blob)
    # If no explicit NC text is present, we do not fail this check.
    if not mentioned:
        return True
    return candidate in mentioned


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


def configure_stage_args(base: argparse.Namespace, args: argparse.Namespace) -> argparse.Namespace:
    stage_args = argparse.Namespace(**vars(base))
    stage_args.solver_name = args.solver_name
    stage_args.linear_solver = args.linear_solver
    stage_args.tee = args.tee
    stage_args.nc_library = args.nc_library
    stage_args.seed_library = args.seed_library
    stage_args.max_iter = int(env_or_default("SMB_IPOPT_MAX_ITER", "1000"))
    stage_args.tol = float(env_or_default("SMB_IPOPT_TOL", "1e-5"))
    stage_args.acceptable_tol = float(env_or_default("SMB_IPOPT_ACCEPTABLE_TOL", "1e-4"))
    stage_args.nfex = int(env_or_default("SMB_NFEX", str(stage_args.nfex)))
    stage_args.nfet = int(env_or_default("SMB_NFET", str(stage_args.nfet)))
    stage_args.ncp = int(env_or_default("SMB_NCP", str(stage_args.ncp)))
    stage_args.ffeed_bounds = env_or_default("SMB_FFEED_BOUNDS", stage_args.ffeed_bounds)
    stage_args.f1_bounds = env_or_default("SMB_F1_BOUNDS", stage_args.f1_bounds)
    stage_args.fdes_bounds = env_or_default("SMB_FDES_BOUNDS", stage_args.fdes_bounds)
    stage_args.fex_bounds = env_or_default("SMB_FEX_BOUNDS", stage_args.fex_bounds)
    stage_args.fraf_bounds = env_or_default("SMB_FRAF_BOUNDS", stage_args.fraf_bounds)
    stage_args.tstep_bounds = env_or_default("SMB_TSTEP_BOUNDS", stage_args.tstep_bounds)
    stage_args.max_pump_flow = float(env_or_default("SMB_MAX_PUMP_FLOW_ML_MIN", str(stage_args.max_pump_flow)))
    stage_args.f1_max_flow = float(env_or_default("SMB_F1_MAX_FLOW", str(stage_args.f1_max_flow)))
    stage_args.f1_max = float(env_or_default("SMB_F1_MAX_FLOW", str(stage_args.f1_max_flow)))
    stage_args.purity_min = float(env_or_default("SMB_TARGET_PURITY_EX_MEOH_FREE", str(stage_args.purity_min)))
    stage_args.recovery_ga_min = float(env_or_default("SMB_TARGET_RECOVERY_GA", str(stage_args.recovery_ga_min)))
    stage_args.recovery_ma_min = float(env_or_default("SMB_TARGET_RECOVERY_MA", str(stage_args.recovery_ma_min)))
    stage_args.project_purity_min = float(
        env_or_default("SMB_PROJECT_TARGET_PURITY_EX_MEOH_FREE", str(getattr(args, "project_purity_min", stage_args.purity_min)))
    )
    stage_args.project_recovery_ga_min = float(
        env_or_default("SMB_PROJECT_TARGET_RECOVERY_GA", str(getattr(args, "project_recovery_ga_min", stage_args.recovery_ga_min)))
    )
    stage_args.project_recovery_ma_min = float(
        env_or_default("SMB_PROJECT_TARGET_RECOVERY_MA", str(getattr(args, "project_recovery_ma_min", stage_args.recovery_ma_min)))
    )
    stage_args.meoh_max_raff_wt = float(env_or_default("SMB_MEOH_MAX_RAFF_WT", str(stage_args.meoh_max_raff_wt)))
    stage_args.water_max_ex_wt = float(env_or_default("SMB_WATER_MAX_EX_WT", str(stage_args.water_max_ex_wt)))
    stage_args.water_max_zone1_entry_wt = float(
        env_or_default("SMB_WATER_MAX_ZONE1_ENTRY_WT", str(stage_args.water_max_zone1_entry_wt))
    )
    return stage_args


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        enabled: bool,
        api_key: str = "ollama",
        fallback_enabled: bool = False,
        fallback_base_url: str = "",
        fallback_model: str = "",
        fallback_api_key: str = "",
        conversation_stream_path: Optional[Path] = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.enabled = enabled and bool(self.base_url) and bool(self.model)
        self.fallback_base_url = fallback_base_url.rstrip("/")
        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_enabled = (
            fallback_enabled
            and bool(self.fallback_base_url)
            and bool(self.fallback_model)
            and bool(self.fallback_api_key)
        )
        self.last_backend = "none"
        self.call_counter = 0
        self.conversations: List[Dict[str, object]] = []
        self.conversation_stream_path = conversation_stream_path
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    def _append_conversation_stream(self, record: Dict[str, object]) -> None:
        if self.conversation_stream_path is None:
            return
        try:
            self.conversation_stream_path.parent.mkdir(parents=True, exist_ok=True)
            with self.conversation_stream_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            # Streaming is best-effort; full transcript is still persisted at run end.
            return

    def _chat_once(
        self,
        base_url: str,
        model: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        stop_sequences: Sequence[str],
    ) -> Tuple[Optional[str], str]:
        if not base_url or not model:
            return None, "missing_base_url_or_model"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def build_payload(include_stop: bool, include_temperature: bool) -> Dict[str, object]:
            payload: Dict[str, object] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if include_temperature:
                payload["temperature"] = temperature
            if include_stop:
                payload["stop"] = list(stop_sequences)
            return payload

        # Some providers/models reject stop sequences and/or custom temperature controls.
        payload_variants = [
            ("full", build_payload(True, True)),
            ("no_stop", build_payload(False, True)),
            ("no_temp", build_payload(True, False)),
            ("no_stop_no_temp", build_payload(False, False)),
        ]
        last_error = "unknown_error"

        for variant_name, payload in payload_variants:
            for attempt in range(1, self.max_retries + 1):
                req = request.Request(
                    f"{base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                try:
                    with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"], ""
                except error.HTTPError as exc:
                    response_body = ""
                    try:
                        response_body = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        response_body = ""
                    response_body = " ".join(response_body.split())[:320]
                    last_error = f"http_error_{exc.code}" + (f": {response_body}" if response_body else "")

                    # Retry transient HTTP failures.
                    if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                        if self.retry_backoff_seconds > 0:
                            time.sleep(self.retry_backoff_seconds * attempt)
                        continue

                    # If the provider rejects stop controls, retry with the next payload variant
                    # that omits stop. This must apply to any variant that currently includes stop.
                    if (
                        exc.code == 400
                        and "stop" in payload
                        and "stop" in response_body.lower()
                        and ("unsupported" in response_body.lower() or "not supported" in response_body.lower())
                    ):
                        break
                    # If the provider rejects custom temperature values, retry with the next payload
                    # variant that omits temperature.
                    if (
                        exc.code == 400
                        and "temperature" in payload
                        and "temperature" in response_body.lower()
                        and ("unsupported" in response_body.lower() or "default" in response_body.lower())
                    ):
                        break
                    return None, last_error
                except error.URLError as exc:
                    last_error = f"url_error_{str(exc.reason)}"
                    if attempt < self.max_retries:
                        if self.retry_backoff_seconds > 0:
                            time.sleep(self.retry_backoff_seconds * attempt)
                        continue
                    return None, last_error
                except TimeoutError:
                    last_error = "timeout"
                    if attempt < self.max_retries:
                        if self.retry_backoff_seconds > 0:
                            time.sleep(self.retry_backoff_seconds * attempt)
                        continue
                    return None, last_error
                except KeyError:
                    return None, "missing_choices_message_content"
                except json.JSONDecodeError:
                    return None, "invalid_json_response"
                except Exception as exc:  # pragma: no cover - defensive fallback
                    return None, f"unexpected_error_{type(exc).__name__}"
        return None, last_error

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_role: str = "generic",
        metadata: Optional[Dict[str, object]] = None,
        temperature: float = 0.2,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        resolved_stop = tuple(stop_sequences or ("<|endoftext|>", "<|im_start|>", "<|im_end|>"))
        self.call_counter += 1
        record: Dict[str, object] = {
            "call_id": self.call_counter,
            "timestamp_utc": utc_now_text(),
            "role": conversation_role,
            "metadata": metadata or {},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "attempts": [],
        }

        if self.enabled:
            content, err = self._chat_once(
                self.base_url,
                self.model,
                self.api_key,
                system_prompt,
                user_prompt,
                temperature,
                resolved_stop,
            )
            record["attempts"].append(
                {
                    "backend": "primary",
                    "base_url": self.base_url,
                    "model": self.model,
                    "success": content is not None,
                    "error": err,
                }
            )
            if content is not None:
                self.last_backend = "primary"
                record["final_backend"] = self.last_backend
                record["assistant_response"] = content
                self.conversations.append(record)
                self._append_conversation_stream(record)
                return content
        if self.fallback_enabled:
            content, err = self._chat_once(
                self.fallback_base_url,
                self.fallback_model,
                self.fallback_api_key,
                system_prompt,
                user_prompt,
                temperature,
                resolved_stop,
            )
            record["attempts"].append(
                {
                    "backend": "fallback",
                    "base_url": self.fallback_base_url,
                    "model": self.fallback_model,
                    "success": content is not None,
                    "error": err,
                }
            )
            if content is not None:
                self.last_backend = "fallback"
                record["final_backend"] = self.last_backend
                record["assistant_response"] = content
                self.conversations.append(record)
                self._append_conversation_stream(record)
                return content
        self.last_backend = "none"
        record["final_backend"] = self.last_backend
        self.conversations.append(record)
        self._append_conversation_stream(record)
        return None

    @staticmethod
    def extract_json(text: Optional[str]) -> Optional[Dict[str, object]]:
        if not text:
            return None
        cleaned = text
        # Keep chain-of-thought enabled upstream, but strip think blocks before JSON decoding.
        cleaned = re.sub(
            r"<\s*think\s*>.*?<\s*/\s*think\s*>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for marker in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
            idx = cleaned.find(marker)
            if idx != -1:
                cleaned = cleaned[:idx]
        decoder = json.JSONDecoder()
        for match in re.finditer(r"{", cleaned):
            start = match.start()
            try:
                candidate, _ = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                return candidate
        return None


def read_doc_excerpt(path: str, max_chars: int = 4000) -> str:
    p = Path(path)
    if not p.exists():
        return f"Missing file: {path}"
    text = p.read_text(encoding="utf-8")
    return compact_prompt_block(text, max_chars=max_chars, max_lines=200)


def compact_prompt_block(text: str, max_chars: int = 2000, max_lines: int = 80) -> str:
    """Compress context blocks for prompts while preserving high-signal constraints."""
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    out_lines: List[str] = []
    seen: set[str] = set()
    blank_pending = False
    for raw in normalized.split("\n"):
        line = " ".join(raw.strip().split())
        if not line:
            blank_pending = True
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        if blank_pending and out_lines and out_lines[-1] != "":
            out_lines.append("")
        blank_pending = False
        out_lines.append(line)
        if len(out_lines) >= max_lines:
            break
    compacted = "\n".join(out_lines).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 1].rstrip() + "…"


def markdown_focused_excerpt(
    path: str,
    heading_keywords: Sequence[str],
    max_chars: int,
    max_lines: int = 120,
) -> str:
    p = Path(path)
    if not p.exists():
        return f"Missing file: {path}"
    text = p.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(re.finditer(r"^##\s+(.+)$", text, flags=re.MULTILINE))
    if not matches:
        return compact_prompt_block(text, max_chars=max_chars, max_lines=max_lines)
    keywords = [k.lower() for k in heading_keywords]
    selected_chunks: List[str] = []
    for idx, match in enumerate(matches):
        heading = match.group(1).strip().lower()
        if not any(key in heading for key in keywords):
            continue
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        selected_chunks.append(text[start:end].strip())
    if not selected_chunks:
        return compact_prompt_block(text, max_chars=max_chars, max_lines=max_lines)
    merged = "\n\n".join(selected_chunks)
    return compact_prompt_block(merged, max_chars=max_chars, max_lines=max_lines)


def build_heuristics_context(max_chars: int = 4000) -> str:
    """Build a compact summary of hypotheses.json and failures.json for agent context.

    This gives the agent access to accumulated heuristics so it can make
    informed decisions grounded in prior knowledge, not just raw data.
    """
    lines: List[str] = []

    # --- Hypotheses summary ---
    hyp_path = REPO_ROOT / "agents" / "hypotheses.json"
    if hyp_path.exists():
        try:
            hyp_data = json.loads(hyp_path.read_text(encoding="utf-8"))
            hypotheses = hyp_data.get("hypotheses", [])
            lines.append(f"HYPOTHESES ({len(hypotheses)} total):")
            for h in hypotheses:
                hid = h.get("id", "?")
                title = h.get("title", "")
                status = h.get("status", "unknown")
                confidence = h.get("confidence", "unknown")
                n_results = len(h.get("simulation_results", []))
                statement = h.get("statement", "")[:120]
                lines.append(
                    f"- {hid}: [{status}/{confidence}] {title} ({n_results} results)"
                )
                lines.append(f"  claim: {statement}")
                # Show latest result verdict if any
                results = h.get("simulation_results", [])
                for r in results[-2:]:
                    if r.get("run_name"):
                        lines.append(
                            f"  last_evidence: run={r.get('run_name')} verdict={r.get('verdict')} "
                            f"notes={str(r.get('notes', ''))[:80]}"
                        )
        except (json.JSONDecodeError, KeyError):
            lines.append("HYPOTHESES: failed to parse hypotheses.json")
    else:
        lines.append("HYPOTHESES: hypotheses.json not found")

    lines.append("")

    # --- Failures summary ---
    fail_path = REPO_ROOT / "agents" / "failures.json"
    if fail_path.exists():
        try:
            fail_data = json.loads(fail_path.read_text(encoding="utf-8"))
            failures = fail_data.get("failures", [])
            lines.append(f"FAILURE MODES ({len(failures)} known):")
            for f_item in failures:
                fid = f_item.get("id", "?")
                title = f_item.get("title", "")
                severity = f_item.get("severity", "unknown")
                n_occurrences = len(f_item.get("occurrences", []))
                symptoms = f_item.get("symptoms", [])
                symptom_text = symptoms[0][:80] if symptoms else ""
                lines.append(
                    f"- {fid}: [{severity}] {title} ({n_occurrences} occurrences)"
                )
                if symptom_text:
                    lines.append(f"  symptom: {symptom_text}")
                # Show prevention hint
                prevention = f_item.get("prevention", [])
                if prevention:
                    lines.append(f"  prevent: {prevention[0][:80]}")
        except (json.JSONDecodeError, KeyError):
            lines.append("FAILURE MODES: failed to parse failures.json")
    else:
        lines.append("FAILURE MODES: failures.json not found")

    result = "\n".join(lines)
    return result[:max_chars]


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def normalize_text_list(value: object, max_items: int = 8) -> List[str]:
    items: List[str] = []
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, str):
                text = " ".join(entry.split())
                if text:
                    items.append(text)
    elif isinstance(value, str):
        for entry in value.splitlines():
            text = " ".join(entry.split())
            if text:
                items.append(text)
    return items[:max_items]


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


def default_initial_priority_plan(args: argparse.Namespace) -> Dict[str, object]:
    n_layouts = len(rs.parse_nc_library(args.nc_library))
    return {
        "mode": "deterministic",
        "priorities": [
            "Feasibility-first: reduce normalized_total_violation before maximizing productivity.",
            "Respect hard bounds and flow consistency: keep flows in configured bounds and treat raffinate as derived.",
            f"Pre-screen all {n_layouts} NC layouts by evidence and scientific prior before deep seed sweeps.",
            "Screen layouts quickly at medium fidelity, then validate top candidates at high fidelity.",
            f"Use solver stack {args.solver_name}/{args.linear_solver} and track termination_condition per run.",
            "Use provisional metrics only as direction signals; prefer validated metrics for ranking.",
        ],
        "proposed_simulations": [
            "Run each nc layout with the reference seed first to establish layout ranking under fixed conditions.",
            "Only then expand to non-reference seeds for top-ranked layouts.",
            "Perturb feed/desorbent/extract around best near-feasible point to reduce violation.",
            "Promote top candidates to high-fidelity validation.",
        ],
        "risks": [
            "Local infeasibility from tight purity/recovery constraints.",
            "Solver-status 'other' without usable primal variables.",
            "Bounds clipping on internal velocities when tstep/flows are inconsistent.",
        ],
        "nc_screening_strategy": [
            f"Screen all {n_layouts} NC layouts using the reference seed first, then expand seeds on top-ranked layouts.",
            "Use NC ranking criteria: prior closeness to reference, solver-error history, best violation, and runtime cost.",
        ],
    }


def initial_priority_plan(
    client: OpenAICompatClient,
    args: argparse.Namespace,
    objectives_excerpt: str,
    soul_excerpt: str,
    problem_definition_excerpt: str,
    skills_excerpt: str,
    codebase_excerpt: str,
    sqlite_excerpt: str,
    nc_strategy_excerpt: str,
    compute_context_excerpt: str,
    constraint_context_excerpt: str,
) -> Dict[str, object]:
    default_plan = default_initial_priority_plan(args)
    prompt_warning = ""
    try:
        objectives_compact = compact_prompt_block(objectives_excerpt, max_chars=2600, max_lines=70)
        soul_compact = compact_prompt_block(soul_excerpt, max_chars=1700, max_lines=55)
        problem_compact = compact_prompt_block(problem_definition_excerpt, max_chars=1600, max_lines=50)
        skills_compact = compact_prompt_block(skills_excerpt, max_chars=1400, max_lines=45)
        codebase_compact = compact_prompt_block(codebase_excerpt, max_chars=1400, max_lines=45)
        compute_compact = compact_prompt_block(compute_context_excerpt, max_chars=900, max_lines=35)
        constraint_compact = compact_prompt_block(constraint_context_excerpt, max_chars=1100, max_lines=45)
        sqlite_compact = compact_prompt_block(sqlite_excerpt, max_chars=1400, max_lines=55)
        nc_strategy_compact = compact_prompt_block(nc_strategy_excerpt, max_chars=1200, max_lines=40)
        prompt = textwrap.dedent(
            f"""
            You are generating the initial research plan for a two-scientist SMB campaign.
            Objective context:
            {objectives_compact}

            Scientist operating rules:
            {soul_compact}

            Problem framing context:
            {problem_compact}

            SMB physics context:
            {skills_compact}

            Codebase context:
            {codebase_compact}

            Runtime compute context:
            {compute_compact}

            Simulation objective/constraint context:
            {constraint_compact}

            Existing SQLite run history:
            {sqlite_compact}

            NC strategy board (screen all layouts before deep sweeps):
            {nc_strategy_compact}

            Requirements:
            - provide concrete strategy for screening all NC layouts in this library before deep seed exploration
            - reference compute budget explicitly
            - reference constraints explicitly

            Respond with JSON only:
            {{
              "priorities": ["..."],
              "proposed_simulations": ["..."],
              "risks": ["..."],
              "nc_screening_strategy": ["..."],
              "reason": "..."
            }}
            """
        ).strip()
    except Exception as exc:
        prompt_warning = f"Prompt build warning: {type(exc).__name__}: {exc}"
        prompt = (
            "You are generating an initial SMB research plan. Return JSON only with keys "
            "priorities, proposed_simulations, risks, nc_screening_strategy, reason.\n\n"
            f"Objective context:\n{objectives_excerpt}\n\n"
            f"Problem framing context:\n{problem_definition_excerpt}\n\n"
            f"SMB physics context:\n{skills_excerpt}\n\n"
            f"Runtime compute context:\n{compute_context_excerpt}\n\n"
            f"Constraint context:\n{constraint_context_excerpt}\n\n"
            f"NC strategy board:\n{nc_strategy_excerpt}\n\n"
            "Requirements: strategy must screen all NC layouts, reference compute budget, and explicit constraints."
        )
    raw = client.chat(
        "You are a principal SMB process scientist. Return JSON only.",
        prompt,
        conversation_role="initial_priority_plan",
        temperature=0.2,
        metadata={
            "phase": "planning",
            "solver_name": args.solver_name,
            "linear_solver": args.linear_solver,
            "nc_library": args.nc_library,
            "seed_library": args.seed_library,
        },
    )
    data = client.extract_json(raw)
    if not isinstance(data, dict):
        return default_plan
    priorities = normalize_text_list(data.get("priorities"), max_items=8)
    simulations = normalize_text_list(data.get("proposed_simulations"), max_items=8)
    risks = normalize_text_list(data.get("risks"), max_items=8)
    nc_screening_strategy = normalize_text_list(data.get("nc_screening_strategy"), max_items=10)
    if not priorities:
        return default_plan
    return {
        "mode": "llm",
        "priorities": priorities,
        "proposed_simulations": simulations or default_plan["proposed_simulations"],
        "risks": risks or default_plan["risks"],
        "nc_screening_strategy": nc_screening_strategy or default_plan["nc_screening_strategy"],
        "reason": str(data.get("reason", "")),
        "prompt_warning": prompt_warning,
        "raw": raw,
    }


def read_research_tail(path: Path, max_chars: int) -> str:
    if not path.exists():
        return "No research log yet."
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def append_research(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(section)


def reset_research_run_section(path: Path, run_name: str) -> None:
    if not path.exists():
        return
    marker = f"\n## Run: {run_name}\n"
    text = path.read_text(encoding="utf-8")
    idx = text.find(marker)
    if idx == -1:
        return
    path.write_text(text[:idx].rstrip() + "\n", encoding="utf-8")


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


def append_iteration_research(
    path: Path,
    iteration: int,
    task: Dict[str, object],
    a_note: Dict[str, object],
    b_note: Dict[str, object],
    scientist_a_proposed_task: Optional[Dict[str, object]] = None,
    effective_task_after_policy: Optional[Dict[str, object]] = None,
    scientist_b_reviewed_task: Optional[Dict[str, object]] = None,
    executive_note: Optional[Dict[str, object]] = None,
) -> None:
    proposed_task = scientist_a_proposed_task or task
    reviewed_task = scientist_b_reviewed_task or task
    lines = [
        f"\n### Search Iteration {iteration:02d}",
        f"- timestamp_utc: {utc_now_text()}",
        f"- candidate_nc: {task.get('nc')}",
        f"- candidate_seed: {task.get('seed_name')}",
        f"- scientist_a_proposed_task: {proposed_task}",
        f"- effective_task_after_policy: {effective_task_after_policy if isinstance(effective_task_after_policy, dict) else task}",
        f"- scientist_b_reviewed_task: {reviewed_task}",
        f"- scientist_a_reason: {a_note.get('reason')}",
        f"- scientist_a_mode: {a_note.get('mode')}",
        f"- scientist_a_llm_backend: {a_note.get('llm_backend', '')}",
        f"- scientist_b_decision: {b_note.get('decision')}",
        f"- scientist_b_reason: {b_note.get('reason')}",
        f"- scientist_b_mode: {b_note.get('mode')}",
        f"- scientist_b_llm_backend: {b_note.get('llm_backend', '')}",
    ]
    a_updates = normalize_text_list(a_note.get("priority_updates"), max_items=6)
    b_updates = normalize_text_list(b_note.get("priority_updates"), max_items=6)
    if a_updates or b_updates:
        lines.append("- priority_updates:")
        for item in a_updates + b_updates:
            lines.append(f"  - {item}")
    a_compare = normalize_text_list(a_note.get("comparison_to_previous"), max_items=8)
    if a_compare:
        lines.append("- scientist_a_comparison_to_previous:")
        for item in a_compare:
            lines.append(f"  - {item}")
    a_last_two = normalize_text_list(a_note.get("last_two_run_comparison"), max_items=6)
    if a_last_two:
        lines.append("- scientist_a_last_two_run_comparison:")
        for item in a_last_two:
            lines.append(f"  - {item}")
    a_flow = normalize_text_list(a_note.get("flowrate_comparison"), max_items=8)
    if a_flow:
        lines.append("- scientist_a_flowrate_comparison:")
        for item in a_flow:
            lines.append(f"  - {item}")
    a_deltas = normalize_text_list(a_note.get("delta_summary"), max_items=8)
    if a_deltas:
        lines.append("- scientist_a_delta_summary:")
        for item in a_deltas:
            lines.append(f"  - {item}")
    if str(a_note.get("physics_rationale", "")).strip():
        lines.append(f"- scientist_a_physics_rationale: {str(a_note.get('physics_rationale'))}")
    a_evidence = normalize_text_list(a_note.get("evidence"), max_items=8)
    if a_evidence:
        lines.append("- scientist_a_evidence:")
        for item in a_evidence:
            lines.append(f"  - {item}")
    a_nc_comp = normalize_text_list(a_note.get("nc_competitor_comparison"), max_items=8)
    if a_nc_comp:
        lines.append("- scientist_a_nc_competitor_comparison:")
        for item in a_nc_comp:
            lines.append(f"  - {item}")
    a_topology = normalize_text_list(a_note.get("column_topology_comparison"), max_items=8)
    if a_topology:
        lines.append("- scientist_a_column_topology_comparison:")
        for item in a_topology:
            lines.append(f"  - {item}")
    if str(a_note.get("diagnostic_hypothesis", "")).strip():
        lines.append(f"- scientist_a_diagnostic_hypothesis: {str(a_note.get('diagnostic_hypothesis'))}")
    a_fail = normalize_text_list(a_note.get("failure_criteria"), max_items=8)
    if a_fail:
        lines.append("- scientist_a_failure_criteria:")
        for item in a_fail:
            lines.append(f"  - {item}")
    b_compare = normalize_text_list(b_note.get("comparison_assessment"), max_items=8)
    if b_compare:
        lines.append("- scientist_b_comparison_assessment:")
        for item in b_compare:
            lines.append(f"  - {item}")
    b_last_two = normalize_text_list(b_note.get("last_two_run_audit"), max_items=6)
    if b_last_two:
        lines.append("- scientist_b_last_two_run_audit:")
        for item in b_last_two:
            lines.append(f"  - {item}")
    b_flow = normalize_text_list(b_note.get("flowrate_audit"), max_items=8)
    if b_flow:
        lines.append("- scientist_b_flowrate_audit:")
        for item in b_flow:
            lines.append(f"  - {item}")
    b_deltas = normalize_text_list(b_note.get("delta_audit"), max_items=8)
    if b_deltas:
        lines.append("- scientist_b_delta_audit:")
        for item in b_deltas:
            lines.append(f"  - {item}")
    if str(b_note.get("physics_audit", "")).strip():
        lines.append(f"- scientist_b_physics_audit: {str(b_note.get('physics_audit'))}")
    if isinstance(b_note.get("counterproposal_run"), dict):
        lines.append(f"- scientist_b_counterproposal_run: {b_note.get('counterproposal_run')}")
    b_nc_assess = normalize_text_list(b_note.get("nc_strategy_assessment"), max_items=8)
    if b_nc_assess:
        lines.append("- scientist_b_nc_strategy_assessment:")
        for item in b_nc_assess:
            lines.append(f"  - {item}")
    b_topology = normalize_text_list(b_note.get("column_topology_audit"), max_items=8)
    if b_topology:
        lines.append("- scientist_b_column_topology_audit:")
        for item in b_topology:
            lines.append(f"  - {item}")
    if str(b_note.get("compute_assessment", "")).strip():
        lines.append(f"- scientist_b_compute_assessment: {str(b_note.get('compute_assessment'))}")
    b_counter = normalize_text_list(b_note.get("counterarguments"), max_items=8)
    if b_counter:
        lines.append("- scientist_b_counterarguments:")
        for item in b_counter:
            lines.append(f"  - {item}")
    b_risks = normalize_text_list(b_note.get("risk_flags"), max_items=8)
    if b_risks:
        lines.append("- scientist_b_risk_flags:")
        for item in b_risks:
            lines.append(f"  - {item}")
    b_checks = normalize_text_list(b_note.get("required_checks"), max_items=8)
    if b_checks:
        lines.append("- scientist_b_required_checks:")
        for item in b_checks:
            lines.append(f"  - {item}")
    if isinstance(executive_note, dict):
        lines.append(f"- executive_decision: {executive_note.get('decision')}")
        lines.append(f"- executive_reason: {executive_note.get('reason')}")
        if executive_note.get("forced_task"):
            lines.append(f"- executive_forced_task: {executive_note.get('forced_task')}")
        if executive_note.get("forced_reason"):
            lines.append(f"- executive_forced_reason: {executive_note.get('forced_reason')}")
        e_updates = normalize_text_list(executive_note.get("priority_updates"), max_items=8)
        if e_updates:
            lines.append("- executive_priority_updates:")
            for item in e_updates:
                lines.append(f"  - {item}")
    append_research(path, "\n".join(lines) + "\n")


def append_result_research(path: Path, result: Dict[str, object], phase: str) -> None:
    flow = effective_flow(result) or {}
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        provisional = result.get("provisional")
        metrics = provisional.get("metrics") if isinstance(provisional, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
    slacks = result.get("constraint_slacks")
    if not isinstance(slacks, dict):
        slacks = {}

    lines = [
        f"- {phase}_result_run: {result.get('run_name')}",
        f"  - status: {result.get('status')}",
        f"  - feasible: {result.get('feasible')}",
        f"  - termination: {(result.get('solver') or {}).get('termination_condition') if isinstance(result.get('solver'), dict) else ''}",
        f"  - productivity_ex_ga_ma: {metrics.get('productivity_ex_ga_ma')}",
        f"  - purity_ex_meoh_free: {metrics.get('purity_ex_meoh_free')}",
        f"  - recovery_ex_GA: {metrics.get('recovery_ex_GA')}",
        f"  - recovery_ex_MA: {metrics.get('recovery_ex_MA')}",
        f"  - normalized_total_violation: {slacks.get('normalized_total_violation')}",
        f"  - flow: {flow}",
    ]
    execution_policy = result.get("execution_policy")
    if isinstance(execution_policy, dict):
        lines.append(f"  - execution_policy_note: {execution_policy.get('note')}")
        lines.append(f"  - execution_policy_fidelity_override: {execution_policy.get('fidelity_override')}")
    comp = composition_metrics_from_result(result)
    if comp is not None:
        lines.append(
            "  - composition_ce_cr: "
            + f"CE_acid={comp.get('ce_acid')} CE_water={comp.get('ce_water')} CE_meoh={comp.get('ce_meoh')} "
            + f"CR_acid={comp.get('cr_acid')} CR_water={comp.get('cr_water')} CR_meoh={comp.get('cr_meoh')} "
            + f"source={comp.get('source')}"
        )
    append_research(path, "\n".join(lines) + "\n")


def append_final_research(
    path: Path,
    best_result: Optional[Dict[str, object]],
    ranked_search: List[Dict[str, object]],
    ranked_validation: List[Dict[str, object]],
) -> None:
    lines = [
        "\n### Run Closing Summary",
        f"- finished_utc: {utc_now_text()}",
        f"- best_result: {best_result.get('run_name') if isinstance(best_result, dict) else 'none'}",
        f"- best_status: {best_result.get('status') if isinstance(best_result, dict) else 'n/a'}",
        f"- search_results_count: {len(ranked_search)}",
        f"- validation_results_count: {len(ranked_validation)}",
        "",
        "### Proposed Next Simulations",
    ]
    if ranked_search:
        top = ranked_search[:3]
        for item in top:
            flow = effective_flow(item) or {}
            lines.append(
                f"- Probe around run={item.get('run_name')} nc={item.get('nc')} "
                f"with +/- small perturbations on Ffeed/Fdes/Fex while preserving flow consistency. Base flow={flow}"
            )
    else:
        lines.append("- Re-run layout/seed screening with broader nc and notebook seeds; no valid search record yet.")
    append_research(path, "\n".join(lines) + "\n")


def merge_priority_board(current: List[str], *notes: Dict[str, object]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for item in current:
        text = " ".join(str(item).split())
        if text and text not in seen:
            seen.add(text)
            merged.append(text)
    for note in notes:
        for item in normalize_text_list(note.get("priority_updates"), max_items=8):
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged[:16]


def safe_result_metric(result: Dict[str, object], key: str) -> Optional[float]:
    if result.get("status") == "ok":
        metrics = result.get("metrics") or {}
        if key in metrics:
            return float(metrics[key])  # type: ignore[arg-type]
    provisional = result.get("provisional") or {}
    metrics = provisional.get("metrics") or {}
    if key in metrics:
        return float(metrics[key])  # type: ignore[arg-type]
    return None


def effective_flow(result: Dict[str, object]) -> Optional[Dict[str, float]]:
    for key in ("optimized_flow", "provisional_optimized_flow", "initial_flow", "flow"):
        value = result.get(key)
        if isinstance(value, dict):
            return {k: float(v) for k, v in value.items()}
    return None


def stream_components_from_outlets(outlets: Dict[str, object], stream_key: str) -> Optional[Dict[str, float]]:
    values = outlets.get(stream_key)
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return None
    comps = [as_float(values[i]) for i in range(4)]
    if any(v is None for v in comps):
        return None
    return {
        "GA": float(comps[0]),
        "MA": float(comps[1]),
        "Water": float(comps[2]),
        "MeOH": float(comps[3]),
    }


def composition_metrics_from_result(result: Dict[str, object]) -> Optional[Dict[str, object]]:
    source = "validated"
    outlets_obj = result.get("outlets")
    if not isinstance(outlets_obj, dict):
        provisional = result.get("provisional")
        if isinstance(provisional, dict) and isinstance(provisional.get("outlets"), dict):
            outlets_obj = provisional.get("outlets")
            source = "provisional"
        else:
            return None
    outlets = outlets_obj  # narrowed to dict
    ce = stream_components_from_outlets(outlets, "CE")
    cr = stream_components_from_outlets(outlets, "CR")
    if ce is None or cr is None:
        return None
    return {
        "source": source,
        "ce_acid": ce["GA"] + ce["MA"],
        "ce_water": ce["Water"],
        "ce_meoh": ce["MeOH"],
        "cr_acid": cr["GA"] + cr["MA"],
        "cr_water": cr["Water"],
        "cr_meoh": cr["MeOH"],
    }


def composition_metrics_from_raw_json(raw_json: str) -> Optional[Dict[str, object]]:
    if not raw_json:
        return None
    try:
        payload = json.loads(raw_json)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return composition_metrics_from_result(payload)


def linear_slope(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    if var_x <= 1e-12:
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov_xy / var_x


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


def effective_violation(result: Dict[str, object]) -> float:
    slacks = result.get("constraint_slacks")
    if isinstance(slacks, dict) and "normalized_total_violation" in slacks:
        return float(slacks["normalized_total_violation"])
    provisional = result.get("provisional")
    if isinstance(provisional, dict):
        provisional_slacks = provisional.get("constraint_slacks")
        if isinstance(provisional_slacks, dict) and "normalized_total_violation" in provisional_slacks:
            return float(provisional_slacks["normalized_total_violation"])
        metrics = provisional.get("metrics") or {}
        if isinstance(metrics, dict):
            inferred = inferred_violation_from_metrics(metrics)
            if inferred is not None:
                return inferred
            return -float(metrics.get("productivity_ex_ga_ma", 0.0))
    return 1e9


def search_score(result: Dict[str, object]) -> Tuple[int, float, float]:
    feasible = 1 if result.get("feasible") else 0
    productivity = safe_result_metric(result, "productivity_ex_ga_ma") or float("-inf")
    violation = effective_violation(result)
    return feasible, productivity, -violation


def summarize_result(result: Dict[str, object]) -> str:
    flow = effective_flow(result) or {}
    productivity = safe_result_metric(result, "productivity_ex_ga_ma")
    purity = safe_result_metric(result, "purity_ex_meoh_free")
    rga = safe_result_metric(result, "recovery_ex_GA")
    rma = safe_result_metric(result, "recovery_ex_MA")
    return (
        f"run={result.get('run_name')} nc={result.get('nc')} status={result.get('status')} "
        f"feasible={result.get('feasible')} "
        f"prod={productivity} purity={purity} rGA={rga} rMA={rma} "
        f"flow={flow}"
    )


def recent_two_run_review_context(results: List[Dict[str, object]]) -> Tuple[str, List[str]]:
    if not results:
        return "none", []
    recent = results[-2:]
    labels: List[str] = []
    lines: List[str] = []
    for idx, item in enumerate(recent, start=1):
        label = f"R-{idx}"
        labels.append(label)
        termination = ""
        solver = item.get("solver")
        if isinstance(solver, dict):
            termination = str(solver.get("termination_condition", ""))
        lines.append(
            f"- {label}: run_name={item.get('run_name')} nc={item.get('nc')} status={item.get('status')} "
            f"termination={termination} feasible={item.get('feasible')} "
            f"prod={safe_result_metric(item, 'productivity_ex_ga_ma')} "
            f"purity={safe_result_metric(item, 'purity_ex_meoh_free')} "
            f"rGA={safe_result_metric(item, 'recovery_ex_GA')} "
            f"rMA={safe_result_metric(item, 'recovery_ex_MA')} "
            f"viol={effective_violation(item)}"
        )
    return "\n".join(lines), labels


def deterministic_select(tasks: List[Dict[str, object]], tried: set[Tuple[Tuple[int, ...], str]]) -> int:
    for idx, task in enumerate(tasks):
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key not in tried:
            return idx
    return 0


def is_reference_seed_name(seed_name: object) -> bool:
    return str(seed_name or "").strip().lower() == "reference"


def low_fidelity_limits(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "nfex": max(1, int(getattr(args, "finalization_low_fidelity_nfex", getattr(args, "probe_nfex", 5)))),
        "nfet": max(1, int(getattr(args, "finalization_low_fidelity_nfet", getattr(args, "probe_nfet", 2)))),
        "ncp": max(1, int(getattr(args, "finalization_low_fidelity_ncp", getattr(args, "probe_ncp", 1)))),
    }


def fidelity_triplet(result: Dict[str, object]) -> Optional[Tuple[int, int, int]]:
    fidelity = result.get("fidelity")
    if not isinstance(fidelity, dict):
        return None
    try:
        return int(fidelity.get("nfex", 0)), int(fidelity.get("nfet", 0)), int(fidelity.get("ncp", 0))
    except Exception:
        return None


def is_low_fidelity_result(result: Dict[str, object], args: argparse.Namespace) -> bool:
    triplet = fidelity_triplet(result)
    if triplet is None:
        return False
    limits = low_fidelity_limits(args)
    return triplet[0] <= limits["nfex"] and triplet[1] <= limits["nfet"] and triplet[2] <= limits["ncp"]


def has_metric_evidence(result: Dict[str, object]) -> bool:
    status = str(result.get("status", "")).strip().lower()
    if status in {"ok", "solver_error"}:
        return True
    return (
        safe_result_metric(result, "purity_ex_meoh_free") is not None
        or safe_result_metric(result, "recovery_ex_GA") is not None
        or safe_result_metric(result, "productivity_ex_ga_ma") is not None
    )


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


def reference_probe_runs_completed(results: List[Dict[str, object]]) -> int:
    return sum(1 for item in results if is_reference_seed_name(item.get("seed_name")))


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


def apply_probe_reference_gate(
    args: argparse.Namespace,
    tasks: List[Dict[str, object]],
    tried: set[Tuple[Tuple[int, ...], str]],
    search_results: List[Dict[str, object]],
    requested_idx: int,
) -> Tuple[int, Optional[Dict[str, object]]]:
    min_required = max(0, int(getattr(args, "min_probe_reference_runs", 0)))
    if min_required <= 0:
        return requested_idx, None

    total_reference_tasks = len(ranked_reference_indices(tasks))
    if total_reference_tasks <= 0:
        return requested_idx, None

    required = min(min_required, total_reference_tasks)
    completed = reference_probe_runs_completed(search_results)
    if completed >= required:
        return requested_idx, None

    requested_task = tasks[requested_idx]
    if is_reference_seed_name(requested_task.get("seed_name")):
        return requested_idx, None

    forced_idx = first_untried_reference_index(tasks, tried)
    if forced_idx is None:
        return requested_idx, {
            "applied": False,
            "reason": (
                f"Probe gate active ({completed}/{required}) but no untried reference task remains; "
                "cannot enforce further reference probing."
            ),
            "completed_reference_runs": completed,
            "required_reference_runs": required,
        }

    forced_task = tasks[forced_idx]
    return forced_idx, {
        "applied": True,
        "reason": (
            f"Probe gate enforced: completed_reference_runs={completed}/{required}. "
            f"Blocked non-reference seed '{requested_task.get('seed_name')}' and forced reference probe."
        ),
        "completed_reference_runs": completed,
        "required_reference_runs": required,
        "requested_task": requested_task,
        "forced_task": forced_task,
    }


def probe_reference_runs_required(args: argparse.Namespace, tasks: List[Dict[str, object]]) -> int:
    total_reference_tasks = len(ranked_reference_indices(tasks))
    if total_reference_tasks <= 0:
        return 0
    return min(max(0, int(getattr(args, "min_probe_reference_runs", 0))), total_reference_tasks)


def search_execution_policy(
    args: argparse.Namespace,
    tasks: List[Dict[str, object]],
    search_results: List[Dict[str, object]],
    task: Dict[str, object],
) -> Dict[str, object]:
    required = probe_reference_runs_required(args, tasks)
    completed = reference_probe_runs_completed(search_results)
    low_fidelity_enabled = bool(int(getattr(args, "probe_low_fidelity_enabled", 1)))
    probe_phase_active = required > 0 and completed < required

    policy: Dict[str, object] = {
        "probe_phase_active": probe_phase_active,
        "completed_reference_runs": completed,
        "required_reference_runs": required,
        "low_fidelity_enabled": low_fidelity_enabled,
    }
    if not probe_phase_active:
        if not bool(int(getattr(args, "finalization_hard_gate_enabled", 1))):
            return policy
        if is_reference_seed_name(task.get("seed_name")):
            return policy
        nc = tuple(task.get("nc", []))
        if has_low_fidelity_optimization_evidence_for_nc(args, search_results, nc):
            return policy
        limits = low_fidelity_limits(args)
        policy["fidelity_override"] = {
            "nfex": limits["nfex"],
            "nfet": limits["nfet"],
            "ncp": limits["ncp"],
        }
        policy["reason"] = (
            "Finalization hard gate precheck: forcing first non-reference optimization for this NC "
            f"to low-fidelity (nfex={limits['nfex']}, nfet={limits['nfet']}, ncp={limits['ncp']}) "
            "before expensive final optimization is allowed."
        )
        return policy
    if not low_fidelity_enabled:
        policy["reason"] = "Probe phase active, but low-fidelity override is disabled."
        return policy
    if not is_reference_seed_name(task.get("seed_name")):
        policy["reason"] = "Probe phase active, waiting for required reference runs before non-reference seeds."
        return policy

    policy["fidelity_override"] = {
        "nfex": max(1, int(getattr(args, "probe_nfex", 5))),
        "nfet": max(1, int(getattr(args, "probe_nfet", 2))),
        "ncp": max(1, int(getattr(args, "probe_ncp", 1))),
    }
    policy["reason"] = (
        f"Probe phase reference run {completed + 1}/{required}: forcing low-fidelity "
        f"(nfex={policy['fidelity_override']['nfex']}, "
        f"nfet={policy['fidelity_override']['nfet']}, "
        f"ncp={policy['fidelity_override']['ncp']})."
    )
    return policy


def has_any_feasible(results: List[Dict[str, object]]) -> bool:
    return any(bool(item.get("feasible")) for item in results)


def ranked_reference_indices(tasks: List[Dict[str, object]]) -> List[int]:
    return [idx for idx, task in enumerate(tasks) if str(task.get("seed_name", "")).strip().lower() == "reference"]


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


def executive_controller_decide(
    args: argparse.Namespace,
    tasks: List[Dict[str, object]],
    tried: set[Tuple[Tuple[int, ...], str]],
    candidate_idx: int,
    candidate_task: Dict[str, object],
    b_note: Dict[str, object],
    search_results: List[Dict[str, object]],
    consecutive_rejects: int,
    debate_round: int = 0,
) -> Dict[str, object]:
    """
    Enhanced Executive Controller with immediate decision-making and debate round limits.
    
    Args:
        debate_round: Current debate round (0 = initial decision, 1 = first debate, 2 = final round)
    
    Returns:
        Executive decision with immediate action or debate continuation directive
    """
    decision = str(b_note.get("decision", "")).lower()
    
    # Immediate decision after Scientist B judgment
    if decision == "approve":
        return {
            "decision": "not_needed",
            "reason": "Scientist_B approved candidate; executive override not needed.",
            "priority_updates": [],
            "immediate_action": True,
            "debate_round": debate_round,
        }
    
    if not bool(args.executive_controller_enabled):
        return {
            "decision": "disabled",
            "reason": "Executive controller disabled by configuration.",
            "priority_updates": [],
            "immediate_action": True,
            "debate_round": debate_round,
        }
    
    # Check if we've reached maximum debate rounds
    if debate_round >= 2:
        return {
            "decision": "final_decision",
            "reason": f"Maximum debate rounds ({debate_round}) reached. Making final executive decision.",
            "priority_updates": ["Maximum debate rounds exhausted - executive must decide now."],
            "immediate_action": True,
            "debate_round": debate_round,
            "max_debates_reached": True,
        }
    
    # If feasible baseline exists, respect Scientist B's rejection
    if has_any_feasible(search_results):
        return {
            "decision": "respect_reject",
            "reason": "Feasible baseline exists; keep scientist rejection in effect.",
            "priority_updates": [],
            "immediate_action": True,
            "debate_round": debate_round,
        }
    
    # Check consecutive rejection conditions
    if consecutive_rejects < int(args.executive_trigger_rejects):
        return {
            "decision": "respect_reject",
            "reason": f"Consecutive rejects={consecutive_rejects} below trigger={int(args.executive_trigger_rejects)}.",
            "priority_updates": [],
            "immediate_action": True,
            "debate_round": debate_round,
        }
    
    if consecutive_rejects < int(args.executive_force_after_rejects):
        return {
            "decision": "respect_reject",
            "reason": (
                f"Consecutive rejects reached trigger ({consecutive_rejects} >= {int(args.executive_trigger_rejects)}), "
                f"but below force_after={int(args.executive_force_after_rejects)}."
            ),
            "priority_updates": [
                "Executive warning: next reject may force top-priority diagnostic execution."
            ],
            "immediate_action": True,
            "debate_round": debate_round,
        }
    
    # Executive override conditions met - force execution
    forced_idx, forced_reason = executive_forced_index(tasks, tried, int(args.executive_top_k_lock))
    forced_task = tasks[forced_idx]
    forced_key = (tuple(forced_task["nc"]), str(forced_task["seed_name"]))
    
    if forced_key in tried:
        return {
            "decision": "respect_reject",
            "reason": "No untried executive-forced task available; respecting rejection.",
            "priority_updates": [],
            "immediate_action": True,
            "debate_round": debate_round,
        }
    
    return {
        "decision": "override_execute",
        "reason": (
            f"Hard controller override: no feasible baseline and consecutive rejects={consecutive_rejects} "
            f"(trigger={int(args.executive_trigger_rejects)}). Force execution of top-priority reference candidate."
        ),
        "forced_candidate_index": forced_idx,
        "forced_task": forced_task,
        "forced_reason": forced_reason,
        "priority_updates": [
            "Executive override executed to break reject loop and establish feasibility baseline.",
            "Run top-ranked reference candidates before additional NC rotation.",
        ],
        "immediate_action": True,
        "debate_round": debate_round,
        "executive_override_executed": True,
    }


def deterministic_review(candidate: Dict[str, object], best_result: Optional[Dict[str, object]]) -> Dict[str, object]:
    if best_result and candidate["nc"] == best_result.get("nc") and candidate["seed_name"] == best_result.get("seed_name"):
        return {
            "decision": "reject",
            "reason": "Already evaluated this layout and seed.",
            "comparison_assessment": [
                f"Compared against best prior run {best_result.get('run_name')} with same nc/seed; this would be a duplicate."
            ],
            "nc_strategy_assessment": [
                "Candidate does not improve NC coverage because this nc/seed pair is already evaluated."
            ],
            "compute_assessment": "Reject duplicate to preserve budget for unexplored NC layouts and seeds.",
            "priority_updates": ["Avoid duplicate nc/seed evaluations unless bounds or fidelity changed."],
            "counterarguments": ["No new evidence is provided for a duplicate nc/seed attempt."],
            "risk_flags": ["Wasted budget on duplicate search point."],
            "required_checks": ["Only retry duplicates when bounds/fidelity or solver stack changed."],
        }
    return {
        "decision": "approve",
        "reason": "Candidate is within current bounds and still untested.",
        "comparison_assessment": [
            "Compared candidate against tried set and current best run; this nc/seed has not been executed yet."
        ],
        "nc_strategy_assessment": [
            "Candidate expands NC/seed evidence coverage and can improve ranking confidence across layout alternatives."
        ],
        "compute_assessment": "Approve as a bounded, untried point with acceptable incremental budget impact.",
        "priority_updates": ["Continue feasibility-first screening, then rank by productivity among low-violation runs."],
        "counterarguments": ["Approval is provisional until solver status and post-check metrics are reviewed."],
        "risk_flags": ["Potential local infeasibility despite bounded flows."],
        "required_checks": ["Confirm effective post-bounds flow vector and solver termination condition."],
    }


def single_scientist_policy_review(candidate: Dict[str, object], best_result: Optional[Dict[str, object]]) -> Dict[str, object]:
    review = deterministic_review(candidate, best_result)
    review = dict(review)
    review["mode"] = "single_scientist_policy"
    review["reason"] = (
        "Scientist_B bypassed by single-scientist mode. "
        + str(review.get("reason", "")).strip()
    ).strip()
    updates = normalize_text_list(review.get("priority_updates"), max_items=8)
    updates.append("Single-scientist mode active: using deterministic policy gate instead of LLM review.")
    review["priority_updates"] = normalize_text_list(updates, max_items=8)
    return review


def rank_any_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked = sorted(results, key=search_score, reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank_any"] = idx
    return ranked


def build_search_tasks(args: argparse.Namespace) -> List[Dict[str, object]]:
    nc_library = rs.parse_nc_library(args.nc_library)
    nc_library = sorted(nc_library, key=nc_prior_score, reverse=True)
    seed_library = rs.parse_seed_library(args.seed_library)
    if not seed_library:
        return []

    reference_idx = 0
    for idx, seed in enumerate(seed_library):
        if str(seed.get("name", "")).strip().lower() == "reference":
            reference_idx = idx
            break
    reference_seed = seed_library[reference_idx]
    remaining_seeds = [seed for i, seed in enumerate(seed_library) if i != reference_idx]

    tasks: List[Dict[str, object]] = []
    # Pass 1: cover all layouts with the reference seed first.
    for nc in nc_library:
        tasks.append({"nc": list(nc), "seed_name": str(reference_seed["name"]), "seed": reference_seed})
    # Pass 2: deepen with non-reference seeds on the same ranked layout order.
    for seed in remaining_seeds:
        for nc in nc_library:
            tasks.append({"nc": list(nc), "seed_name": str(seed["name"]), "seed": seed})
    return tasks


def scientist_a_pick(
    client: OpenAICompatClient,
    candidate_tasks: List[Dict[str, object]],
    results: List[Dict[str, object]],
    tried: set[Tuple[Tuple[int, ...], str]],
    args: argparse.Namespace,
    objectives_excerpt: str,
    soul_excerpt: str,
    codebase_context_excerpt: str,
    compute_context_excerpt: str,
    constraint_context_excerpt: str,
    nc_strategy_excerpt: str,
    research_excerpt: str,
    current_priorities: List[str],
    sqlite_context_excerpt: str,
    budget_used: float,
    iteration: int,
    heuristics_context: str = "",
    convergence_context: str = "",
) -> Tuple[int, Dict[str, object]]:
    remaining = [task for task in candidate_tasks if (tuple(task["nc"]), str(task["seed_name"])) not in tried]
    shortlist = remaining[: min(len(remaining), 8)]
    default_index = deterministic_select(candidate_tasks, tried)
    if not shortlist:
        return default_index, {"mode": "deterministic", "reason": "No remaining tasks."}

    best = rank_any_results(results)[0] if results else None
    recent_two_block, recent_two_labels = recent_two_run_review_context(results)
    prompt_warning = ""
    try:
        prompt = textwrap.dedent(
            f"""
            You are Scientist_A for an SMB optimization benchmark.
            Think aggressively and evidence-first. Do not give generic plans.
            Every proposal must be triple-grounded: DATA (SQLite history, convergence tracker) + PHYSICS (zone theory, mass balance) + HEURISTICS (hypotheses.json, failures.json patterns).
            Before choosing a new experiment, you must compare it against previous results (at minimum: current best and one recent failed run).
            If evidence is weak, propose a diagnostic run and state why.
            Each simulation is expensive. Your competitive advantage over brute-force MINLP is choosing the HIGHEST-VALUE next experiment. Justify why this candidate is the most informative use of the next solver call.

            Objective summary:
            {objectives_excerpt}

            Scientist rules summary:
            {soul_excerpt}

            Codebase context summary:
            {codebase_context_excerpt}

            Runtime compute context:
            {compute_context_excerpt}

            Simulation objective/constraint context:
            {constraint_context_excerpt}

            Accumulated heuristics (hypotheses and known failure modes):
            {heuristics_context}

            Convergence progress:
            {convergence_context}

            Current research log tail:
            {research_excerpt}

            NC strategy board (all layouts in current library):
            {nc_strategy_excerpt}

            Current priority board:
            {json.dumps(current_priorities, indent=2)}

            Historical simulation context (queried from SQLite):
            {sqlite_context_excerpt}

            Counted benchmark budget is {args.benchmark_hours:.1f} SMB hours with {args.search_hours:.1f} search hours and {args.validation_hours:.1f} validation hours.
            Search wall-hours used so far: {budget_used:.4f}
            Hard policy: complete at least {int(getattr(args, "min_probe_reference_runs", 0))} reference-seed probe runs before proposing non-reference seed optimization.
            Hard policy: final high-fidelity optimization is allowed only after low-fidelity reference and low-fidelity non-reference optimization evidence exist on the same NC.

            Current best result:
            {summarize_result(best) if best else "None yet."}

            Recent two completed runs (must be reviewed deeply when available):
            {recent_two_block}

            Required rigor:
            - compare candidate NC against at least two alternative NC layouts from the strategy board
            - compare candidate against previous result evidence (current best + recent failure when available)
            - include quantitative metric evidence in comparisons (at least one of: productivity, purity, recovery, violation, feasible/J)
            - when two prior runs exist, include explicit deep comparison to BOTH R-1 and R-2 using run_name and metric deltas
            - include explicit flowrate comparisons (Ffeed/F1/Fdes/Fex/Fraf/tstep) across at least two prior runs
            - include explicit delta vectors using this schema token style: Δprod, Δpurity, ΔrGA, ΔrMA, Δviol, ΔFfeed, ΔF1, ΔFdes, ΔFex, ΔFraf, Δtstep
            - include explicit column topology comparison (nc and zone-column deltas, e.g., ΔZ1..ΔZ4) against R-1, R-2, and one competitor
            - include physics-based rationale (mass balance, zone allocation effects, adsorption/desorption/selectivity), not rank-only claims
            - include explicit compute/budget impact and stopping/failure criteria

            Acquisition strategy requirement (MANDATORY):
            - classify this proposal as exactly one of: EXPLORE, EXPLOIT, or VERIFY
            - state what this run will teach that we don't already know (information_target)
            - list at least 2 alternative candidates considered and why they were rejected
            - identify the coverage gap this fills (untested NC, unexplored flow region, untested hypothesis)
            - reference at least one hypothesis from hypotheses.json that this run tests or one failure mode from failures.json that it risks
            - assess convergence: are we improving? stagnating? should we shift strategy?

            Remaining candidate shortlist:
            {json.dumps(shortlist, indent=2)}

            Respond with JSON only:
            {{
              "candidate_index": <0-based index into shortlist>,
              "reason": "<brief reason>",
              "acquisition_type": "EXPLORE | EXPLOIT | VERIFY",
              "information_target": "<what will this run teach us that we don't already know?>",
              "alternatives_considered": ["<candidate X rejected because...>", "<candidate Y rejected because...>"],
              "coverage_gap": "<what untested NC / flow region / hypothesis does this fill?>",
              "hypothesis_connection": "<which hypothesis ID from hypotheses.json does this test, or which failure mode ID does it risk?>",
              "convergence_assessment": "<are we improving? stagnating? should we shift strategy?>",
              "evidence": ["<specific evidence item>", "..."],
              "comparison_to_previous": ["<explicit comparison to named prior run with metric/termination evidence>", "..."],
              "last_two_run_comparison": ["<R-1: run_name + metrics + what changed>", "<R-2: run_name + metrics + what changed>"],
              "flowrate_comparison": ["<flow deltas across runs with Ffeed/F1/Fdes/Fex/Fraf/tstep and implication>", "..."],
              "delta_summary": ["<vs R-1: Δprod=..., Δpurity=..., ΔrGA=..., ΔrMA=..., Δviol=..., ΔFfeed=..., ΔF1=..., ΔFdes=..., ΔFex=..., ΔFraf=..., Δtstep=...>", "<vs R-2: ...>", "<vs competitor nc=[...]: Δprod=..., Δpurity=..., ΔrGA=..., ΔrMA=..., Δviol=...>"],
              "column_topology_comparison": ["<vs R-1: nc=[a,b,c,d] -> candidate=[...], ΔZ1=..., ΔZ2=..., ΔZ3=..., ΔZ4=... and expected impact>", "<vs R-2: ...>", "<vs competitor nc=[...]: topology/zone tradeoff>"],
              "physics_rationale": "<physics-based explanation using zones, flow splits, mass-transfer or mass-balance logic>",
              "nc_competitor_comparison": ["<candidate nc vs two alternatives with rationale>", "..."],
              "diagnostic_hypothesis": "<what this run is testing>",
              "failure_criteria": ["<what would make this a bad proposal>", "..."],
              "fidelity": "medium",
              "priority_updates": ["..."],
              "proposed_followups": ["..."]
            }}
            """
        ).strip()
    except Exception as exc:
        prompt_warning = f"Prompt build warning: {type(exc).__name__}: {exc}"
        prompt = (
            "You are Scientist_A for SMB optimization. Return JSON only.\n\n"
            f"Current best result: {summarize_result(best) if best else 'None yet.'}\n"
            f"Recent two completed runs:\n{recent_two_block}\n\n"
            f"Remaining candidate shortlist:\n{json.dumps(shortlist, indent=2)}\n\n"
            "Respond with keys: candidate_index, reason, evidence, comparison_to_previous, "
            "last_two_run_comparison, flowrate_comparison, delta_summary, column_topology_comparison, physics_rationale, nc_competitor_comparison, diagnostic_hypothesis, failure_criteria, fidelity, priority_updates, proposed_followups."
        )
    raw = client.chat(
        "You are an aggressive optimization scientist. Return JSON only and ground claims in evidence.",
        prompt,
        conversation_role="scientist_a_pick",
        temperature=0.2,
        metadata={
            "iteration": iteration,
            "search_hours_used": budget_used,
            "shortlist_size": len(shortlist),
            "remaining_count": len(remaining),
            "tried_count": len(tried),
        },
    )
    data = client.extract_json(raw)
    if data and isinstance(data.get("candidate_index"), int):
        idx = int(data["candidate_index"])
        if 0 <= idx < len(shortlist):
            evidence = normalize_text_list(data.get("evidence"), max_items=8)
            comparisons = normalize_text_list(data.get("comparison_to_previous"), max_items=8)
            last_two_comparisons = normalize_text_list(data.get("last_two_run_comparison"), max_items=4)
            flow_comparisons = normalize_text_list(data.get("flowrate_comparison"), max_items=6)
            delta_summary = normalize_text_list(data.get("delta_summary"), max_items=8)
            topology_comparisons = normalize_text_list(data.get("column_topology_comparison"), max_items=8)
            physics_rationale = str(data.get("physics_rationale", "")).strip()
            nc_comparisons = normalize_text_list(data.get("nc_competitor_comparison"), max_items=8)
            has_history = (len(results) > 0) or (sqlite_total_records_from_excerpt(sqlite_context_excerpt) > 0)
            if len(evidence) < 2:
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: missing minimum evidence detail.",
                    "priority_updates": [
                        "Require at least two concrete evidence items (history/constraints/compute/signals) before proposing experiments."
                    ],
                }
            if not comparisons:
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: missing required comparison to previous results.",
                    "priority_updates": [
                        "Require explicit comparison against prior runs (best and recent failures) before proposing new experiments."
                    ],
                }
            if has_history and not text_mentions_prior_runs(comparisons):
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: comparison text does not cite concrete prior-run evidence.",
                    "priority_updates": [
                        "Require run-level evidence (run name/status/violation/productivity) in comparison-to-previous."
                    ],
                }
            if has_history and not text_mentions_metric_signals(comparisons):
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: comparison text is not metric-grounded.",
                    "priority_updates": [
                        "Require quantitative metrics (productivity/purity/recovery/violation/feasible/J) in comparison-to-previous."
                    ],
                }
            if len(recent_two_labels) >= 2:
                if len(last_two_comparisons) < 2:
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: missing required deep comparison of last two completed runs.",
                        "priority_updates": [
                            "Require explicit R-1 and R-2 comparison entries with run-level metrics before proposing new experiments."
                        ],
                    }
                if not text_mentions_required_labels(last_two_comparisons, recent_two_labels):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: last-two comparison does not reference both required run labels (R-1 and R-2).",
                        "priority_updates": [
                            "Require both R-1 and R-2 references in last-two comparison block."
                        ],
                    }
                if not text_mentions_metric_signals(last_two_comparisons) or not text_mentions_numeric_values(last_two_comparisons):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: last-two comparison is not metric- and number-grounded.",
                        "priority_updates": [
                            "Require numeric metric evidence (prod/purity/recovery/violation) in R-1 and R-2 analysis."
                        ],
                    }
                if not text_mentions_run_name_signals(last_two_comparisons):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: last-two comparison must cite explicit run names.",
                        "priority_updates": [
                            "Require run_name-level evidence in R-1/R-2 deep comparison block."
                        ],
                    }
                if len(delta_summary) < 3:
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: missing required delta summary for R-1, R-2, and competitor.",
                        "priority_updates": [
                            "Require explicit delta summary entries for both prior runs and at least one competitor NC."
                        ],
                    }
                if not text_mentions_required_labels(delta_summary, recent_two_labels):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: delta summary must reference both R-1 and R-2.",
                        "priority_updates": [
                            "Require R-1 and R-2 labels in delta summary block."
                        ],
                    }
                if not text_mentions_delta_metric_signals(delta_summary):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: delta summary must include Δprod/Δpurity/ΔrGA/ΔrMA/Δviol.",
                        "priority_updates": [
                            "Require explicit metric delta vector for each deep comparison."
                        ],
                    }
                if not text_mentions_delta_flow_signals(delta_summary, min_count=3):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: delta summary must include explicit flow deltas.",
                        "priority_updates": [
                            "Require explicit ΔFfeed/ΔF1/ΔFdes/ΔFex/ΔFraf/Δtstep signals."
                        ],
                    }
                if len(topology_comparisons) < 3:
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: missing column topology comparison against R-1, R-2, and competitor.",
                        "priority_updates": [
                            "Require explicit topology comparison entries with nc and zone-column deltas."
                        ],
                    }
                if not text_mentions_required_labels(topology_comparisons, recent_two_labels):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: column topology comparison must reference both R-1 and R-2.",
                        "priority_updates": [
                            "Require R-1 and R-2 references in column topology comparison."
                        ],
                    }
                if not text_mentions_topology_signals(topology_comparisons) or not text_mentions_numeric_values(topology_comparisons):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: topology comparison must include NC/zone details with numeric deltas.",
                        "priority_updates": [
                            "Require nc/zone-column topology deltas (e.g., ΔZ1..ΔZ4) with numbers."
                        ],
                    }
            if has_history:
                if len(flow_comparisons) < 1 or not text_mentions_flow_signals(flow_comparisons):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: missing explicit flowrate comparison across prior runs.",
                        "priority_updates": [
                            "Require flowrate comparison using Ffeed/F1/Fdes/Fex/Fraf/tstep with implications."
                        ],
                    }
                if not text_mentions_numeric_values(flow_comparisons):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: flowrate comparison lacks numeric evidence.",
                        "priority_updates": [
                            "Require numeric deltas in flowrate comparison (e.g., Ffeed, tstep changes)."
                        ],
                    }
                if not physics_rationale or not text_mentions_physics_signals([physics_rationale]):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: missing physics-based rationale.",
                        "priority_updates": [
                            "Require physics rationale tied to mass balance, zone behavior, and adsorption/desorption effects."
                        ],
                    }
            if len(nc_comparisons) < 2:
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: NC competitor comparison is too weak.",
                    "priority_updates": [
                        "Require explicit candidate-vs-alternative NC comparisons before choosing next run."
                    ],
                }
            if not text_mentions_metric_signals(nc_comparisons) or not text_mentions_numeric_values(nc_comparisons):
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: NC competitor comparison must be metric- and number-grounded.",
                    "priority_updates": [
                        "Require quantitative competitor NC comparisons (productivity/purity/recovery/violation)."
                    ],
                }
            # --- Acquisition strategy validation ---
            acquisition_type = str(data.get("acquisition_type", "")).strip().upper()
            if acquisition_type not in {"EXPLORE", "EXPLOIT", "VERIFY"}:
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: missing or invalid acquisition_type (must be EXPLORE, EXPLOIT, or VERIFY).",
                    "priority_updates": [
                        "Every proposal must classify itself as EXPLORE, EXPLOIT, or VERIFY per the Acquisition Strategy Protocol."
                    ],
                }
            information_target = str(data.get("information_target", "")).strip()
            if len(information_target) < 10:
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: information_target is missing or too vague.",
                    "priority_updates": [
                        "State specifically what this run will teach us that we don't already know."
                    ],
                }
            alternatives_considered = normalize_text_list(data.get("alternatives_considered"), max_items=6)
            if len(alternatives_considered) < 2:
                return default_index, {
                    "mode": "deterministic",
                    "reason": "Rejected LLM proposal: must consider at least 2 alternatives before choosing this candidate.",
                    "priority_updates": [
                        "List at least 2 alternative candidates considered and why they were rejected."
                    ],
                }
            data["evidence"] = evidence
            data["comparison_to_previous"] = comparisons
            data["last_two_run_comparison"] = last_two_comparisons
            data["flowrate_comparison"] = flow_comparisons
            data["delta_summary"] = delta_summary
            data["column_topology_comparison"] = topology_comparisons
            data["physics_rationale"] = physics_rationale
            data["nc_competitor_comparison"] = nc_comparisons
            data["failure_criteria"] = normalize_text_list(data.get("failure_criteria"), max_items=8)
            data["diagnostic_hypothesis"] = str(data.get("diagnostic_hypothesis", "")).strip()
            data["acquisition_type"] = acquisition_type
            data["information_target"] = information_target
            data["alternatives_considered"] = alternatives_considered
            data["coverage_gap"] = str(data.get("coverage_gap", "")).strip()
            data["hypothesis_connection"] = str(data.get("hypothesis_connection", "")).strip()
            data["convergence_assessment"] = str(data.get("convergence_assessment", "")).strip()
            chosen = shortlist[idx]
            absolute_idx = candidate_tasks.index(chosen)
            return absolute_idx, {
                "mode": "llm",
                "llm_backend": client.last_backend,
                "prompt_warning": prompt_warning,
                "raw": raw,
                **data,
            }
    return default_index, {
        "mode": "deterministic",
        "reason": "Falling back to deterministic candidate choice.",
        "evidence": [
            "LLM output unavailable or invalid JSON.",
            "Deterministic fallback selected first untried task in ranked schedule."
        ],
        "comparison_to_previous": [
            "LLM output unavailable; deterministic order selected after checking tried tasks and current best summary."
        ],
        "last_two_run_comparison": [
            "LLM output unavailable; no structured R-1/R-2 deep comparison was produced."
        ],
        "flowrate_comparison": [
            "LLM output unavailable; no structured flowrate comparison was produced."
        ],
        "delta_summary": [
            "LLM output unavailable; no explicit delta summary (R-1/R-2/competitor) was produced."
        ],
        "column_topology_comparison": [
            "LLM output unavailable; no explicit column topology comparison was produced."
        ],
        "physics_rationale": "LLM output unavailable; no physics-based rationale was produced.",
        "nc_competitor_comparison": [
            "Deterministic fallback does not provide model-generated NC tradeoff reasoning."
        ],
        "failure_criteria": [
            "Reject if solver status is solver_error/other with no usable primal values.",
            "Reject if normalized_total_violation does not improve against current best evidence."
        ],
        "priority_updates": [
            "Use deterministic task order when model output is unavailable; collect more observations before strategy changes."
        ],
        "prompt_warning": prompt_warning,
    }


def scientist_b_review(
    client: OpenAICompatClient,
    task: Dict[str, object],
    effective_task: Dict[str, object],
    best_result: Optional[Dict[str, object]],
    results: List[Dict[str, object]],
    args: argparse.Namespace,
    codebase_context_excerpt: str,
    compute_context_excerpt: str,
    constraint_context_excerpt: str,
    nc_strategy_excerpt: str,
    research_excerpt: str,
    current_priorities: List[str],
    sqlite_context_excerpt: str,
    iteration: int,
) -> Dict[str, object]:
    default = deterministic_review(task, best_result)
    prompt_warning = ""
    recent_two_block, recent_two_labels = recent_two_run_review_context(results)
    try:
        prompt = textwrap.dedent(
            f"""
            You are Scientist_B. Review this proposed SMB medium-fidelity optimization attempt.
            Be adversarial and skeptical by default.
            Reject if rationale is generic, evidence is weak, or compute/constraint tradeoffs are ignored.
            You must explicitly compare the proposal against previous results before deciding.
            If you approve, still provide the strongest counterarguments and explicit risk checks.

            Proposed task:
            {json.dumps(task, indent=2)}

            Effective bounded candidate that will actually be executed:
            {json.dumps(effective_task, indent=2)}

            Current best result:
            {summarize_result(best_result) if best_result else "None yet."}

            Recent two completed runs (must be reviewed deeply when available):
            {recent_two_block}

            Codebase context summary:
            {codebase_context_excerpt}

            Runtime compute context:
            {compute_context_excerpt}

            Simulation objective/constraint context:
            {constraint_context_excerpt}

            NC strategy board (all layouts in current library):
            {nc_strategy_excerpt}

            Current research log tail:
            {research_excerpt}

            Current priority board:
            {json.dumps(current_priorities, indent=2)}

            Historical simulation context (queried from SQLite):
            {sqlite_context_excerpt}

            Respond with JSON only:
            {{
              "decision": "approve" or "reject",
              "reason": "<brief reason>",
              "comparison_assessment": ["<explicit comparison vs prior run(s) with quantitative metric/termination evidence>", "..."],
              "last_two_run_audit": ["<R-1: what happened, metrics, implications>", "<R-2: what happened, metrics, implications>"],
              "flowrate_audit": ["<flow deltas across runs with Ffeed/F1/Fdes/Fex/Fraf/tstep and implications>", "..."],
              "delta_audit": ["<vs R-1: Δprod=..., Δpurity=..., ΔrGA=..., ΔrMA=..., Δviol=..., ΔFfeed=..., ΔF1=..., ΔFdes=..., ΔFex=..., ΔFraf=..., Δtstep=...>", "<vs R-2: ...>", "<proposal vs counterproposal: Δprod=..., Δpurity=..., ΔrGA=..., ΔrMA=..., Δviol=...>"],
              "column_topology_audit": ["<vs R-1: nc=[...]->[...], ΔZ1=..., ΔZ2=..., ΔZ3=..., ΔZ4=..., implication>", "<vs R-2: ...>", "<proposal vs counterproposal topology tradeoff>"],
              "physics_audit": "<physics-grounded critique (mass balance, zone effects, adsorption/desorption/selectivity)>",
              "counterproposal_run": {{
                "nc": [a,b,c,d],
                "flow_adjustments": {{"Ffeed": <delta>, "F1": <delta>, "Fdes": <delta>, "Fex": <delta>, "Fraf": <delta>, "tstep": <delta>}},
                "expected_metric_effect": {{"delta_productivity": <value>, "delta_purity": <value>, "delta_recovery_ga": <value>, "delta_recovery_ma": <value>, "delta_violation": <value>}},
                "physics_justification": "<physics-based reason for this counterproposal>"
              }},
              "nc_strategy_assessment": ["<candidate nc vs alternatives and why>", "..."],
              "compute_assessment": "<budget/time parallelism assessment>",
              "counterarguments": ["<strongest objection 1>", "..."],
              "required_checks": ["<check before trusting result>", "..."],
              "priority_updates": ["..."],
              "risk_flags": ["..."]
            }}
            """
        ).strip()
    except Exception as exc:
        prompt_warning = f"Prompt build warning: {type(exc).__name__}: {exc}"
        prompt = (
            "You are Scientist_B reviewer. Return JSON only with keys decision, reason, comparison_assessment, "
            "last_two_run_audit, flowrate_audit, delta_audit, column_topology_audit, physics_audit, counterproposal_run, nc_strategy_assessment, compute_assessment, counterarguments, required_checks, priority_updates, risk_flags.\n\n"
            f"Proposed task:\n{json.dumps(task, indent=2)}\n\n"
            f"Effective candidate:\n{json.dumps(effective_task, indent=2)}\n\n"
            f"Current best result: {summarize_result(best_result) if best_result else 'None yet.'}\n\n"
            f"Recent two completed runs:\n{recent_two_block}"
        )
    raw = client.chat(
        "You are a hard-nosed numerical reviewer. Return JSON only and challenge weak proposals.",
        prompt,
        conversation_role="scientist_b_review",
        temperature=0.1,
        metadata={
            "iteration": iteration,
            "candidate_nc": task.get("nc"),
            "candidate_seed_name": task.get("seed_name"),
            "effective_flow": effective_task.get("flow", {}),
            "has_best_result": best_result is not None,
        },
    )
    data = client.extract_json(raw)
    if data and str(data.get("decision", "")).lower() in {"approve", "reject"}:
        comparisons = normalize_text_list(data.get("comparison_assessment"), max_items=8)
        last_two_audit = normalize_text_list(data.get("last_two_run_audit"), max_items=4)
        flow_audit = normalize_text_list(data.get("flowrate_audit"), max_items=6)
        delta_audit = normalize_text_list(data.get("delta_audit"), max_items=8)
        topology_audit = normalize_text_list(data.get("column_topology_audit"), max_items=8)
        physics_audit = str(data.get("physics_audit", "")).strip()
        counterproposal = data.get("counterproposal_run")
        nc_assessment = normalize_text_list(data.get("nc_strategy_assessment"), max_items=8)
        has_history = best_result is not None or (sqlite_total_records_from_excerpt(sqlite_context_excerpt) > 0)
        if not review_references_candidate_nc(
            str(data.get("reason", "")),
            comparisons,
            nc_assessment,
            task.get("nc", []),
        ):
            data["decision"] = "reject"
            data["reason"] = "Rejected: review appears to reference a different NC candidate than the proposed task."
            data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                "Review consistency risk: cited NC does not match proposed candidate."
            ]
            data["comparison_assessment"] = comparisons or [
                "Unable to verify candidate-specific comparison; review appears to cite a different NC."
            ]
        if not comparisons:
            data["decision"] = "reject"
            data["reason"] = "Rejected: review must include explicit comparison to previous results."
            data["priority_updates"] = normalize_text_list(data.get("priority_updates"), max_items=6) + [
                "Require comparison-to-history before any approval decision."
            ]
            data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                "Decision quality risk: no comparison against prior runs."
            ]
            data["comparison_assessment"] = [
                "No comparison provided; cannot assess whether proposal improves on prior evidence."
            ]
        if has_history and not text_mentions_prior_runs(comparisons):
            data["decision"] = "reject"
            data["reason"] = "Rejected: review comparison lacks concrete prior-run references."
            data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                "Review quality risk: missing run-level evidence in comparison assessment."
            ]
            data["comparison_assessment"] = comparisons or [
                "No run-level prior evidence referenced."
            ]
        if has_history and not text_mentions_metric_signals(comparisons):
            data["decision"] = "reject"
            data["reason"] = "Rejected: review comparison is not metric-grounded."
            data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                "Review quality risk: comparison lacks quantitative metrics."
            ]
            data["comparison_assessment"] = comparisons or [
                "No quantitative metric evidence referenced."
            ]
        if len(recent_two_labels) >= 2:
            if len(last_two_audit) < 2:
                data["decision"] = "reject"
                data["reason"] = "Rejected: review must include deep audit of both last two completed runs."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: missing R-1/R-2 deep audit block."
                ]
                data["last_two_run_audit"] = last_two_audit or [
                    "Missing required R-1 and R-2 deep audit."
                ]
            elif not text_mentions_required_labels(last_two_audit, recent_two_labels):
                data["decision"] = "reject"
                data["reason"] = "Rejected: last-two audit must explicitly reference both R-1 and R-2."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: R-1/R-2 labels not both present."
                ]
            elif not text_mentions_metric_signals(last_two_audit) or not text_mentions_numeric_values(last_two_audit):
                data["decision"] = "reject"
                data["reason"] = "Rejected: last-two audit is not metric- and number-grounded."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: R-1/R-2 audit lacks quantitative evidence."
                ]
            elif not text_mentions_run_name_signals(last_two_audit):
                data["decision"] = "reject"
                data["reason"] = "Rejected: last-two audit must cite explicit run names."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: R-1/R-2 audit missing run_name references."
                ]
            if len(delta_audit) < 3:
                data["decision"] = "reject"
                data["reason"] = "Rejected: review must include delta audit for R-1, R-2, and counterproposal."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: missing delta audit block."
                ]
            elif not text_mentions_required_labels(delta_audit, recent_two_labels):
                data["decision"] = "reject"
                data["reason"] = "Rejected: delta audit must explicitly reference both R-1 and R-2."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: delta audit missing R-1/R-2 labels."
                ]
            elif not text_mentions_delta_metric_signals(delta_audit):
                data["decision"] = "reject"
                data["reason"] = "Rejected: delta audit must include Δprod/Δpurity/ΔrGA/ΔrMA/Δviol."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: delta audit missing metric deltas."
                ]
            elif not text_mentions_delta_flow_signals(delta_audit, min_count=3):
                data["decision"] = "reject"
                data["reason"] = "Rejected: delta audit must include explicit flow deltas."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: delta audit missing flow deltas."
                ]
            if len(topology_audit) < 3:
                data["decision"] = "reject"
                data["reason"] = "Rejected: review must include column topology audit for R-1, R-2, and counterproposal."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: missing column topology audit block."
                ]
            elif not text_mentions_required_labels(topology_audit, recent_two_labels):
                data["decision"] = "reject"
                data["reason"] = "Rejected: column topology audit must reference both R-1 and R-2."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: topology audit missing R-1/R-2 labels."
                ]
            elif not text_mentions_topology_signals(topology_audit) or not text_mentions_numeric_values(topology_audit):
                data["decision"] = "reject"
                data["reason"] = "Rejected: column topology audit must include NC/zone numeric deltas."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: topology audit lacks numeric topology detail."
                ]
        if has_history:
            if len(flow_audit) < 1 or not text_mentions_flow_signals(flow_audit):
                data["decision"] = "reject"
                data["reason"] = "Rejected: review must include explicit flowrate audit across prior runs."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: missing flowrate audit (Ffeed/F1/Fdes/Fex/Fraf/tstep)."
                ]
                data["flowrate_audit"] = flow_audit or [
                    "Missing explicit flowrate audit with named variables and implications."
                ]
            elif not text_mentions_numeric_values(flow_audit):
                data["decision"] = "reject"
                data["reason"] = "Rejected: flowrate audit lacks numeric evidence."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: flowrate audit has no numeric deltas."
                ]
            if not physics_audit or not text_mentions_physics_signals([physics_audit]):
                data["decision"] = "reject"
                data["reason"] = "Rejected: review must include physics-grounded critique."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: missing physics-based audit."
                ]
            if not isinstance(counterproposal, dict):
                data["decision"] = "reject"
                data["reason"] = "Rejected: review must include a structured counterproposal_run."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: missing counterproposal run object."
                ]
                data["counterproposal_run"] = {
                    "nc": list(task.get("nc", [])) if isinstance(task.get("nc"), list) else [],
                    "flow_adjustments": {"Ffeed": 0.0, "F1": 0.0, "Fdes": 0.0, "Fex": 0.0, "Fraf": 0.0, "tstep": 0.0},
                    "expected_metric_effect": {
                        "delta_productivity": 0.0,
                        "delta_purity": 0.0,
                        "delta_recovery_ga": 0.0,
                        "delta_recovery_ma": 0.0,
                        "delta_violation": 0.0,
                    },
                    "physics_justification": "Missing counterproposal output from reviewer.",
                }
            else:
                cp_nc = counterproposal.get("nc")
                cp_flow = counterproposal.get("flow_adjustments")
                cp_effect = counterproposal.get("expected_metric_effect")
                cp_physics = str(counterproposal.get("physics_justification", "")).strip()
                valid_nc = isinstance(cp_nc, list) and len(cp_nc) == 4 and all(isinstance(v, (int, float)) for v in cp_nc)
                flow_numeric_count = 0
                if isinstance(cp_flow, dict):
                    for key in ("Ffeed", "F1", "Fdes", "Fex", "Fraf", "tstep"):
                        if isinstance(cp_flow.get(key), (int, float)):
                            flow_numeric_count += 1
                effect_numeric_count = 0
                if isinstance(cp_effect, dict):
                    for key in ("delta_productivity", "delta_purity", "delta_recovery_ga", "delta_recovery_ma", "delta_violation"):
                        if isinstance(cp_effect.get(key), (int, float)):
                            effect_numeric_count += 1
                if not valid_nc or flow_numeric_count < 2 or effect_numeric_count < 3 or not text_mentions_physics_signals([cp_physics]):
                    data["decision"] = "reject"
                    data["reason"] = "Rejected: counterproposal_run is incomplete; require NC + numeric flow edits + expected metric deltas + physics basis."
                    data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                        "Review quality risk: weak counterproposal detail."
                    ]
        if len(nc_assessment) < 2:
            data["decision"] = "reject"
            data["reason"] = "Rejected: review must include NC strategy assessment against alternatives."
            data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                "NC strategy assessment missing or too weak."
            ]
            data["nc_strategy_assessment"] = nc_assessment or [
                "No explicit candidate-vs-alternative NC assessment was provided."
            ]
        if str(data.get("decision", "")).lower() == "approve":
            counter = normalize_text_list(data.get("counterarguments"), max_items=3)
            checks = normalize_text_list(data.get("required_checks"), max_items=3)
            if not counter or not checks:
                data["decision"] = "reject"
                data["reason"] = "Rejected: approval must include explicit counterarguments and required checks."
                data["priority_updates"] = normalize_text_list(data.get("priority_updates"), max_items=6) + [
                    "Require adversarial review details before approving new tasks."
                ]
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Weak review quality due to missing counterarguments/checks."
                ]
                data["counterarguments"] = counter or [
                    "No explicit counterargument was provided by the reviewer."
                ]
                data["required_checks"] = checks or [
                    "Re-run review with explicit checks tied to bounds, solver behavior, and feasibility."
                ]
        data["comparison_assessment"] = normalize_text_list(data.get("comparison_assessment"), max_items=8)
        data["last_two_run_audit"] = normalize_text_list(data.get("last_two_run_audit"), max_items=4)
        data["flowrate_audit"] = normalize_text_list(data.get("flowrate_audit"), max_items=6)
        data["delta_audit"] = normalize_text_list(data.get("delta_audit"), max_items=8)
        data["column_topology_audit"] = normalize_text_list(data.get("column_topology_audit"), max_items=8)
        data["physics_audit"] = str(data.get("physics_audit", "")).strip()
        if isinstance(data.get("counterproposal_run"), dict):
            data["counterproposal_run"] = data.get("counterproposal_run")
        data["nc_strategy_assessment"] = normalize_text_list(data.get("nc_strategy_assessment"), max_items=8)
        data["compute_assessment"] = str(data.get("compute_assessment", "")).strip()
        return {
            "mode": "llm",
            "llm_backend": client.last_backend,
            "prompt_warning": prompt_warning,
            "raw": raw,
            **data,
        }
    return {"mode": "deterministic", **default}


def execute_search_task(
    args: argparse.Namespace,
    task: Dict[str, object],
    *,
    fidelity_override: Optional[Dict[str, int]] = None,
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
    candidate_args.run_name = f"{args.run_name}_search_nc_{'-'.join(str(v) for v in task['nc'])}_{candidate_args.seed_name}"
    result = rs.evaluate_optimized_layout(candidate_args, tuple(task["nc"]))
    if isinstance(fidelity_override, dict) or execution_note:
        result["execution_policy"] = {
            "fidelity_override": fidelity_override or {},
            "note": execution_note,
        }
    return result


def effective_search_task(args: argparse.Namespace, task: Dict[str, object]) -> Dict[str, object]:
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


def initialize_conversation_stream(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def write_artifact(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def write_conversation_log(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    artifact = artifact_path(args)
    conversation_artifact = conversation_log_path(args)
    conversation_stream_artifact = conversation_stream_log_path(args, conversation_artifact)
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
        args.llm_soul_file,
        heading_keywords=(
            "role",
            "core principle",
            "acquisition strategy protocol",
            "mandatory deep review",
            "what scientist_b must check",
            "when to stop",
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
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff_seconds,
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
    heuristics_excerpt = build_heuristics_context(max_chars=4000)
    sim_counter = 0  # global simulation counter for convergence tracking

    try:
        initialize_conversation_stream(conversation_stream_artifact)
        if args.reset_research_section:
            reset_research_run_section(research_path, args.run_name)
        nc_library_values = [list(nc) for nc in rs.parse_nc_library(args.nc_library)]
        initial_sqlite_excerpt = sqlite_history_context(sqlite_conn)
        initial_nc_strategy_excerpt = nc_strategy_board(sqlite_conn, nc_library_values)
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

        solver_check = rs.run_solver_check(configure_stage_args(make_stage_args("solver-check"), args))
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
            sqlite_excerpt = sqlite_history_context(sqlite_conn)
            nc_strategy_excerpt = nc_strategy_board(sqlite_conn, nc_library_values)
            research_excerpt = read_research_tail(research_path, args.research_tail_chars)
            convergence_excerpt = sqlite_convergence_context(sqlite_conn, args.run_name)
            try:
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
            best_so_far = rank_any_results(search_results)[0] if search_results else None
            if bool(int(getattr(args, "single_scientist_mode", 0))):
                b_note = single_scientist_policy_review(task, best_so_far)
            else:
                try:
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
                    )
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
            b_approved = str(b_note.get("decision", "approve")).lower() == "approve"
            if b_approved:
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1

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
                result["executive_forced"] = True
                result["executive_forced_from_task"] = task
                search_results.append(result)
                persist_result_to_sqlite(sqlite_conn, args.run_name, "search", result)
                sim_counter += 1
                record_convergence_snapshot(
                    sqlite_conn, args.run_name, "agent", sim_counter, result,
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
                consecutive_rejects = 0
                continue

            tried.add(task_key)
            execution_policy = search_execution_policy(args, search_tasks, search_results, task)
            if execution_policy.get("reason"):
                append_research(
                    research_path,
                    f"- execution_policy: {execution_policy.get('reason')}\n",
                )
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
            search_results.append(result)
            persist_result_to_sqlite(sqlite_conn, args.run_name, "search", result)
            sim_counter += 1
            acq_type = str(a_note.get("acquisition_type", "")).strip().upper() if isinstance(a_note, dict) else ""
            record_convergence_snapshot(
                sqlite_conn, args.run_name, "agent", sim_counter, result,
                search_hours_used * 3600.0, search_hours_used,
                acquisition_type=acq_type,
            )
            # Refresh heuristics after each run so next proposal uses updated knowledge
            heuristics_excerpt = build_heuristics_context(max_chars=4000)
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
            validation = execute_validation(args, candidate, ordinal)
            validation_results.append(validation)
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
                "method": "agent",
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
        }
        write_artifact(artifact, payload)
        print(json.dumps({"artifact": str(artifact), "status": "error", "run_name": args.run_name}, indent=2))
        return 1
    finally:
        sqlite_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
