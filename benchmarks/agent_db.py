from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .agent_results import (
    as_float,
    layout_text,
    effective_flow,
    extract_metrics_with_validity,
    effective_violation,
    rank_any_results,
    composition_metrics_from_raw_json,
    linear_slope,
)
from .agent_evidence import (
    compact_prompt_block,
    budget_evidence_pack_json,
    normalize_text_list,
)


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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


def nc_strategy_board(conn: sqlite3.Connection, nc_library: Sequence[Sequence[int]]) -> str:
    from .agent_policy import nc_key, nc_prior_score

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
    args: object,
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
        f"\n## Run: {getattr(args, 'run_name', '')}\n",
        f"- started_utc: {utc_now_text()}",
        f"- benchmark_hours: {getattr(args, 'benchmark_hours', '')}",
        f"- search_hours: {getattr(args, 'search_hours', '')}",
        f"- validation_hours: {getattr(args, 'validation_hours', '')}",
        f"- max_search_evals: {getattr(args, 'max_search_evals', '')}",
        f"- max_validations: {getattr(args, 'max_validations', '')}",
        f"- min_probe_reference_runs: {getattr(args, 'min_probe_reference_runs', '')}",
        f"- probe_low_fidelity_enabled: {bool(int(getattr(args, 'probe_low_fidelity_enabled', 0)))}",
        f"- probe_fidelity: nfex={getattr(args, 'probe_nfex', '')}, nfet={getattr(args, 'probe_nfet', '')}, ncp={getattr(args, 'probe_ncp', '')}",
        f"- finalization_hard_gate_enabled: {bool(int(getattr(args, 'finalization_hard_gate_enabled', 0)))}",
        f"- finalization_low_fidelity_requirements: nfex<={getattr(args, 'finalization_low_fidelity_nfex', '')}, nfet<={getattr(args, 'finalization_low_fidelity_nfet', '')}, ncp<={getattr(args, 'finalization_low_fidelity_ncp', '')}",
        f"- ipopt_defaults: max_iter={int(os.environ.get('SMB_IPOPT_MAX_ITER', '1000'))}, tol={float(os.environ.get('SMB_IPOPT_TOL', '1e-5'))}, acceptable_tol={float(os.environ.get('SMB_IPOPT_ACCEPTABLE_TOL', '1e-4'))}",
        f"- solver_name: {getattr(args, 'solver_name', '')}",
        f"- linear_solver: {getattr(args, 'linear_solver', '')}",
        f"- nc_library: {getattr(args, 'nc_library', '')}",
        f"- seed_library: {getattr(args, 'seed_library', '')}",
        f"- exploratory_targets: purity={getattr(args, 'purity_min', '')}, recovery_ga={getattr(args, 'recovery_ga_min', '')}, recovery_ma={getattr(args, 'recovery_ma_min', '')}",
        f"- project_objective_targets: purity={getattr(args, 'project_purity_min', '')}, recovery_ga={getattr(args, 'project_recovery_ga_min', '')}, recovery_ma={getattr(args, 'project_recovery_ma_min', '')}",
        f"- executive_controller: enabled={bool(getattr(args, 'executive_controller_enabled', False))}, trigger_rejects={getattr(args, 'executive_trigger_rejects', '')}, force_after={getattr(args, 'executive_force_after_rejects', '')}, top_k_lock={getattr(args, 'executive_top_k_lock', '')}",
        f"- single_scientist_mode: {bool(int(getattr(args, 'single_scientist_mode', 0)))}",
        f"- sqlite_db: {getattr(args, 'sqlite_db', '')}",
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
        if executive_note.get("acquisition_type"):
            lines.append(f"- executive_acquisition_type: {executive_note.get('acquisition_type')}")
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
    from .agent_results import composition_metrics_from_result
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
        lines.append(f"  - execution_policy_flow_override: {execution_policy.get('flow_override')}")
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
