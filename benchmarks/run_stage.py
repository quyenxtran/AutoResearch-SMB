#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SMB_ROOT = REPO_ROOT / "SembaSMB"
if str(SMB_ROOT) not in sys.path:
    sys.path.insert(0, str(SMB_ROOT))


REFERENCE_LAYOUT = (1, 2, 3, 2)
REFERENCE_WT0 = (0.003, 0.004, 0.990, 0.003)
REFERENCE_RHO = (1.5, 1.6, 1.0, 0.79)
REFERENCE_KAPP = (0.8, 1.22, 1.0, 0.69)


def parse_nc(raw: str) -> Tuple[int, int, int, int]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 4:
        raise ValueError(f"Expected four comma-separated integers for nc, got {raw!r}")
    if any(val <= 0 for val in values):
        raise ValueError(f"All nc values must be positive, got {values}")
    if sum(values) != 8:
        raise ValueError(f"nc must sum to 8 for this benchmark, got {values}")
    return values  # type: ignore[return-value]


def parse_nc_library(raw: str) -> List[Tuple[int, int, int, int]]:
    value = raw.strip().lower()
    if value in {"all", "all-positive", "all_positive"}:
        return list(generate_all_layouts())
    return [parse_nc(part.strip()) for part in raw.split(";") if part.strip()]


def generate_all_layouts(total_cols: int = 8, zones: int = 4) -> Iterable[Tuple[int, int, int, int]]:
    if zones != 4:
        raise ValueError("This helper currently assumes four SMB zones.")
    for a in range(1, total_cols - 2):
        for b in range(1, total_cols - a - 1):
            for c in range(1, total_cols - a - b):
                d = total_cols - a - b - c
                if d > 0:
                    yield (a, b, c, d)


def derive_fraf(ffeed: float, fdes: float, fex: float) -> float:
    return ffeed + fdes - fex


def load_config(args: argparse.Namespace, nc: Sequence[int]) -> SMBConfig:
    from src import SMBConfig  # type: ignore

    return SMBConfig(
        nc=tuple(nc),
        nfex=args.nfex,
        nfet=args.nfet,
        ncp=args.ncp,
        comps=("GA", "MA", "Water", "MeOH"),
        F1_init=args.f1,
        Fdes_init=args.fdes,
        Fex_init=args.fex,
        Ffeed_init=args.ffeed,
        Fraf_init=args.fraf if args.fraf is not None else derive_fraf(args.ffeed, args.fdes, args.fex),
        tstep_init=args.tstep,
        L=args.length,
        d=args.diameter,
        eb=args.eb,
        ep=args.ep,
        isoth="MLL",
        kapp=REFERENCE_KAPP,
        rho=REFERENCE_RHO,
        wt0=REFERENCE_WT0,
        Pe=args.pe,
        xscheme=args.xscheme,
    )


def build_flow(args: argparse.Namespace) -> FlowRates:
    from src import FlowRates  # type: ignore

    fraf = args.fraf if args.fraf is not None else derive_fraf(args.ffeed, args.fdes, args.fex)
    return FlowRates(
        F1=args.f1,
        Fdes=args.fdes,
        Fex=args.fex,
        Ffeed=args.ffeed,
        Fraf=fraf,
        tstep=args.tstep,
        run_name=args.run_name,
    )


def resolve_solver_name(requested: str) -> str:
    from src.smb_solver import check_solver_available  # type: ignore

    if requested != "auto":
        return requested
    if check_solver_available("ipopt_sens"):
        return "ipopt_sens"
    if check_solver_available("ipopt"):
        return "ipopt"
    raise RuntimeError(
        "No IPOPT executable is available. Install ipopt_sens or ipopt, "
        "or pass --solver-name with a working solver."
    )


def build_solver_options(args: argparse.Namespace) -> Dict[str, object]:
    from src import default_ipopt_options  # type: ignore

    options: Dict[str, object] = default_ipopt_options()
    if args.linear_solver:
        options["linear_solver"] = args.linear_solver
    if args.max_iter is not None:
        options["max_iter"] = args.max_iter
    if args.tol is not None:
        options["tol"] = args.tol
    if args.acceptable_tol is not None:
        options["acceptable_tol"] = args.acceptable_tol
    return options


def parse_solver_candidates(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def safe_float(value: object) -> float:
    return float(value)


def normalized_constraint_violation(metrics: Dict[str, float], flow: FlowRates, nc: Sequence[int], args: argparse.Namespace) -> Dict[str, float]:
    slacks = {
        "purity_ex_meoh_free": metrics["purity_ex_meoh_free"] - args.purity_min,
        "recovery_ex_GA": metrics["recovery_ex_GA"] - args.recovery_ga_min,
        "recovery_ex_MA": metrics["recovery_ex_MA"] - args.recovery_ma_min,
        "F1_max": args.max_pump_flow - flow.F1,
        "Fdes_max": args.max_pump_flow - flow.Fdes,
        "Fex_max": args.max_pump_flow - flow.Fex,
        "Ffeed_max": args.max_pump_flow - flow.Ffeed,
        "Fraf_max": args.max_pump_flow - flow.Fraf,
        "F2_positive": flow.F1 - flow.Fex,
        "F4_positive": flow.F1 - flow.Fdes,
        "nc_sum": 0.0 if sum(nc) == 8 else -abs(sum(nc) - 8),
    }
    normalized = (
        max(0.0, -slacks["purity_ex_meoh_free"]) / max(args.purity_min, 1e-12)
        + max(0.0, -slacks["recovery_ex_GA"]) / max(args.recovery_ga_min, 1e-12)
        + max(0.0, -slacks["recovery_ex_MA"]) / max(args.recovery_ma_min, 1e-12)
        + max(0.0, -slacks["F1_max"]) / max(args.max_pump_flow, 1e-12)
        + max(0.0, -slacks["Fdes_max"]) / max(args.max_pump_flow, 1e-12)
        + max(0.0, -slacks["Fex_max"]) / max(args.max_pump_flow, 1e-12)
        + max(0.0, -slacks["Ffeed_max"]) / max(args.max_pump_flow, 1e-12)
        + max(0.0, -slacks["Fraf_max"]) / max(args.max_pump_flow, 1e-12)
        + max(0.0, -slacks["F2_positive"]) / max(args.max_pump_flow, 1e-12)
        + max(0.0, -slacks["F4_positive"]) / max(args.max_pump_flow, 1e-12)
        + max(0.0, -slacks["nc_sum"])
    )
    slacks["normalized_total_violation"] = normalized
    return slacks


def evaluate_candidate(args: argparse.Namespace, nc: Sequence[int]) -> Dict[str, object]:
    from src import (  # type: ignore
        apply_discretization,
        build_inputs,
        build_model,
        compute_outlet_averages,
        compute_purity_recovery,
        solve_model,
    )

    solver_name = resolve_solver_name(args.solver_name)
    solver_options = build_solver_options(args)
    config = load_config(args, nc)
    flow = build_flow(args)

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    inputs = build_inputs(config, flow)
    m = build_model(config, inputs)
    apply_discretization(m, config, inputs)
    results = solve_model(m, solver_name=solver_name, options=solver_options, tee=args.tee)
    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    outlets = compute_outlet_averages(m, inputs)
    metrics = compute_purity_recovery(m, inputs, outlets)
    slacks = normalized_constraint_violation(metrics, flow, nc, args)
    feasible = (
        slacks["purity_ex_meoh_free"] >= 0.0
        and slacks["recovery_ex_GA"] >= 0.0
        and slacks["recovery_ex_MA"] >= 0.0
        and slacks["F1_max"] >= 0.0
        and slacks["Fdes_max"] >= 0.0
        and slacks["Fex_max"] >= 0.0
        and slacks["Ffeed_max"] >= 0.0
        and slacks["Fraf_max"] >= 0.0
        and slacks["F2_positive"] > 0.0
        and slacks["F4_positive"] > 0.0
        and slacks["nc_sum"] >= 0.0
    )
    cpus_used = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SMB_CPU_TASKS", "1")))
    wall_seconds = end_wall - start_wall
    cpu_seconds = end_cpu - start_cpu

    return {
        "status": "ok",
        "stage": args.stage,
        "run_name": args.run_name,
        "nc": list(nc),
        "flow": {
            "Ffeed": flow.Ffeed,
            "F1": flow.F1,
            "Fdes": flow.Fdes,
            "Fex": flow.Fex,
            "Fraf": flow.Fraf,
            "tstep": flow.tstep,
        },
        "fidelity": {"nfex": config.nfex, "nfet": config.nfet, "ncp": config.ncp, "xscheme": config.xscheme},
        "solver": {
            "solver_name": solver_name,
            "solver_options": solver_options,
            "status": str(results.solver.status),
            "termination_condition": str(results.solver.termination_condition),
        },
        "metrics": {key: safe_float(val) for key, val in metrics.items()},
        "constraint_slacks": slacks,
        "feasible": feasible,
        "J_validated": safe_float(metrics["productivity_ex_ga_ma"]) if feasible else None,
        "timing": {
            "wall_seconds": wall_seconds,
            "cpu_seconds_python": cpu_seconds,
            "cpus_used_for_accounting": cpus_used,
            "cpu_hours_accounted": wall_seconds * cpus_used / 3600.0,
        },
    }


def run_solver_check(args: argparse.Namespace) -> Dict[str, object]:
    from src.smb_solver import check_solver_available  # type: ignore

    candidates = parse_solver_candidates(args.solver_candidates)
    solver_reports = {}
    for solver_name in candidates:
        solver_path = shutil.which(solver_name)
        output = ""
        if solver_path:
            try:
                output = subprocess.run(
                    [solver_name, "--print-options"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                ).stdout
            except Exception:
                output = ""
        try:
            available = check_solver_available(solver_name)
        except Exception:
            available = False
        solver_reports[solver_name] = {
            "solver_path": solver_path,
            "solver_available": available,
            "detected_linear_solvers": {
                "ma57": "MA57 Linear Solver" in output,
                "mumps": "Mumps Linear Solver" in output,
                "ma97": "MA97 Linear Solver" in output,
                "pardiso": "Pardiso Linear Solver" in output,
            },
        }

    resolved_solver = None
    try:
        resolved_solver = resolve_solver_name(args.solver_name)
    except Exception:
        resolved_solver = None

    any_available = any(item["solver_available"] for item in solver_reports.values())
    return {
        "status": "ok" if any_available else "missing",
        "stage": args.stage,
        "requested_solver_name": args.solver_name,
        "resolved_solver_name": resolved_solver,
        "solver_reports": solver_reports,
    }


def rank_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def sort_key(item: Dict[str, object]) -> Tuple[int, float, float]:
        feasible = 1 if item.get("feasible") else 0
        j_validated = item.get("J_validated")
        score = float(j_validated) if j_validated is not None else float("-inf")
        violation = float(item["constraint_slacks"]["normalized_total_violation"])  # type: ignore[index]
        return (feasible, score, -violation)

    ranked = sorted(results, key=sort_key, reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx
    return ranked


def run_nc_screen(args: argparse.Namespace) -> Dict[str, object]:
    nc_library = parse_nc_library(args.nc_library)
    results: List[Dict[str, object]] = []
    for nc in nc_library:
        candidate_args = argparse.Namespace(**vars(args))
        candidate_args.run_name = f"{args.run_name}_nc_{'-'.join(str(v) for v in nc)}"
        try:
            results.append(evaluate_candidate(candidate_args, nc))
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    "stage": args.stage,
                    "run_name": candidate_args.run_name,
                    "nc": list(nc),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    successful = [item for item in results if item.get("status") == "ok"]
    ranked = rank_results(successful) if successful else []
    return {
        "status": "ok",
        "stage": args.stage,
        "nc_library": [list(nc) for nc in nc_library],
        "results": results,
        "ranked_results": ranked,
        "best_result": ranked[0] if ranked else None,
    }


def artifact_path(args: argparse.Namespace) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return Path(args.artifact_dir) / f"{args.stage}.{job_id}.{args.run_name}.json"


def write_artifact(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one SMB benchmark stage.")
    parser.add_argument("--stage", choices=["solver-check", "reference-eval", "nc-screen"], required=True)
    parser.add_argument("--run-name", default="pace_stage")
    parser.add_argument("--artifact-dir", default=str(REPO_ROOT / "artifacts" / "smb_stage_runs"))
    parser.add_argument("--solver-name", default="auto")
    parser.add_argument("--solver-candidates", default="ipopt_sens,ipopt,bonmin,couenne,cbc,glpk")
    parser.add_argument("--linear-solver")
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--acceptable-tol", type=float)
    parser.add_argument("--tee", action="store_true")

    parser.add_argument("--nc", default="1,2,3,2")
    parser.add_argument("--nc-library", default="1,2,3,2;2,2,2,2")
    parser.add_argument("--f1", type=float, default=2.2)
    parser.add_argument("--fdes", type=float, default=1.2)
    parser.add_argument("--fex", type=float, default=0.9)
    parser.add_argument("--ffeed", type=float, default=1.3)
    parser.add_argument("--fraf", type=float)
    parser.add_argument("--tstep", type=float, default=9.4)

    parser.add_argument("--nfex", type=int, default=10)
    parser.add_argument("--nfet", type=int, default=5)
    parser.add_argument("--ncp", type=int, default=2)
    parser.add_argument("--xscheme", default="CENTRAL")
    parser.add_argument("--length", type=float, default=20.0)
    parser.add_argument("--diameter", type=float, default=1.0)
    parser.add_argument("--eb", type=float, default=0.44)
    parser.add_argument("--ep", type=float, default=0.66)
    parser.add_argument("--pe", type=float, default=1000.0)

    parser.add_argument("--purity-min", type=float, default=0.90)
    parser.add_argument("--recovery-ga-min", type=float, default=0.90)
    parser.add_argument("--recovery-ma-min", type=float, default=0.90)
    parser.add_argument("--max-pump-flow", type=float, default=2.5)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    path = artifact_path(args)
    try:
        if args.stage == "solver-check":
            payload = run_solver_check(args)
        elif args.stage == "reference-eval":
            payload = evaluate_candidate(args, parse_nc(args.nc))
        elif args.stage == "nc-screen":
            payload = run_nc_screen(args)
        else:
            raise ValueError(f"Unknown stage {args.stage}")
        write_artifact(path, payload)
        print(json.dumps({"artifact": str(path), "stage": args.stage, "status": payload.get("status")}, indent=2))
        return 0
    except Exception as exc:
        payload = {
            "status": "error",
            "stage": args.stage,
            "run_name": args.run_name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_artifact(path, payload)
        print(json.dumps({"artifact": str(path), "stage": args.stage, "status": "error"}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
