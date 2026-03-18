#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from itertools import product
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Sequence, Tuple

from pyomo.environ import value


REPO_ROOT = Path(__file__).resolve().parents[1]
SMB_ROOT = REPO_ROOT / "src"
if str(SMB_ROOT) not in sys.path:
    sys.path.insert(0, str(SMB_ROOT))


REFERENCE_LAYOUT = (1, 2, 3, 2)
REFERENCE_WT0 = (0.003, 0.004, 0.990, 0.003)
REFERENCE_RHO = (1.5, 1.6, 1.0, 0.79)
REFERENCE_KAPP = (0.8, 1.22, 1.0, 0.69)
NOTEBOOK_SEEDS = [
    {"name": "reference", "F1": 2.2, "Fdes": 1.2, "Fex": 0.9, "Ffeed": 1.3, "Fraf": 1.6, "tstep": 9.4},
    {"name": "optimized_a", "F1": 3.6, "Fdes": 2.0, "Fex": 1.9, "Ffeed": 2.4, "Fraf": 2.5, "tstep": 8.0},
    {"name": "optimized_a_minus", "F1": 3.6, "Fdes": 2.0, "Fex": 1.9, "Ffeed": 2.35, "Fraf": 2.45, "tstep": 8.0},
    {"name": "optimized_a_plus", "F1": 3.6, "Fdes": 2.0, "Fex": 1.9, "Ffeed": 2.45, "Fraf": 2.55, "tstep": 8.0},
    {"name": "optimized_b", "F1": 3.7, "Fdes": 2.0, "Fex": 1.9, "Ffeed": 2.4, "Fraf": 2.5, "tstep": 8.0},
    {"name": "optimized_c", "F1": 3.5, "Fdes": 2.0, "Fex": 1.9, "Ffeed": 2.4, "Fraf": 2.5, "tstep": 8.0},
    {"name": "optimized_2f1", "F1": 3.4, "Fdes": 1.9, "Fex": 1.7, "Ffeed": 2.5, "Fraf": 2.7, "tstep": 8.0},
    {"name": "optimized_2f2", "F1": 3.5, "Fdes": 1.9, "Fex": 1.7, "Ffeed": 2.5, "Fraf": 2.7, "tstep": 8.0},
]


IPOPT_ITER_RE = re.compile(
    r"^\s*(\d+)\s+([\-+0-9.eE]+)\s+([\-+0-9.eE]+)\s+([\-+0-9.eE]+)\s+([\-+0-9.eE]+)\s+([\-+0-9.eE]+)\s+([\-+0-9.eE]+)\s+([\-+0-9.eE]+)\s+([\-+0-9.eE]+)\S*\s+(\d+)\s*$"
)


class IpoptLiveMonitor:
    def __init__(
        self,
        log_path: Path,
        poll_seconds: float,
        window_iters: int,
        stall_eps: float,
        watchdog_enabled: bool,
        watchdog_min_iters: int,
        watchdog_stall_windows: int,
        watchdog_max_inf_du: float,
        watchdog_max_mumps_realloc: int,
        watchdog_kill_callback: Callable[[str], Dict[str, object]] | None = None,
    ) -> None:
        self.log_path = log_path
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.window_iters = max(5, int(window_iters))
        self.stall_eps = max(0.0, float(stall_eps))
        self.watchdog_enabled = bool(watchdog_enabled)
        self.watchdog_min_iters = max(1, int(watchdog_min_iters))
        self.watchdog_stall_windows = max(1, int(watchdog_stall_windows))
        self.watchdog_max_inf_du = max(0.0, float(watchdog_max_inf_du))
        self.watchdog_max_mumps_realloc = max(0, int(watchdog_max_mumps_realloc))
        self.watchdog_kill_callback = watchdog_kill_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._iter_count = 0
        self._last_iter = -1
        self._last_objective: float | None = None
        self._last_inf_pr: float | None = None
        self._last_inf_du: float | None = None
        self._min_inf_pr: float | None = None
        self._min_inf_du: float | None = None
        self._max_inf_du: float | None = None
        self._ls_sum = 0
        self._mumps_realloc_count = 0
        self._stall_events = 0
        self._window: Deque[Tuple[int, float]] = deque(maxlen=self.window_iters)
        self._last_exit_line = ""
        self._warnings: List[str] = []
        self._watchdog_triggered = False
        self._watchdog_reason = ""
        self._watchdog_action: Dict[str, object] = {}

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.log_path.unlink(missing_ok=True)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        handle = None
        while not self._stop_event.is_set():
            try:
                if handle is None:
                    if not self.log_path.exists():
                        time.sleep(self.poll_seconds)
                        continue
                    handle = self.log_path.open("r", encoding="utf-8", errors="replace")

                line = handle.readline()
                if line:
                    self._consume_line(line.rstrip("\n"))
                    continue
                time.sleep(self.poll_seconds)
            except Exception:
                time.sleep(self.poll_seconds)
        try:
            if handle is not None:
                handle.close()
        except Exception:
            pass

    def _consume_line(self, line: str) -> None:
        if "MUMPS returned INFO(1) = -9" in line:
            self._mumps_realloc_count += 1
            self._warnings.append("MUMPS requested memory reallocation (INFO(1)=-9).")
            if (
                self.watchdog_enabled
                and self.watchdog_max_mumps_realloc > 0
                and self._mumps_realloc_count >= self.watchdog_max_mumps_realloc
                and not self._watchdog_triggered
            ):
                self._trigger_watchdog(
                    f"MUMPS reallocations reached threshold {self._mumps_realloc_count}/{self.watchdog_max_mumps_realloc}."
                )
        if line.strip().startswith("WARNING:"):
            self._warnings.append(line.strip())
        if "EXIT:" in line:
            self._last_exit_line = line.strip()

        match = IPOPT_ITER_RE.match(line)
        if not match:
            return

        it = int(match.group(1))
        objective = float(match.group(2))
        inf_pr = float(match.group(3))
        inf_du = float(match.group(4))
        ls = int(match.group(10))

        self._iter_count += 1
        self._last_iter = it
        self._last_objective = objective
        self._last_inf_pr = inf_pr
        self._last_inf_du = inf_du
        self._ls_sum += ls
        self._min_inf_pr = inf_pr if self._min_inf_pr is None else min(self._min_inf_pr, inf_pr)
        self._min_inf_du = inf_du if self._min_inf_du is None else min(self._min_inf_du, inf_du)
        self._max_inf_du = inf_du if self._max_inf_du is None else max(self._max_inf_du, inf_du)

        self._window.append((it, inf_pr))
        if len(self._window) >= self.window_iters:
            first_it, first_inf = self._window[0]
            last_it, last_inf = self._window[-1]
            denom = max(abs(first_inf), 1e-12)
            rel_improve = (first_inf - last_inf) / denom
            if (last_it - first_it) >= (self.window_iters - 1) and rel_improve <= self.stall_eps and last_inf > 1e-2:
                self._stall_events += 1
                if (
                    self.watchdog_enabled
                    and not self._watchdog_triggered
                    and self._iter_count >= self.watchdog_min_iters
                    and self._stall_events >= self.watchdog_stall_windows
                ):
                    self._trigger_watchdog(
                        f"Primal infeasibility stalled: inf_pr window improvement <= {self.stall_eps:.3g} for {self._stall_events} windows."
                    )
        if (
            self.watchdog_enabled
            and not self._watchdog_triggered
            and self.watchdog_max_inf_du > 0.0
            and self._iter_count >= self.watchdog_min_iters
            and inf_du >= self.watchdog_max_inf_du
        ):
            self._trigger_watchdog(
                f"Dual infeasibility exceeded threshold: inf_du={inf_du:.6g} >= {self.watchdog_max_inf_du:.6g}."
            )

    def _trigger_watchdog(self, reason: str) -> None:
        if self._watchdog_triggered:
            return
        self._watchdog_triggered = True
        self._watchdog_reason = reason
        self._warnings.append(f"WATCHDOG_TRIGGERED: {reason}")
        action: Dict[str, object] = {"attempted": False, "reason": reason}
        if self.watchdog_kill_callback is not None:
            try:
                action = self.watchdog_kill_callback(reason)
            except Exception as exc:
                action = {"attempted": True, "reason": reason, "errors": [f"kill_callback_failed: {type(exc).__name__}: {exc}"]}
        self._watchdog_action = action

    def snapshot(self) -> Dict[str, object]:
        avg_ls = (self._ls_sum / self._iter_count) if self._iter_count > 0 else 0.0
        suggestions: List[str] = []
        if self._stall_events > 0:
            suggestions.append(
                f"Inf_pr stalled {self._stall_events} window(s); consider early-stop/seed-switch if stagnation persists."
            )
        if self._mumps_realloc_count > 0:
            suggestions.append(
                f"MUMPS memory reallocated {self._mumps_realloc_count} time(s); consider smaller fidelity or higher memory."
            )
        if (self._max_inf_du or 0.0) > 1e6:
            suggestions.append("High dual infeasibility detected (>1e6); expect numerical stiffness.")
        return {
            "enabled": True,
            "log_path": str(self.log_path),
            "iterations_seen": self._iter_count,
            "last_iter": self._last_iter,
            "last_objective": self._last_objective,
            "last_inf_pr": self._last_inf_pr,
            "last_inf_du": self._last_inf_du,
            "best_inf_pr": self._min_inf_pr,
            "best_inf_du": self._min_inf_du,
            "max_inf_du": self._max_inf_du,
            "avg_ls": avg_ls,
            "mumps_realloc_count": self._mumps_realloc_count,
            "stall_events": self._stall_events,
            "last_exit_line": self._last_exit_line,
            "warnings": self._warnings[-20:],
            "suggestions": suggestions,
            "watchdog_enabled": self.watchdog_enabled,
            "watchdog_triggered": self._watchdog_triggered,
            "watchdog_reason": self._watchdog_reason,
            "watchdog_action": self._watchdog_action,
        }


def terminate_ipopt_descendants(reason: str) -> Dict[str, object]:
    action: Dict[str, object] = {
        "attempted": False,
        "reason": reason,
        "found_pids": [],
        "killed_pids": [],
        "remaining_pids": [],
        "errors": [],
    }
    try:
        import psutil  # type: ignore
    except Exception as exc:
        action["errors"] = [f"psutil_unavailable: {type(exc).__name__}: {exc}"]
        # Linux fallback without psutil.
        try:
            ps_out = subprocess.run(
                ["ps", "-eo", "pid=,ppid=,comm=,args="],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            ).stdout
            parent_pid = os.getpid()
            ppid_children: Dict[int, List[int]] = {}
            proc_info: Dict[int, Tuple[str, str]] = {}
            for line in ps_out.splitlines():
                parts = line.strip().split(None, 3)
                if len(parts) < 4:
                    continue
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                except Exception:
                    continue
                comm = parts[2]
                args = parts[3]
                ppid_children.setdefault(ppid, []).append(pid)
                proc_info[pid] = (comm, args)

            descendants: List[int] = []
            stack = list(ppid_children.get(parent_pid, []))
            while stack:
                child = stack.pop()
                descendants.append(child)
                stack.extend(ppid_children.get(child, []))

            targets: List[int] = []
            for pid in descendants:
                info = proc_info.get(pid)
                if not info:
                    continue
                comm_l = info[0].lower()
                args_l = info[1].lower()
                if "ipopt" in comm_l or "ipopt" in args_l:
                    targets.append(pid)

            action["found_pids"] = [int(pid) for pid in targets]
            if not targets:
                return action
            action["attempted"] = True
            for pid in targets:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception as kill_exc:
                    action["errors"].append(f"term_failed_pid_{pid}: {type(kill_exc).__name__}: {kill_exc}")
            time.sleep(1.0)
            alive: List[int] = []
            for pid in targets:
                try:
                    os.kill(pid, 0)
                    alive.append(pid)
                except Exception:
                    action["killed_pids"].append(pid)
            for pid in alive:
                try:
                    os.kill(pid, signal.SIGKILL)
                    action["killed_pids"].append(pid)
                except Exception as kill_exc:
                    action["errors"].append(f"kill_failed_pid_{pid}: {type(kill_exc).__name__}: {kill_exc}")
            action["remaining_pids"] = []
            for pid in alive:
                try:
                    os.kill(pid, 0)
                    action["remaining_pids"].append(pid)
                except Exception:
                    pass
        except Exception as fb_exc:
            action["errors"].append(f"fallback_kill_failed: {type(fb_exc).__name__}: {fb_exc}")
        return action

    try:
        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)
        targets = []
        for proc in children:
            try:
                name = (proc.name() or "").lower()
                cmdline = " ".join(proc.cmdline()).lower()
            except Exception:
                continue
            if "ipopt" in name or "ipopt" in cmdline:
                targets.append(proc)
        action["found_pids"] = [int(proc.pid) for proc in targets]
        if not targets:
            return action

        action["attempted"] = True
        for proc in targets:
            try:
                proc.terminate()
            except Exception as exc:
                action["errors"].append(f"terminate_failed_pid_{proc.pid}: {type(exc).__name__}: {exc}")
        gone, alive = psutil.wait_procs(targets, timeout=3.0)
        for proc in alive:
            try:
                proc.kill()
            except Exception as exc:
                action["errors"].append(f"kill_failed_pid_{proc.pid}: {type(exc).__name__}: {exc}")
        gone2, alive2 = psutil.wait_procs(alive, timeout=2.0)
        killed = list(gone) + list(gone2)
        action["killed_pids"] = [int(proc.pid) for proc in killed]
        action["remaining_pids"] = [int(proc.pid) for proc in alive2]
    except Exception as exc:
        action["errors"].append(f"watchdog_exception: {type(exc).__name__}: {exc}")
    return action


def maybe_start_ipopt_monitor(
    args: argparse.Namespace,
    watchdog_kill_callback: Callable[[str], Dict[str, object]] | None = None,
) -> Tuple[IpoptLiveMonitor | None, Path | None]:
    if not bool(getattr(args, "executive_live_monitor", False)):
        return None, None
    monitor_dir = Path(getattr(args, "ipopt_monitor_dir", str(REPO_ROOT / "artifacts" / "ipopt_live")))
    stage_name = str(getattr(args, "stage", "stage")).replace("/", "_")
    run_name = str(getattr(args, "run_name", "run")).replace("/", "_")
    log_path = monitor_dir / f"{stage_name}.{run_name}.ipopt.log"
    monitor = IpoptLiveMonitor(
        log_path=log_path,
        poll_seconds=float(getattr(args, "executive_monitor_poll_seconds", 1.0)),
        window_iters=int(getattr(args, "executive_monitor_window_iters", 12)),
        stall_eps=float(getattr(args, "executive_monitor_stall_eps", 0.01)),
        watchdog_enabled=bool(getattr(args, "executive_watchdog_enabled", False)),
        watchdog_min_iters=int(getattr(args, "executive_watchdog_min_iters", 80)),
        watchdog_stall_windows=int(getattr(args, "executive_watchdog_stall_windows", 2)),
        watchdog_max_inf_du=float(getattr(args, "executive_watchdog_max_inf_du", 0.0)),
        watchdog_max_mumps_realloc=int(getattr(args, "executive_watchdog_max_mumps_realloc", 0)),
        watchdog_kill_callback=watchdog_kill_callback,
    )
    monitor.start()
    return monitor, log_path


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


def parse_float_library(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one comma-separated float, got {raw!r}")
    return values


def parse_bounds(raw: str | None) -> Tuple[float, float] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    values = parse_float_library(text)
    if len(values) != 2:
        raise ValueError(f"Expected exactly two comma-separated floats for bounds, got {raw!r}")
    lb, ub = values
    if lb > ub:
        raise ValueError(f"Lower bound must be <= upper bound, got {raw!r}")
    return (lb, ub)


def parse_seed_library(raw: str) -> List[Dict[str, float | str]]:
    value = raw.strip().lower()
    if value in {"notebook", "kraton-notebook", "kraton_notebook"}:
        return [dict(seed) for seed in NOTEBOOK_SEEDS]
    seeds: List[Dict[str, float | str]] = []
    for idx, part in enumerate(raw.split(";"), start=1):
        text = part.strip()
        if not text:
            continue
        numbers = parse_float_library(text)
        if len(numbers) == 5:
            f1, fdes, fex, ffeed, tstep = numbers
            fraf = derive_fraf(ffeed, fdes, fex)
        elif len(numbers) == 6:
            f1, fdes, fex, ffeed, fraf, tstep = numbers
        else:
            raise ValueError(
                "Each seed entry must contain either 5 floats "
                "(F1,Fdes,Fex,Ffeed,tstep) or 6 floats (F1,Fdes,Fex,Ffeed,Fraf,tstep)."
            )
        seeds.append(
            {
                "name": f"seed_{idx:02d}",
                "F1": f1,
                "Fdes": fdes,
                "Fex": fex,
                "Ffeed": ffeed,
                "Fraf": fraf,
                "tstep": tstep,
            }
        )
    if not seeds:
        raise ValueError(f"Expected at least one seed entry, got {raw!r}")
    return seeds


def clip_to_bounds(value: float, bounds: Tuple[float, float] | None) -> float:
    if bounds is None:
        return value
    return min(max(value, bounds[0]), bounds[1])


def apply_seed_to_args(
    args: argparse.Namespace,
    seed: Dict[str, float | str],
    *,
    tstep_bounds: Tuple[float, float] | None,
    ffeed_bounds: Tuple[float, float] | None,
    fdes_bounds: Tuple[float, float] | None,
    fex_bounds: Tuple[float, float] | None,
    fraf_bounds: Tuple[float, float] | None,
    f1_bounds: Tuple[float, float] | None,
) -> argparse.Namespace:
    candidate_args = argparse.Namespace(**vars(args))
    candidate_args.seed_name = str(seed["name"])
    candidate_args.seed_flow_original = {
        "F1": float(seed["F1"]),
        "Fdes": float(seed["Fdes"]),
        "Fex": float(seed["Fex"]),
        "Ffeed": float(seed["Ffeed"]),
        "Fraf": float(seed["Fraf"]),
        "tstep": float(seed["tstep"]),
    }
    candidate_args.f1 = clip_to_bounds(float(seed["F1"]), f1_bounds)
    candidate_args.fdes = clip_to_bounds(float(seed["Fdes"]), fdes_bounds)
    candidate_args.fex = clip_to_bounds(float(seed["Fex"]), fex_bounds)
    candidate_args.ffeed = clip_to_bounds(float(seed["Ffeed"]), ffeed_bounds)
    candidate_args.fraf = clip_to_bounds(float(seed["Fraf"]), fraf_bounds)
    candidate_args.tstep = clip_to_bounds(float(seed["tstep"]), tstep_bounds)
    return candidate_args


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
    from sembasmb import check_solver_available

    if requested != "auto":
        return requested
    if check_solver_available("ipopt"):
        return "ipopt"
    if check_solver_available("ipopt_sens"):
        return "ipopt_sens"
    raise RuntimeError(
        "No IPOPT executable is available. Install ipopt or ipopt_sens, "
        "or pass --solver-name with a working solver."
    )


def build_solver_options(args: argparse.Namespace) -> Dict[str, object]:
    from src import default_ipopt_options  # type: ignore

    options: Dict[str, object] = default_ipopt_options()
    force_mumps_only = os.environ.get("SMB_FORCE_MUMPS_ONLY", "0") == "1"
    if force_mumps_only:
        # Hard lock for production runs where MA57/other linear solvers are known unstable.
        options["linear_solver"] = "mumps"
    elif args.linear_solver:
        options["linear_solver"] = args.linear_solver
    if args.max_iter is not None:
        options["max_iter"] = args.max_iter
    if args.tol is not None:
        options["tol"] = args.tol
    if args.acceptable_tol is not None:
        options["acceptable_tol"] = args.acceptable_tol
    max_solve_seconds = float(getattr(args, "max_solve_seconds", 0.0) or 0.0)
    if max_solve_seconds > 0:
        # IPOPT-level kill switch for long or stalled runs.
        options["max_wall_time"] = max_solve_seconds
        options["max_cpu_time"] = max_solve_seconds
    return options


def parse_solver_candidates(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def safe_float(value: object) -> float:
    return float(value)


def collect_profile_payload(m, inputs) -> Dict[str, object]:
    from src import compute_outlet_averages, compute_purity_recovery  # type: ignore

    outlets = compute_outlet_averages(m, inputs)
    metrics = compute_purity_recovery(m, inputs, outlets)

    t_index = list(m.t)
    t_values = [safe_float(t) for t in t_index]
    t_terminal = t_index[-1]
    col_ex = inputs.nc[0]
    col_raff = inputs.nc[0] + inputs.nc[1] + inputs.nc[2]
    ncol = sum(inputs.nc)

    outlet_time_series = {
        "t": t_values,
        "extract": {},
        "raffinate": {},
    }
    for j, comp_name in enumerate(inputs.comps, start=1):
        outlet_time_series["extract"][comp_name] = [
            safe_float(value(m.C[col_ex, j, t, 1.0])) for t in t_index
        ]
        outlet_time_series["raffinate"][comp_name] = [
            safe_float(value(m.C[col_raff, j, t, 1.0])) for t in t_index
        ]

    terminal_column_profile = {
        "time": safe_float(t_terminal),
        "columns": list(range(1, ncol + 1)),
        "components": {
            comp_name: [
                safe_float(value(m.C[col, j, t_terminal, 1.0])) for col in range(1, ncol + 1)
            ]
            for j, comp_name in enumerate(inputs.comps, start=1)
        },
    }

    return {
        "outlets": {
            stream: [safe_float(val) for val in values]
            for stream, values in outlets.items()
        },
        "metrics": {key: safe_float(val) for key, val in metrics.items()},
        "outlet_time_series": outlet_time_series,
        "terminal_column_profile": terminal_column_profile,
    }


def try_collect_profile_payload(m, inputs) -> Dict[str, object] | None:
    try:
        return collect_profile_payload(m, inputs)
    except Exception:
        return None


def solver_result_summary(results: object) -> Dict[str, str]:
    solver = getattr(results, "solver", None)
    status = str(getattr(solver, "status", "unknown"))
    termination = str(getattr(solver, "termination_condition", "unknown"))
    message = str(getattr(solver, "message", "") or "")
    return {
        "status": status,
        "termination_condition": termination,
        "message": message,
    }


def solver_result_usable(summary: Dict[str, str]) -> bool:
    return summary["status"].lower() == "ok" and summary["termination_condition"].lower() in {
        "optimal",
        "locallyoptimal",
        "feasible",
    }


def normalized_constraint_violation(metrics: Dict[str, float], flow: FlowRates, nc: Sequence[int], args: argparse.Namespace) -> Dict[str, float]:
    slacks = {
        "purity_ex_meoh_free": metrics["purity_ex_meoh_free"] - args.purity_min,
        "recovery_ex_GA": metrics["recovery_ex_GA"] - args.recovery_ga_min,
        "recovery_ex_MA": metrics["recovery_ex_MA"] - args.recovery_ma_min,
        "F1_max": args.f1_max_flow - flow.F1,
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
        + max(0.0, -slacks["F1_max"]) / max(args.f1_max_flow, 1e-12)
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


def try_constraint_slacks_from_metrics(
    metrics_obj: object,
    flow: FlowRates,
    nc: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, float] | None:
    if not isinstance(metrics_obj, dict):
        return None
    required = ("purity_ex_meoh_free", "recovery_ex_GA", "recovery_ex_MA")
    metric_values: Dict[str, float] = {}
    try:
        for key in required:
            metric_values[key] = safe_float(metrics_obj[key])  # type: ignore[index]
    except Exception:
        return None
    try:
        return normalized_constraint_violation(metric_values, flow, nc, args)
    except Exception:
        return None


def extract_optimized_flows(m, inputs, run_name: str) -> FlowRates:
    area_eb = inputs.area * inputs.eb
    return FlowRates(
        F1=value(m.U[1]) * area_eb,
        Fdes=value(m.UD) * area_eb,
        Fex=value(m.UE) * area_eb,
        Ffeed=value(m.UF) * area_eb,
        Fraf=value(m.UR) * area_eb,
        tstep=value(m.tstep),
        run_name=run_name,
    )


def extract_model_state(m, inputs) -> Dict[str, object]:
    """Capture solved variable values from a reference-eval model.

    Returns a dict of {variable_index: value} for C, Q, Cp, U, UF, UD, UE, UR,
    tstep — everything needed to warm-start an optimization model with the same
    NC layout at the same (or similar) fidelity.
    """
    state: Dict[str, object] = {}

    # Concentration, solid-phase, and particle-phase profiles
    c_vals = {}
    q_vals = {}
    cp_vals = {}
    for idx in m.C:
        v = m.C[idx].value
        if v is not None:
            c_vals[idx] = float(v)
    for idx in m.Q:
        v = m.Q[idx].value
        if v is not None:
            q_vals[idx] = float(v)
    for idx in m.Cp:
        v = m.Cp[idx].value
        if v is not None:
            cp_vals[idx] = float(v)
    state["C"] = c_vals
    state["Q"] = q_vals
    state["Cp"] = cp_vals

    # Flow velocities per column
    u_vals = {}
    for col in m.col:
        v = m.U[col].value
        if v is not None:
            u_vals[col] = float(v)
    state["U"] = u_vals

    # Global flow velocities and switching time
    state["UF"] = float(value(m.UF))
    state["UD"] = float(value(m.UD))
    state["UE"] = float(value(m.UE))
    state["UR"] = float(value(m.UR))
    state["tstep"] = float(value(m.tstep))

    return state


def apply_warm_start_state(m, state: Dict[str, object]) -> None:
    """Initialize an optimization model's variables from a prior solved state.

    Call this AFTER build_model + apply_discretization + add_optimization, but
    BEFORE solve_model. Sets variable values (not bounds) so IPOPT starts from
    the reference solution instead of a cold default initial point.
    """
    # Concentration, solid-phase, and particle-phase profiles
    c_vals = state.get("C", {})
    for idx, val in c_vals.items():
        try:
            m.C[idx].set_value(float(val))
        except (KeyError, ValueError):
            pass  # Index mismatch between fidelity levels — skip silently

    q_vals = state.get("Q", {})
    for idx, val in q_vals.items():
        try:
            m.Q[idx].set_value(float(val))
        except (KeyError, ValueError):
            pass

    cp_vals = state.get("Cp", {})
    for idx, val in cp_vals.items():
        try:
            m.Cp[idx].set_value(float(val))
        except (KeyError, ValueError):
            pass

    # Flow velocities per column
    u_vals = state.get("U", {})
    for col, val in u_vals.items():
        try:
            m.U[col].set_value(float(val))
        except (KeyError, ValueError):
            pass

    # Global flow velocities and switching time — set value (not fix)
    for attr in ("UF", "UD", "UE", "UR", "tstep"):
        val = state.get(attr)
        if val is not None:
            try:
                getattr(m, attr).set_value(float(val))
            except (AttributeError, ValueError):
                pass


def _ensure_ipopt_logfile(solver_options: Dict[str, object], args: argparse.Namespace) -> None:
    """Ensure IPOPT always writes a log file with full iteration detail.

    If the executive live monitor already set ``output_file``, this is a no-op.
    Otherwise, create a log file in the artifact directory so PACE users can
    ``tail -f`` the iteration table while the solver runs.
    """
    if "output_file" in solver_options:
        return  # Already configured by the monitor
    log_dir = Path(args.artifact_dir) / "ipopt_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ipopt_{args.run_name}.log"
    solver_options["output_file"] = str(log_path)
    solver_options.setdefault("file_print_level", 5)


def evaluate_candidate(args: argparse.Namespace, nc: Sequence[int], *, return_model_state: bool = False) -> Dict[str, object]:
    from src import (  # type: ignore
        apply_discretization,
        build_inputs,
        build_model,
        solve_model,
    )

    solver_name = resolve_solver_name(args.solver_name)
    solver_options = dict(build_solver_options(args))
    config = load_config(args, nc)
    flow = build_flow(args)
    monitor, monitor_log_path = maybe_start_ipopt_monitor(args, watchdog_kill_callback=terminate_ipopt_descendants)
    if monitor_log_path is not None:
        solver_options["output_file"] = str(monitor_log_path)
        solver_options.setdefault("file_print_level", 5)
        solver_options.setdefault("print_level", 5)
    _ensure_ipopt_logfile(solver_options, args)

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    inputs = build_inputs(config, flow)
    m = build_model(config, inputs)
    apply_discretization(m, config, inputs)
    solve_exc: Exception | None = None
    results = None
    try:
        results = solve_model(m, solver_name=solver_name, options=solver_options, tee=args.tee)
    except Exception as exc:
        solve_exc = exc
    finally:
        if monitor is not None:
            monitor.stop()
    monitor_snapshot = monitor.snapshot() if monitor is not None else None
    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    cpus_used = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SMB_CPU_TASKS", "1")))
    wall_seconds = end_wall - start_wall
    cpu_seconds = end_cpu - start_cpu
    if solve_exc is not None:
        watchdog_reason = ""
        if isinstance(monitor_snapshot, dict):
            watchdog_reason = str(monitor_snapshot.get("watchdog_reason", "") or "")
        error_msg = f"Solver execution failed: {type(solve_exc).__name__}: {solve_exc}"
        if watchdog_reason:
            error_msg += f" | watchdog={watchdog_reason}"
        payload = {
            "status": "solver_error",
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
            },
            "executive_live_monitor": monitor_snapshot,
            "error": error_msg,
            "timing": {
                "wall_seconds": wall_seconds,
                "cpu_seconds_python": cpu_seconds,
                "cpus_used_for_accounting": cpus_used,
                "cpu_hours_accounted": wall_seconds * cpus_used / 3600.0,
            },
        }
        provisional = try_collect_profile_payload(m, inputs)
        if provisional is not None:
            payload["provisional"] = {
                "source": "infeasible_last_iterate",
                "validated": False,
                **provisional,
            }
            slacks = try_constraint_slacks_from_metrics(provisional.get("metrics"), flow, nc, args)
            if slacks is not None:
                payload["constraint_slacks"] = slacks
                payload["feasible"] = False
                payload["infeasible_converged"] = True
                payload["J_validated"] = None
        return payload

    solver_summary = solver_result_summary(results)
    if not solver_result_usable(solver_summary):
        payload = {
            "status": "solver_error",
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
                **solver_summary,
            },
            "executive_live_monitor": monitor_snapshot,
            "error": (
                "Solver did not return a usable solution; metrics were not evaluated."
                + (
                    f" Watchdog triggered: {monitor_snapshot.get('watchdog_reason')}"
                    if isinstance(monitor_snapshot, dict) and monitor_snapshot.get("watchdog_triggered")
                    else ""
                )
            ),
            "timing": {
                "wall_seconds": wall_seconds,
                "cpu_seconds_python": cpu_seconds,
                "cpus_used_for_accounting": cpus_used,
                "cpu_hours_accounted": wall_seconds * cpus_used / 3600.0,
            },
        }
        provisional = try_collect_profile_payload(m, inputs)
        if provisional is not None:
            payload["provisional"] = {
                "source": "infeasible_last_iterate",
                "validated": False,
                **provisional,
            }
            slacks = try_constraint_slacks_from_metrics(provisional.get("metrics"), flow, nc, args)
            if slacks is not None:
                payload["constraint_slacks"] = slacks
                payload["feasible"] = False
                payload["infeasible_converged"] = True
                payload["J_validated"] = None
        return payload

    profile_payload = collect_profile_payload(m, inputs)
    metrics = profile_payload["metrics"]
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
    payload = {
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
            **solver_summary,
        },
        "executive_live_monitor": monitor_snapshot,
        **profile_payload,
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
    # Capture solved model state for warm-starting downstream optimization
    if return_model_state:
        try:
            payload["_model_state"] = extract_model_state(m, inputs)
        except Exception:
            pass  # Non-critical — optimization can still run without warm-start
    return payload


def evaluate_optimized_layout(
    args: argparse.Namespace,
    nc: Sequence[int],
    *,
    warm_start_state: Dict[str, object] | None = None,
) -> Dict[str, object]:
    from src import (  # type: ignore
        add_optimization,
        apply_discretization,
        build_inputs,
        build_model,
        solve_model,
    )

    solver_name = resolve_solver_name(args.solver_name)
    solver_options = dict(build_solver_options(args))
    config = load_config(args, nc)
    flow = build_flow(args)
    monitor, monitor_log_path = maybe_start_ipopt_monitor(args, watchdog_kill_callback=terminate_ipopt_descendants)
    if monitor_log_path is not None:
        solver_options["output_file"] = str(monitor_log_path)
        solver_options.setdefault("file_print_level", 5)
        solver_options.setdefault("print_level", 5)
    _ensure_ipopt_logfile(solver_options, args)
    tstep_bounds = parse_bounds(args.tstep_bounds)
    ffeed_bounds = parse_bounds(args.ffeed_bounds)
    fdes_bounds = parse_bounds(args.fdes_bounds)
    fex_bounds = parse_bounds(args.fex_bounds)
    fraf_bounds = parse_bounds(args.fraf_bounds)
    f1_bounds = parse_bounds(args.f1_bounds)

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    inputs = build_inputs(config, flow)
    m = build_model(config, inputs)
    apply_discretization(m, config, inputs)
    add_optimization(
        m,
        inputs,
        purity_min=args.purity_min,
        recovery_min_ga=args.recovery_ga_min,
        recovery_min_ma=args.recovery_ma_min,
        meoh_max_raff_wt=args.meoh_max_raff_wt,
        water_max_ex_wt=args.water_max_ex_wt,
        water_max_zone1_entry_wt=args.water_max_zone1_entry_wt,
        tstep_bounds=tstep_bounds,
        ffeed_bounds=ffeed_bounds,
        fdes_bounds=fdes_bounds,
        fex_bounds=fex_bounds,
        fraf_bounds=fraf_bounds,
        f1_bounds=f1_bounds,
        f1_min=args.f1_min,
        f1_max=args.f1_max,
    )

    # Apply warm-start state from a prior reference evaluation if available.
    # This initializes variable values (C, Q, Cp, U, tstep) from a solved
    # reference-eval run, giving IPOPT a much better starting point than the
    # cold default.  The warm-start state is optional — if missing or if
    # fidelity differs, apply_warm_start_state silently skips mismatched indices.
    if warm_start_state is not None:
        try:
            apply_warm_start_state(m, warm_start_state)
        except Exception:
            pass  # Non-critical — proceed with default initialization

    # Two-phase solve: Phase 1 finds a feasible point, Phase 2 optimizes from it.
    two_phase = bool(int(os.environ.get("SMB_TWO_PHASE_SOLVE", "1")))
    solve_exc: Exception | None = None
    results = None
    phase1_status = None
    try:
        if two_phase:
            from src import add_feasibility_objective, restore_productivity_objective  # type: ignore
            from src import feasibility_restoration_options, warm_start_options  # type: ignore

            # Phase 1: minimize constraint violation
            add_feasibility_objective(m, inputs)
            phase1_options = dict(solver_options)
            phase1_options.update(feasibility_restoration_options())
            if monitor_log_path is not None:
                phase1_options["output_file"] = str(monitor_log_path)
            try:
                phase1_results = solve_model(m, solver_name=solver_name, options=phase1_options, tee=args.tee)
                phase1_status = str(phase1_results.solver.termination_condition) if phase1_results else "unknown"
            except Exception:
                phase1_status = "phase1_error"

            # Phase 2: restore productivity objective with warm-start
            restore_productivity_objective(m)
            phase2_options = dict(solver_options)
            if phase1_status not in ("phase1_error",):
                phase2_options.update(warm_start_options())
            results = solve_model(m, solver_name=solver_name, options=phase2_options, tee=args.tee)
        else:
            results = solve_model(m, solver_name=solver_name, options=solver_options, tee=args.tee)
    except Exception as exc:
        solve_exc = exc
    finally:
        if monitor is not None:
            monitor.stop()
    monitor_snapshot = monitor.snapshot() if monitor is not None else None
    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    cpus_used = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SMB_CPU_TASKS", "1")))
    wall_seconds = end_wall - start_wall
    cpu_seconds = end_cpu - start_cpu

    if solve_exc is not None:
        watchdog_reason = ""
        if isinstance(monitor_snapshot, dict):
            watchdog_reason = str(monitor_snapshot.get("watchdog_reason", "") or "")
        error_msg = f"Solver execution failed: {type(solve_exc).__name__}: {solve_exc}"
        if watchdog_reason:
            error_msg += f" | watchdog={watchdog_reason}"
        slack_flow = flow
        payload = {
            "status": "solver_error",
            "stage": args.stage,
            "run_name": args.run_name,
            "nc": list(nc),
            "seed_name": getattr(args, "seed_name", None),
            "seed_flow_original": getattr(args, "seed_flow_original", None),
            "initial_flow": {
                "Ffeed": flow.Ffeed,
                "F1": flow.F1,
                "Fdes": flow.Fdes,
                "Fex": flow.Fex,
                "Fraf": flow.Fraf,
                "tstep": flow.tstep,
            },
            "solver": {
                "solver_name": solver_name,
                "solver_options": solver_options,
            },
            "executive_live_monitor": monitor_snapshot,
            "optimization_bounds": {
                "tstep_bounds": tstep_bounds,
                "ffeed_bounds": ffeed_bounds,
                "fdes_bounds": fdes_bounds,
                "fex_bounds": fex_bounds,
                "fraf_bounds": fraf_bounds,
                "f1_bounds": f1_bounds,
            },
            "error": error_msg,
            "timing": {
                "wall_seconds": wall_seconds,
                "cpu_seconds_python": cpu_seconds,
                "cpus_used_for_accounting": cpus_used,
                "cpu_hours_accounted": wall_seconds * cpus_used / 3600.0,
            },
        }
        try:
            provisional_flow = extract_optimized_flows(m, inputs, args.run_name)
            slack_flow = provisional_flow
            payload["provisional_optimized_flow"] = {
                "Ffeed": provisional_flow.Ffeed,
                "F1": provisional_flow.F1,
                "Fdes": provisional_flow.Fdes,
                "Fex": provisional_flow.Fex,
                "Fraf": provisional_flow.Fraf,
                "tstep": provisional_flow.tstep,
            }
        except Exception:
            pass
        provisional = try_collect_profile_payload(m, inputs)
        if provisional is not None:
            payload["provisional"] = {
                "source": "infeasible_last_iterate",
                "validated": False,
                **provisional,
            }
            slacks = try_constraint_slacks_from_metrics(provisional.get("metrics"), slack_flow, nc, args)
            if slacks is not None:
                payload["constraint_slacks"] = slacks
                payload["feasible"] = False
                payload["infeasible_converged"] = True
                payload["J_validated"] = None
        return payload

    solver_summary = solver_result_summary(results)
    if not solver_result_usable(solver_summary):
        slack_flow = flow
        payload = {
            "status": "solver_error",
            "stage": args.stage,
            "run_name": args.run_name,
            "nc": list(nc),
            "seed_name": getattr(args, "seed_name", None),
            "seed_flow_original": getattr(args, "seed_flow_original", None),
            "initial_flow": {
                "Ffeed": flow.Ffeed,
                "F1": flow.F1,
                "Fdes": flow.Fdes,
                "Fex": flow.Fex,
                "Fraf": flow.Fraf,
                "tstep": flow.tstep,
            },
            "solver": {
                "solver_name": solver_name,
                "solver_options": solver_options,
                **solver_summary,
            },
            "executive_live_monitor": monitor_snapshot,
            "optimization_bounds": {
                "tstep_bounds": tstep_bounds,
                "ffeed_bounds": ffeed_bounds,
                "fdes_bounds": fdes_bounds,
                "fex_bounds": fex_bounds,
                "fraf_bounds": fraf_bounds,
                "f1_bounds": f1_bounds,
            },
            "error": (
                "Solver did not return a usable solution; optimized metrics were not evaluated."
                + (
                    f" Watchdog triggered: {monitor_snapshot.get('watchdog_reason')}"
                    if isinstance(monitor_snapshot, dict) and monitor_snapshot.get("watchdog_triggered")
                    else ""
                )
            ),
            "timing": {
                "wall_seconds": wall_seconds,
                "cpu_seconds_python": cpu_seconds,
                "cpus_used_for_accounting": cpus_used,
                "cpu_hours_accounted": wall_seconds * cpus_used / 3600.0,
            },
        }
        try:
            provisional_flow = extract_optimized_flows(m, inputs, args.run_name)
            slack_flow = provisional_flow
            payload["provisional_optimized_flow"] = {
                "Ffeed": provisional_flow.Ffeed,
                "F1": provisional_flow.F1,
                "Fdes": provisional_flow.Fdes,
                "Fex": provisional_flow.Fex,
                "Fraf": provisional_flow.Fraf,
                "tstep": provisional_flow.tstep,
            }
        except Exception:
            pass
        provisional = try_collect_profile_payload(m, inputs)
        if provisional is not None:
            payload["provisional"] = {
                "source": "infeasible_last_iterate",
                "validated": False,
                **provisional,
            }
            slacks = try_constraint_slacks_from_metrics(provisional.get("metrics"), slack_flow, nc, args)
            if slacks is not None:
                payload["constraint_slacks"] = slacks
                payload["feasible"] = False
                payload["infeasible_converged"] = True
                payload["J_validated"] = None
        return payload

    optimized_flow = extract_optimized_flows(m, inputs, args.run_name)
    profile_payload = collect_profile_payload(m, inputs)
    metrics = profile_payload["metrics"]
    slacks = normalized_constraint_violation(metrics, optimized_flow, nc, args)
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
    return {
        "status": "ok",
        "stage": args.stage,
        "run_name": args.run_name,
        "nc": list(nc),
        "seed_name": getattr(args, "seed_name", None),
        "seed_flow_original": getattr(args, "seed_flow_original", None),
        "initial_flow": {
            "Ffeed": flow.Ffeed,
            "F1": flow.F1,
            "Fdes": flow.Fdes,
            "Fex": flow.Fex,
            "Fraf": flow.Fraf,
            "tstep": flow.tstep,
        },
        "optimized_flow": {
            "Ffeed": optimized_flow.Ffeed,
            "F1": optimized_flow.F1,
            "Fdes": optimized_flow.Fdes,
            "Fex": optimized_flow.Fex,
            "Fraf": optimized_flow.Fraf,
            "tstep": optimized_flow.tstep,
        },
        "fidelity": {"nfex": config.nfex, "nfet": config.nfet, "ncp": config.ncp, "xscheme": config.xscheme},
        "solver": {
            "solver_name": solver_name,
            "solver_options": solver_options,
            "two_phase_solve": two_phase,
            "phase1_status": phase1_status,
            "warm_started_from_reference": warm_start_state is not None,
            **solver_summary,
        },
        "executive_live_monitor": monitor_snapshot,
        "optimization_bounds": {
            "tstep_bounds": tstep_bounds,
            "ffeed_bounds": ffeed_bounds,
            "fdes_bounds": fdes_bounds,
            "fex_bounds": fex_bounds,
            "fraf_bounds": fraf_bounds,
            "f1_bounds": f1_bounds,
        },
        **profile_payload,
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
    from sembasmb import check_solver_available

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


def run_flow_screen(args: argparse.Namespace) -> Dict[str, object]:
    nc = parse_nc(args.nc)
    flow_grid = {
        "ffeed": parse_float_library(args.ffeed_library),
        "f1": parse_float_library(args.f1_library),
        "fdes": parse_float_library(args.fdes_library),
        "fex": parse_float_library(args.fex_library),
        "tstep": parse_float_library(args.tstep_library),
    }
    results: List[Dict[str, object]] = []

    for idx, (ffeed, f1, fdes, fex, tstep) in enumerate(
        product(
            flow_grid["ffeed"],
            flow_grid["f1"],
            flow_grid["fdes"],
            flow_grid["fex"],
            flow_grid["tstep"],
        ),
        start=1,
    ):
        candidate_args = argparse.Namespace(**vars(args))
        candidate_args.ffeed = ffeed
        candidate_args.f1 = f1
        candidate_args.fdes = fdes
        candidate_args.fex = fex
        candidate_args.tstep = tstep
        candidate_args.fraf = args.fraf if args.fraf is not None else derive_fraf(ffeed, fdes, fex)
        candidate_args.run_name = (
            f"{args.run_name}_pt_{idx:03d}"
            f"_ffeed_{ffeed:g}_f1_{f1:g}_fdes_{fdes:g}_fex_{fex:g}_t_{tstep:g}"
        )
        try:
            results.append(evaluate_candidate(candidate_args, nc))
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    "stage": args.stage,
                    "run_name": candidate_args.run_name,
                    "nc": list(nc),
                    "flow": {
                        "Ffeed": ffeed,
                        "F1": f1,
                        "Fdes": fdes,
                        "Fex": fex,
                        "Fraf": candidate_args.fraf,
                        "tstep": tstep,
                    },
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    successful = [item for item in results if item.get("status") == "ok"]
    ranked = rank_results(successful) if successful else []
    return {
        "status": "ok",
        "stage": args.stage,
        "nc": list(nc),
        "flow_grid": flow_grid,
        "num_candidates": len(results),
        "results": results,
        "ranked_results": ranked,
        "best_result": ranked[0] if ranked else None,
    }


def _pick_best_reference_state(
    ref_results: List[Dict[str, object]],
) -> Dict[str, object] | None:
    """Select the best reference-eval model state for warm-starting optimization.

    Preference order:
    1. Feasible result with highest J_validated
    2. Any result with a captured _model_state (even if infeasible — still a
       better starting point than cold default)
    """
    feasible_with_state = [
        r for r in ref_results
        if r.get("feasible") and r.get("_model_state") is not None
    ]
    if feasible_with_state:
        best = max(
            feasible_with_state,
            key=lambda r: float(r.get("J_validated") or 0.0),
        )
        return best["_model_state"]  # type: ignore[return-value]

    # Fall back to any result with a model state
    with_state = [r for r in ref_results if r.get("_model_state") is not None]
    if with_state:
        return with_state[0]["_model_state"]  # type: ignore[return-value]

    return None


def run_optimize_layouts(args: argparse.Namespace) -> Dict[str, object]:
    nc_library = parse_nc_library(args.nc_library)
    tstep_bounds = parse_bounds(args.tstep_bounds)
    ffeed_bounds = parse_bounds(args.ffeed_bounds)
    fdes_bounds = parse_bounds(args.fdes_bounds)
    fex_bounds = parse_bounds(args.fex_bounds)
    fraf_bounds = parse_bounds(args.fraf_bounds)
    f1_bounds = parse_bounds(args.f1_bounds)
    seed_library = parse_seed_library(args.seed_library)
    results: List[Dict[str, object]] = []

    # Convergence tracking for MINLP baseline comparison
    convergence_log: List[Dict[str, object]] = []
    best_feasible_j: float | None = None
    best_feasible_run: str | None = None
    best_feasible_prod: float | None = None
    cumulative_wall_s = 0.0
    sim_number = 0

    # Whether to run reference evals before optimization (env-configurable)
    run_ref_gate = bool(int(os.environ.get("SMB_REFERENCE_GATE", "1")))
    # How many seeds to use for the reference-eval gate (default: 3 — the
    # first 3 NOTEBOOK_SEEDS cover the main operating regions)
    ref_gate_max_seeds = int(os.environ.get("SMB_REFERENCE_GATE_MAX_SEEDS", "3"))

    for nc in nc_library:
        # --- Reference-eval gate: run fixed-flow solves first to find a good
        # warm-start point for optimization.  This is much cheaper than a full
        # optimization run and gives IPOPT a physically reasonable starting
        # point instead of the cold default initial guess.
        warm_start_state: Dict[str, object] | None = None
        ref_gate_results: List[Dict[str, object]] = []
        best_ref_seed_name: str | None = None

        if run_ref_gate:
            ref_seeds = seed_library[:ref_gate_max_seeds]
            for seed in ref_seeds:
                ref_args = apply_seed_to_args(
                    args,
                    seed,
                    tstep_bounds=tstep_bounds,
                    ffeed_bounds=ffeed_bounds,
                    fdes_bounds=fdes_bounds,
                    fex_bounds=fex_bounds,
                    fraf_bounds=fraf_bounds,
                    f1_bounds=f1_bounds,
                )
                ref_args.run_name = f"{args.run_name}_refgate_nc_{'-'.join(str(v) for v in nc)}_{ref_args.seed_name}"
                try:
                    ref_result = evaluate_candidate(ref_args, nc, return_model_state=True)
                    ref_gate_results.append(ref_result)
                except Exception:
                    pass  # Reference gate failure is non-fatal

                # Count reference evals in convergence tracking too
                sim_number += 1
                timing = ref_result.get("timing") or {} if ref_gate_results else {}
                cumulative_wall_s += float(timing.get("wall_seconds", 0.0) if isinstance(timing, dict) else 0.0)
                if ref_gate_results and ref_gate_results[-1].get("feasible"):
                    j_val = ref_gate_results[-1].get("J_validated")
                    if j_val is not None:
                        j_float = float(j_val)
                        if best_feasible_j is None or j_float > best_feasible_j:
                            best_feasible_j = j_float
                            best_feasible_run = str(ref_gate_results[-1].get("run_name", ""))
                            ref_metrics = ref_gate_results[-1].get("metrics") or {}
                            if isinstance(ref_metrics, dict):
                                best_feasible_prod = float(ref_metrics.get("productivity_ex_ga_ma", 0.0))
                convergence_log.append({
                    "sim_number": sim_number,
                    "candidate_run_name": ref_args.run_name,
                    "best_feasible_j": best_feasible_j,
                    "best_feasible_productivity": best_feasible_prod,
                    "best_feasible_run_name": best_feasible_run,
                    "cumulative_wall_seconds": cumulative_wall_s,
                    "nc": list(nc),
                    "seed_name": str(seed.get("name", "")),
                    "status": str(ref_gate_results[-1].get("status", "error")) if ref_gate_results else "error",
                    "feasible": bool(ref_gate_results[-1].get("feasible")) if ref_gate_results else False,
                    "phase": "reference_gate",
                })

            warm_start_state = _pick_best_reference_state(ref_gate_results)
            if warm_start_state is not None:
                # Find which seed produced the best state
                for rr in ref_gate_results:
                    if rr.get("_model_state") is warm_start_state:
                        best_ref_seed_name = str(rr.get("run_name", ""))
                        break

        # --- Optimization runs: use warm-start from best reference eval
        for seed in seed_library:
            candidate_args = apply_seed_to_args(
                args,
                seed,
                tstep_bounds=tstep_bounds,
                ffeed_bounds=ffeed_bounds,
                fdes_bounds=fdes_bounds,
                fex_bounds=fex_bounds,
                fraf_bounds=fraf_bounds,
                f1_bounds=f1_bounds,
            )
            candidate_args.run_name = f"{args.run_name}_nc_{'-'.join(str(v) for v in nc)}_{candidate_args.seed_name}"
            try:
                result = evaluate_optimized_layout(
                    candidate_args, nc, warm_start_state=warm_start_state,
                )
                if best_ref_seed_name is not None:
                    result["warm_start_source"] = best_ref_seed_name
                results.append(result)
            except Exception as exc:
                result = {
                    "status": "error",
                    "stage": args.stage,
                    "run_name": candidate_args.run_name,
                    "nc": list(nc),
                    "seed_name": candidate_args.seed_name,
                    "seed_flow_original": candidate_args.seed_flow_original,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                results.append(result)

            # Track convergence
            sim_number += 1
            timing = result.get("timing") or {}
            cumulative_wall_s += float(timing.get("wall_seconds", 0.0) if isinstance(timing, dict) else 0.0)
            if result.get("feasible"):
                j_val = result.get("J_validated")
                if j_val is not None:
                    j_float = float(j_val)
                    if best_feasible_j is None or j_float > best_feasible_j:
                        best_feasible_j = j_float
                        best_feasible_run = str(result.get("run_name", ""))
                        metrics = result.get("metrics") or {}
                        if isinstance(metrics, dict):
                            best_feasible_prod = float(metrics.get("productivity_ex_ga_ma", 0.0))
            convergence_log.append({
                "sim_number": sim_number,
                "candidate_run_name": str(result.get("run_name", "")),
                "best_feasible_j": best_feasible_j,
                "best_feasible_productivity": best_feasible_prod,
                "best_feasible_run_name": best_feasible_run,
                "cumulative_wall_seconds": cumulative_wall_s,
                "nc": list(nc),
                "seed_name": str(seed.get("name", "")),
                "status": str(result.get("status", "")),
                "feasible": bool(result.get("feasible")),
                "phase": "optimization",
            })

    successful = [item for item in results if item.get("status") == "ok"]
    ranked = rank_results(successful) if successful else []
    return {
        "status": "ok",
        "stage": args.stage,
        "nc_library": [list(nc) for nc in nc_library],
        "seed_library": seed_library,
        "reference_gate_enabled": run_ref_gate,
        "reference_gate_max_seeds": ref_gate_max_seeds,
        "results": results,
        "ranked_results": ranked,
        "best_result": ranked[0] if ranked else None,
        "convergence_log": convergence_log,
    }


def artifact_path(args: argparse.Namespace) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return Path(args.artifact_dir) / f"{args.stage}.{job_id}.{args.run_name}.json"


def write_artifact(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one SMB benchmark stage.")
    parser.add_argument("--stage", choices=["solver-check", "reference-eval", "nc-screen", "flow-screen", "optimize-layouts"], required=True)
    parser.add_argument("--run-name", default="pace_stage")
    parser.add_argument("--artifact-dir", default=str(REPO_ROOT / "artifacts" / "smb_stage_runs"))
    parser.add_argument("--solver-name", default="auto")
    parser.add_argument("--solver-candidates", default="ipopt_sens,ipopt,bonmin,couenne,cbc,glpk")
    parser.add_argument("--linear-solver")
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--acceptable-tol", type=float)
    parser.add_argument("--max-solve-seconds", type=float, default=float(os.environ.get("SMB_MAX_SOLVE_SECONDS", "0")))
    parser.add_argument("--executive-live-monitor", action="store_true", default=os.environ.get("SMB_EXECUTIVE_LIVE_MONITOR", "0") == "1")
    parser.add_argument(
        "--executive-monitor-poll-seconds",
        type=float,
        default=float(os.environ.get("SMB_EXECUTIVE_MONITOR_POLL_SECONDS", "1.0")),
    )
    parser.add_argument(
        "--executive-monitor-window-iters",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_MONITOR_WINDOW_ITERS", "12")),
    )
    parser.add_argument(
        "--executive-monitor-stall-eps",
        type=float,
        default=float(os.environ.get("SMB_EXECUTIVE_MONITOR_STALL_EPS", "0.01")),
    )
    parser.add_argument(
        "--ipopt-monitor-dir",
        default=os.environ.get("SMB_IPOPT_MONITOR_DIR", str(REPO_ROOT / "artifacts" / "ipopt_live")),
    )
    parser.add_argument(
        "--executive-watchdog-enabled",
        action="store_true",
        default=os.environ.get("SMB_EXECUTIVE_WATCHDOG_ENABLED", "0") == "1",
    )
    parser.add_argument(
        "--executive-watchdog-min-iters",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_WATCHDOG_MIN_ITERS", "80")),
    )
    parser.add_argument(
        "--executive-watchdog-stall-windows",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_WATCHDOG_STALL_WINDOWS", "2")),
    )
    parser.add_argument(
        "--executive-watchdog-max-inf-du",
        type=float,
        default=float(os.environ.get("SMB_EXECUTIVE_WATCHDOG_MAX_INF_DU", "0")),
    )
    parser.add_argument(
        "--executive-watchdog-max-mumps-realloc",
        type=int,
        default=int(os.environ.get("SMB_EXECUTIVE_WATCHDOG_MAX_MUMPS_REALLOC", "0")),
    )
    parser.add_argument("--tee", action="store_true")

    parser.add_argument("--nc", default="1,2,3,2")
    parser.add_argument("--nc-library", default="1,2,3,2;2,2,2,2")
    parser.add_argument("--f1", type=float, default=2.2)
    parser.add_argument("--fdes", type=float, default=1.2)
    parser.add_argument("--fex", type=float, default=0.9)
    parser.add_argument("--ffeed", type=float, default=1.3)
    parser.add_argument("--fraf", type=float)
    parser.add_argument("--tstep", type=float, default=9.4)
    parser.add_argument("--f1-library", default="2.2")
    parser.add_argument("--fdes-library", default="1.2")
    parser.add_argument("--fex-library", default="0.9")
    parser.add_argument("--ffeed-library", default="1.3")
    parser.add_argument("--tstep-library", default="9.4")

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
    parser.add_argument("--tstep-bounds", default="8.0,12.0")
    parser.add_argument("--ffeed-bounds", default="0.5,2.5")
    parser.add_argument("--fdes-bounds", default="0.5,2.5")
    parser.add_argument("--fex-bounds", default="0.5,2.5")
    parser.add_argument("--fraf-bounds", default="0.5,2.5")
    parser.add_argument("--f1-bounds", default="0.5,5.0")
    parser.add_argument("--f1-min", type=float, default=0.5)
    parser.add_argument("--f1-max", type=float)
    parser.add_argument("--f1-max-flow", type=float, default=5.0)
    parser.add_argument("--meoh-max-raff-wt", type=float, default=0.10)
    parser.add_argument("--water-max-ex-wt", type=float, default=0.05)
    parser.add_argument("--water-max-zone1-entry-wt", type=float, default=0.01)
    parser.add_argument("--seed-library", default="notebook")
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
        elif args.stage == "flow-screen":
            payload = run_flow_screen(args)
        elif args.stage == "optimize-layouts":
            payload = run_optimize_layouts(args)
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
