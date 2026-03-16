#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


@dataclass
class Row:
    idx: int
    purity: float
    recovery_ga: float
    recovery_ma: float
    productivity: float
    violation: Optional[float]

    @property
    def recovery(self) -> float:
        return min(self.recovery_ga, self.recovery_ma)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot MINLP stage run as Recovery(%) vs Purity(%) colored by Productivity."
    )
    p.add_argument("--stage-json", required=True, help="Path to optimize-layouts.*.json")
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--run-id", default=None, help="Optional run ID override for title")
    p.add_argument("--purity-target", type=float, default=0.85)
    p.add_argument("--recovery-target", type=float, default=0.85)
    p.add_argument("--prod-vmin", type=float, default=0.0)
    p.add_argument("--prod-vmax", type=float, default=0.02)
    p.add_argument(
        "--recovery-xmax",
        type=float,
        default=105.0,
        help="Upper bound for Recovery(%%) x-axis (default: 105).",
    )
    p.add_argument(
        "--physical-bounds-only",
        action="store_true",
        help="Keep only points with 0<=purity<=1 and 0<=recovery_ga/recovery_ma<=1.",
    )
    return p.parse_args()


def infer_run_id(stage_json_path: Path, override: Optional[str]) -> str:
    if override:
        return str(override)
    m = re.search(r"optimize-layouts\.(\d+)\.", stage_json_path.name)
    if m:
        return m.group(1)
    return "unknown"


def _extract_metrics(result: dict) -> Optional[dict]:
    # Prefer validated metrics if present; fallback to provisional (typical for solver_error).
    if isinstance(result.get("metrics"), dict):
        return result["metrics"]
    prov = result.get("provisional")
    if isinstance(prov, dict) and isinstance(prov.get("metrics"), dict):
        return prov["metrics"]
    return None


def _extract_violation(result: dict) -> Optional[float]:
    for key in (
        "normalized_total_violation",
        "violation",
    ):
        val = result.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    slacks = result.get("constraint_slacks")
    if isinstance(slacks, dict):
        val = slacks.get("normalized_total_violation")
        if isinstance(val, (int, float)):
            return float(val)
    prov = result.get("provisional")
    if isinstance(prov, dict):
        slacks = prov.get("constraint_slacks")
        if isinstance(slacks, dict):
            val = slacks.get("normalized_total_violation")
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _within_unit_interval(v: float) -> bool:
    return 0.0 <= v <= 1.0


def load_rows(stage_json_path: Path, physical_bounds_only: bool = False) -> tuple[List[Row], int]:
    payload = json.loads(stage_json_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Expected `results` list in stage JSON.")

    rows: List[Row] = []
    filtered_out = 0
    for i, res in enumerate(results, start=1):
        if not isinstance(res, dict):
            continue
        metrics = _extract_metrics(res)
        if not metrics:
            continue
        try:
            purity = float(metrics["purity_ex_meoh_free"])
            rga = float(metrics["recovery_ex_GA"])
            rma = float(metrics["recovery_ex_MA"])
            prod = float(metrics["productivity_ex_ga_ma"])
        except Exception:
            continue
        if physical_bounds_only and (
            (not _within_unit_interval(purity))
            or (not _within_unit_interval(rga))
            or (not _within_unit_interval(rma))
        ):
            filtered_out += 1
            continue
        rows.append(
            Row(
                idx=i,
                purity=purity,
                recovery_ga=rga,
                recovery_ma=rma,
                productivity=prod,
                violation=_extract_violation(res),
            )
        )
    return rows, filtered_out


def is_feasible(r: Row, purity_target: float, recovery_target: float) -> bool:
    return (
        r.purity >= purity_target
        and r.recovery_ga >= recovery_target
        and r.recovery_ma >= recovery_target
    )


def gap(r: Row, purity_target: float, recovery_target: float) -> float:
    return (
        max(0.0, purity_target - r.purity)
        + max(0.0, recovery_target - r.recovery_ga)
        + max(0.0, recovery_target - r.recovery_ma)
    )


def pick_star(rows: List[Row], purity_target: float, recovery_target: float) -> Row:
    feasible = [r for r in rows if is_feasible(r, purity_target, recovery_target)]
    if feasible:
        return max(feasible, key=lambda r: (r.productivity, -(r.violation or 0.0)))
    return min(rows, key=lambda r: (gap(r, purity_target, recovery_target), -r.productivity))


def main() -> int:
    args = parse_args()
    stage_json = Path(args.stage_json).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not stage_json.exists():
        raise FileNotFoundError(f"Stage JSON not found: {stage_json}")
    if args.prod_vmax <= args.prod_vmin:
        raise ValueError("prod-vmax must be > prod-vmin")

    run_id = infer_run_id(stage_json, args.run_id)
    rows, filtered_out = load_rows(stage_json, physical_bounds_only=args.physical_bounds_only)
    if not rows:
        raise RuntimeError("No plottable MINLP rows found (missing required metrics).")

    x = np.array([r.recovery * 100.0 for r in rows], dtype=float)
    y = np.array([r.purity * 100.0 for r in rows], dtype=float)
    z = np.array([r.productivity for r in rows], dtype=float)
    star = pick_star(rows, args.purity_target, args.recovery_target)

    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    norm = plt.Normalize(args.prod_vmin, args.prod_vmax, clip=True)
    cmap = plt.get_cmap("plasma_r")

    sc = ax.scatter(
        x,
        y,
        c=z,
        cmap=cmap,
        norm=norm,
        s=68,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    star_color = cmap(norm(star.productivity))
    ax.scatter(
        [star.recovery * 100.0],
        [star.purity * 100.0],
        marker="*",
        s=280,
        c=[star_color],
        edgecolors="black",
        linewidths=1.0,
        zorder=10,
    )

    purity_target_pct = args.purity_target * 100.0
    recovery_target_pct = args.recovery_target * 100.0
    ax.axvline(recovery_target_pct, color="#ffa34d", linestyle="--", linewidth=1.6)
    ax.axhline(purity_target_pct, color="#4f83cc", linestyle="--", linewidth=1.6)
    if 100.0 > purity_target_pct and 100.0 > recovery_target_pct:
        ax.add_patch(
            Rectangle(
                (recovery_target_pct, purity_target_pct),
                100.0 - recovery_target_pct,
                100.0 - purity_target_pct,
                facecolor="#90caf9",
                alpha=0.12,
                edgecolor="none",
                zorder=1,
            )
        )

    ax.set_xlim(0.0, float(args.recovery_xmax))
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Recovery (%)", fontsize=14)
    ax.set_ylabel("Purity (%)", fontsize=14)
    ax.set_title(f"MINLP-Driven SMB Optimization", fontsize=19, pad=8)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.28)

    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=star_color, markeredgecolor="black", markersize=13, linewidth=0, label="Best productivity / closest constraints (star color = productivity)"),
        Line2D([0, 1], [0, 0], color="#ffa34d", linestyle="--", linewidth=1.6, label="Recovery target line"),
        Line2D([0, 1], [0, 0], color="#4f83cc", linestyle="--", linewidth=1.6, label="Purity target line"),
        Patch(facecolor="#90caf9", edgecolor="none", alpha=0.18, label="Feasible region"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=12)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Productivity", fontsize=13)
    cbar.set_ticks(np.linspace(args.prod_vmin, args.prod_vmax, 5))
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Rows plotted: {len(rows)}")
    if args.physical_bounds_only:
        print(f"Filtered non-physical points: {filtered_out}")
    print(f"Run ID: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
