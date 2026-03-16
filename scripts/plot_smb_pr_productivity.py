#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
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
        description="Plot Purity vs Recovery colored by Productivity."
    )
    p.add_argument("--db", required=True, help="Path to sqlite DB")
    p.add_argument("--run-name", required=True, help="agent_run_name")
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--title", default="Purity vs Recovery (Color = Productivity)")
    p.add_argument("--purity-target", type=float, default=0.85)
    p.add_argument("--recovery-target", type=float, default=0.85)
    p.add_argument("--prod-vmin", type=float, default=0.0)
    p.add_argument("--prod-vmax", type=float, default=0.02)
    p.add_argument("--label-k", type=int, default=5, help="First/last K labels")
    return p.parse_args()


def load_rows(db_path: Path, run_name: str) -> List[Row]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT
                purity,
                recovery_ga,
                recovery_ma,
                productivity,
                normalized_total_violation
            FROM simulation_results
            WHERE agent_run_name = ?
              AND purity IS NOT NULL
              AND recovery_ga IS NOT NULL
              AND recovery_ma IS NOT NULL
              AND productivity IS NOT NULL
            ORDER BY id ASC
            """,
            (run_name,),
        ).fetchall()
    finally:
        conn.close()

    parsed: List[Row] = []
    for i, r in enumerate(rows, start=1):
        parsed.append(
            Row(
                idx=i,
                purity=float(r[0]),
                recovery_ga=float(r[1]),
                recovery_ma=float(r[2]),
                productivity=float(r[3]),
                violation=float(r[4]) if r[4] is not None else None,
            )
        )
    return parsed


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
    db = Path(args.db).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")
    if args.prod_vmax <= args.prod_vmin:
        raise ValueError("prod-vmax must be > prod-vmin")

    rows = load_rows(db, args.run_name)
    if not rows:
        raise RuntimeError(f"No rows found for run_name={args.run_name}")

    # Display as percentages with requested axis orientation:
    # x = Recovery (%), y = Purity (%)
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
        s=98,
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

    k = max(0, int(args.label_k))
    label_idxs = sorted(set(range(min(k, len(rows)))) | set(range(max(0, len(rows) - k), len(rows))))
    for ridx in label_idxs:
        r = rows[ridx]
        px = r.recovery * 100.0
        py = r.purity * 100.0
        dot_color = cmap(norm(r.productivity))
        ax.scatter(
            [px],
            [py],
            s=320,
            facecolors="none",
            edgecolors="black",
            linewidths=1.20,
            zorder=11,
        )
        ax.scatter(
            [px],
            [py],
            s=250,
            c=[dot_color],
            edgecolors="white",
            linewidths=0.4,
            zorder=11.5,
        )
        lum = 0.299 * dot_color[0] + 0.587 * dot_color[1] + 0.114 * dot_color[2]
        txt_color = "black" if lum > 0.62 else "white"
        outline = "white" if txt_color == "black" else "black"
        txt = ax.text(
            px,
            py,
            str(r.idx),
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=txt_color,
            zorder=12,
            clip_on=True,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.3, foreground=outline, alpha=0.8)])

    purity_target_pct = args.purity_target * 100.0
    recovery_target_pct = args.recovery_target * 100.0
    ax.axvline(recovery_target_pct, color="#ffa34d", linestyle="--", linewidth=1.6)
    ax.axhline(purity_target_pct, color="#4f83cc", linestyle="--", linewidth=1.6)
    if 1.0 > args.purity_target and 1.0 > args.recovery_target:
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

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Recovery (%)", fontsize=14)
    ax.set_ylabel("Purity (%)", fontsize=14)
    ax.set_title(args.title, fontsize=19, pad=8)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.28)

    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=star_color,
            markeredgecolor="black",
            markersize=13,
            linewidth=0,
            label="Best productivity / closest constraints (star color = productivity)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="#cccccc",
            markeredgecolor="black",
            markersize=9,
            linewidth=0,
            label="First/last iterations (numbered dots)",
        ),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
