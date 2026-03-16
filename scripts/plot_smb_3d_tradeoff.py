#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Row:
    idx: int
    db_id: int
    candidate_run_name: str
    nc: Optional[str]
    seed_name: Optional[str]
    status: Optional[str]
    productivity: float
    purity: float
    recovery_ga: float
    recovery_ma: float
    violation: Optional[float]

    @property
    def recovery_min(self) -> float:
        return min(self.recovery_ga, self.recovery_ma)

    @property
    def recovery_avg(self) -> float:
        return 0.5 * (self.recovery_ga + self.recovery_ma)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D SMB tradeoff plot: purity (x), recovery (y), productivity (z)."
    )
    parser.add_argument("--db", required=True, help="Path to sqlite DB containing simulation_results.")
    parser.add_argument("--run-name", required=True, help="agent_run_name to plot.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: artifacts/plots/<run-name>_3d_tradeoff.png",
    )
    parser.add_argument("--purity-target", type=float, default=0.85)
    parser.add_argument("--recovery-target", type=float, default=0.85)
    parser.add_argument(
        "--recovery-axis",
        choices=("min", "ga", "ma", "avg"),
        default="min",
        help="Recovery metric for Y axis.",
    )
    parser.add_argument(
        "--no-threshold-planes",
        action="store_true",
        help="Disable plotting threshold planes at purity/recovery targets.",
    )
    parser.add_argument(
        "--productivity-vmin",
        type=float,
        default=0.0,
        help="Lower bound for productivity colorbars (default: 0.0).",
    )
    parser.add_argument(
        "--productivity-vmax",
        type=float,
        default=0.02,
        help="Upper bound for productivity colorbars (default: 0.02).",
    )
    parser.add_argument(
        "--title-label",
        default=None,
        help="Optional title prefix label (e.g., 'Single Scientist' or 'Two Scientists').",
    )
    return parser.parse_args()


def load_rows(db_path: Path, run_name: str) -> List[Row]:
    conn = sqlite3.connect(str(db_path))
    try:
        q = """
        SELECT
            id,
            candidate_run_name,
            nc,
            seed_name,
            status,
            productivity,
            purity,
            recovery_ga,
            recovery_ma,
            normalized_total_violation
        FROM simulation_results
        WHERE agent_run_name = ?
          AND productivity IS NOT NULL
          AND purity IS NOT NULL
          AND recovery_ga IS NOT NULL
          AND recovery_ma IS NOT NULL
        ORDER BY id ASC
        """
        rows = conn.execute(q, (run_name,)).fetchall()
    finally:
        conn.close()

    parsed: List[Row] = []
    for i, r in enumerate(rows, start=1):
        parsed.append(
            Row(
                idx=i,
                db_id=int(r[0]),
                candidate_run_name=str(r[1]),
                nc=str(r[2]) if r[2] is not None else None,
                seed_name=str(r[3]) if r[3] is not None else None,
                status=str(r[4]) if r[4] is not None else None,
                productivity=float(r[5]),
                purity=float(r[6]),
                recovery_ga=float(r[7]),
                recovery_ma=float(r[8]),
                violation=float(r[9]) if r[9] is not None else None,
            )
        )
    return parsed


def is_feasible(row: Row, purity_target: float, recovery_target: float) -> bool:
    return (
        row.purity >= purity_target
        and row.recovery_ga >= recovery_target
        and row.recovery_ma >= recovery_target
    )


def constraint_gap(row: Row, purity_target: float, recovery_target: float) -> float:
    return (
        max(0.0, purity_target - row.purity)
        + max(0.0, recovery_target - row.recovery_ga)
        + max(0.0, recovery_target - row.recovery_ma)
    )


def pick_star(rows: List[Row], purity_target: float, recovery_target: float) -> Row:
    feasible = [r for r in rows if is_feasible(r, purity_target, recovery_target)]
    if feasible:
        return max(feasible, key=lambda r: (r.productivity, -(r.violation or 0.0)))
    return min(
        rows,
        key=lambda r: (
            constraint_gap(r, purity_target, recovery_target),
            -r.productivity,
            r.violation if r.violation is not None else 1e9,
        ),
    )


def recovery_value(row: Row, recovery_axis: str) -> float:
    if recovery_axis == "ga":
        return row.recovery_ga
    if recovery_axis == "ma":
        return row.recovery_ma
    if recovery_axis == "avg":
        return row.recovery_avg
    return row.recovery_min


def padded_limits(values: np.ndarray, pad_frac: float = 0.06) -> tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, 1e-12)
    pad = span * pad_frac
    return vmin - pad, vmax + pad


def compact_label(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    head = max_len // 2 - 2
    tail = max_len - head - 3
    return f"{text[:head]}...{text[-tail:]}"


def integer_ticks(vmin: float, vmax: float, max_ticks: int = 6) -> np.ndarray:
    lo = int(round(vmin))
    hi = int(round(vmax))
    if lo == hi:
        return np.array([lo], dtype=int)
    count = max(2, min(max_ticks, hi - lo + 1))
    ticks = np.linspace(lo, hi, num=count)
    ticks = np.unique(np.round(ticks).astype(int))
    if ticks[0] != lo:
        ticks = np.insert(ticks, 0, lo)
    if ticks[-1] != hi:
        ticks = np.append(ticks, hi)
    return ticks


def infer_title_label(run_name: str, explicit: Optional[str]) -> str:
    if explicit:
        ex = explicit.strip().lower()
        if "single" in ex and "scient" in ex:
            return "Single-Scientist-Driven SMB Optimization"
        if "two" in ex and "scient" in ex:
            return "Two-Scientists-Driven SMB Optimization"
        return explicit
    rn = run_name.lower()
    if "single" in rn:
        return "Single-Scientist-Driven SMB Optimization"
    if "two_scientists" in rn or "two-scientists" in rn:
        return "Two-Scientists-Driven SMB Optimization"
    return "SMB Optimization"


def main() -> int:
    args = parse_args()
    if args.productivity_vmax <= args.productivity_vmin:
        raise ValueError(
            f"Invalid productivity color range: "
            f"[{args.productivity_vmin}, {args.productivity_vmax}]"
        )
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    rows = load_rows(db_path, args.run_name)
    if not rows:
        raise RuntimeError(
            f"No plottable rows found for run_name='{args.run_name}' in {db_path}"
        )

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (REPO_ROOT / "artifacts" / "plots" / f"{args.run_name}_3d_tradeoff.png")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title_label = infer_title_label(args.run_name, args.title_label)

    star = pick_star(rows, args.purity_target, args.recovery_target)

    x = np.array([r.purity for r in rows], dtype=float)
    y = np.array([recovery_value(r, args.recovery_axis) for r in rows], dtype=float)
    z = np.array([r.productivity for r in rows], dtype=float)
    order = np.array([r.idx for r in rows], dtype=float)
    xlim = padded_limits(x)
    ylim = padded_limits(y)
    zlim = padded_limits(z)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(order.min(), order.max())

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        x,
        y,
        z,
        c=order,
        cmap=cmap,
        norm=norm,
        s=42,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.25,
    )

    if not args.no_threshold_planes:
        # Plane 1: purity = target
        y_span = np.linspace(float(y.min()), float(y.max()), 2)
        z_span = np.linspace(float(z.min()), float(z.max()), 2)
        y_grid, z_grid = np.meshgrid(y_span, z_span)
        x_grid = np.full_like(y_grid, float(args.purity_target))
        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            alpha=0.14,
            color="#4f83cc",
            edgecolor="none",
            zorder=1,
        )

        # Plane 2: recovery = target
        x_span = np.linspace(float(x.min()), float(x.max()), 2)
        x_grid2, z_grid2 = np.meshgrid(x_span, z_span)
        y_grid2 = np.full_like(x_grid2, float(args.recovery_target))
        ax.plot_surface(
            x_grid2,
            y_grid2,
            z_grid2,
            alpha=0.14,
            color="#ffa34d",
            edgecolor="none",
            zorder=1,
        )

        # Highlight points satisfying either threshold on axis metrics.
        either_mask = (x >= float(args.purity_target)) | (y >= float(args.recovery_target))
        if np.any(either_mask):
            ax.scatter(
                x[either_mask],
                y[either_mask],
                z[either_mask],
                s=84,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                alpha=0.95,
                zorder=9,
            )

    star_x = star.purity
    star_y = recovery_value(star, args.recovery_axis)
    star_z = star.productivity
    ax.scatter(
        [star_x],
        [star_y],
        [star_z],
        marker="*",
        s=320,
        c="blue",
        edgecolor="black",
        linewidth=1.0,
        label="Best productivity / closest constraints",
        zorder=10,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.75)
    cbar.set_label("Simulation iteration")
    cbar.set_ticks(integer_ticks(order.min(), order.max(), max_ticks=6))

    recovery_label = {
        "min": "Recovery (min of GA/MA)",
        "ga": "Recovery (GA)",
        "ma": "Recovery (MA)",
        "avg": "Recovery (avg GA/MA)",
    }[args.recovery_axis]
    ax.set_xlabel("Purity")
    ax.set_ylabel(recovery_label)
    ax.set_zlabel("Productivity")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # User preference: purity increases in the opposite visual direction.
    ax.invert_xaxis()
    ax.set_title(title_label)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=12,
            label="Best productivity / closest constraints",
        )
    ]
    if not args.no_threshold_planes:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="none",
                markersize=9,
                linewidth=0,
                label="Satisfies purity>=target OR recovery>=target",
            )
        )
    ax.legend(handles=legend_handles, loc="upper left")

    annotation = (
        f"Star: id={star.db_id}, nc={star.nc}, seed={star.seed_name}, "
        f"prod={star.productivity:.4g}, purity={star.purity:.4g}, "
        f"rGA={star.recovery_ga:.4g}, rMA={star.recovery_ma:.4g}, "
        f"gap={constraint_gap(star, args.purity_target, args.recovery_target):.4g}"
    )
    fig.text(0.02, 0.02, annotation, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    # 2D projections
    proj_out_path = out_path.with_name(f"{out_path.stem}_2d_projections{out_path.suffix}")
    fig2, axes = plt.subplots(1, 3, figsize=(19, 7.2))

    either_mask = (x >= float(args.purity_target)) | (y >= float(args.recovery_target))

    # Projection 1: Purity vs Recovery
    ax1 = axes[0]
    sc1 = ax1.scatter(
        x,
        y,
        c=order,
        cmap=cmap,
        norm=norm,
        s=48,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.3,
    )
    if np.any(either_mask):
        ax1.scatter(
            x[either_mask],
            y[either_mask],
            s=74,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
        )
    ax1.scatter([star_x], [star_y], marker="*", s=220, c="blue", edgecolors="black", linewidths=0.9)
    ax1.axvline(float(args.purity_target), color="#4f83cc", linestyle="--", linewidth=1.3)
    ax1.axhline(float(args.recovery_target), color="#ffa34d", linestyle="--", linewidth=1.3)
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    if xlim[1] > float(args.purity_target) and ylim[1] > float(args.recovery_target):
        feasible_rect = Rectangle(
            (float(args.purity_target), float(args.recovery_target)),
            max(0.0, xlim[1] - float(args.purity_target)),
            max(0.0, ylim[1] - float(args.recovery_target)),
            facecolor="#90caf9",
            alpha=0.12,
            edgecolor="none",
            zorder=0,
        )
        ax1.add_patch(feasible_rect)
    ax1.set_xlabel("Purity", fontsize=14)
    ax1.set_ylabel(recovery_label, fontsize=14)
    ax1.set_title("Purity vs Recovery", fontsize=18, pad=8)
    ax1.grid(True, alpha=0.28)
    ax1.tick_params(axis="both", labelsize=13)

    # Projection 2: Purity vs Productivity
    ax2 = axes[1]
    ax2.scatter(
        x,
        z,
        c=order,
        cmap=cmap,
        norm=norm,
        s=48,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.3,
    )
    if np.any(either_mask):
        ax2.scatter(
            x[either_mask],
            z[either_mask],
            s=74,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
        )
    ax2.scatter([star_x], [star_z], marker="*", s=220, c="blue", edgecolors="black", linewidths=0.9)
    ax2.axvline(float(args.purity_target), color="#4f83cc", linestyle="--", linewidth=1.3)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*zlim)
    ax2.set_xlabel("Purity", fontsize=14)
    ax2.set_ylabel("Productivity", fontsize=14)
    ax2.set_title("Purity vs Productivity", fontsize=18, pad=8)
    ax2.grid(True, alpha=0.28)
    ax2.tick_params(axis="both", labelsize=13)

    # Projection 3: Recovery vs Productivity
    ax3 = axes[2]
    ax3.scatter(
        y,
        z,
        c=order,
        cmap=cmap,
        norm=norm,
        s=48,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.3,
    )
    if np.any(either_mask):
        ax3.scatter(
            y[either_mask],
            z[either_mask],
            s=74,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
        )
    ax3.scatter([star_y], [star_z], marker="*", s=220, c="blue", edgecolors="black", linewidths=0.9)
    ax3.axvline(float(args.recovery_target), color="#ffa34d", linestyle="--", linewidth=1.3)
    ax3.set_xlim(*ylim)
    ax3.set_ylim(*zlim)
    ax3.set_xlabel(recovery_label, fontsize=14)
    ax3.set_ylabel("Productivity", fontsize=14)
    ax3.set_title("Recovery vs Productivity", fontsize=18, pad=8)
    ax3.grid(True, alpha=0.28)
    ax3.tick_params(axis="both", labelsize=13)

    fig2.suptitle(title_label, fontsize=22, y=0.97)
    legend_handles_2d = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=13,
            linewidth=0,
            label="Best productivity / closest constraints",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=10,
            linewidth=0,
            label="Satisfies purity>=target OR recovery>=target",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#4f83cc",
            linestyle="--",
            linewidth=1.6,
            label="Purity target line",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#ffa34d",
            linestyle="--",
            linewidth=1.6,
            label="Recovery target line",
        ),
        Patch(
            facecolor="#90caf9",
            edgecolor="none",
            alpha=0.18,
            label="Feasible region (Purity vs Recovery panel)",
        ),
    ]
    fig2.legend(
        handles=legend_handles_2d,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.14),
    )

    cax2 = fig2.add_axes([0.30, 0.045, 0.40, 0.05])
    cbar2 = fig2.colorbar(
        sc1,
        cax=cax2,
        orientation="horizontal",
    )
    cbar2.set_label("Simulation iteration")
    cbar2.set_ticks(integer_ticks(order.min(), order.max(), max_ticks=6))
    cbar2.ax.tick_params(labelsize=12)
    cbar2.ax.xaxis.label.set_size(13)
    fig2.subplots_adjust(top=0.78, bottom=0.22, wspace=0.30, right=0.98, left=0.06)
    fig2.savefig(proj_out_path, dpi=220)
    plt.close(fig2)

    # Additional 2D plot: Purity vs Recovery with productivity colorbar
    pr_prod_out_path = out_path.with_name(
        f"{out_path.stem}_purity_recovery_productivity{out_path.suffix}"
    )
    fig3, axp = plt.subplots(figsize=(9.4, 7.2))
    prod_norm = plt.Normalize(
        float(args.productivity_vmin),
        float(args.productivity_vmax),
        clip=True,
    )
    # Keep original geometry: x-axis = Purity, y-axis = Recovery.
    # Display as percentages for readability.
    x_purity_pct = x * 100.0
    y_recovery_pct = y * 100.0
    sc3 = axp.scatter(
        x_purity_pct,
        y_recovery_pct,
        c=z,
        cmap="plasma_r",
        norm=prod_norm,
        s=72,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.35,
    )
    star_prod_color = plt.get_cmap("plasma_r")(prod_norm(star_z))
    axp.scatter(
        [star_x * 100.0],
        [star_y * 100.0],
        marker="*",
        s=280,
        c=[star_prod_color],
        edgecolors="black",
        linewidths=1.0,
        zorder=10,
    )
    # Label first 5, last 5, and all points in the feasible shaded area
    # (purity>=target AND plotted recovery>=target) as numbered dots.
    k_labels = 5
    first_last_idxs = set(range(min(k_labels, len(rows)))) | set(
        range(max(0, len(rows) - k_labels), len(rows))
    )
    satisfied_idxs = {
        i
        for i, r in enumerate(rows)
        if (r.purity >= float(args.purity_target))
        and (recovery_value(r, args.recovery_axis) >= float(args.recovery_target))
    }
    label_idxs = sorted(first_last_idxs | satisfied_idxs)
    highlight_x: list[float] = []
    highlight_y: list[float] = []
    highlight_colors: list[tuple[float, float, float, float]] = []
    highlight_labels: list[int] = []
    x_lo, x_hi = 0.0, 1.0
    y_lo, y_hi = 0.0, 1.0
    for ridx in label_idxs:
        if ridx < 0 or ridx >= len(rows):
            continue
        r = rows[ridx]
        purity_raw = r.purity
        recovery_raw = recovery_value(r, args.recovery_axis)
        # Keep numbered overlays within visible axis bounds to avoid layout blow-up.
        if not (x_lo <= purity_raw <= x_hi and y_lo <= recovery_raw <= y_hi):
            continue
        highlight_x.append(purity_raw * 100.0)
        highlight_y.append(recovery_raw * 100.0)
        highlight_colors.append(plt.get_cmap("plasma_r")(prod_norm(r.productivity)))
        highlight_labels.append(r.idx)

    if highlight_x:
        axp.scatter(
            highlight_x,
            highlight_y,
            s=200,
            facecolors="none",
            edgecolors="black",
            linewidths=1.35,
            zorder=11,
        )
        # Re-draw filled centers for labeled points so base data remains visible.
        axp.scatter(
            highlight_x,
            highlight_y,
            s=58,
            c=highlight_colors,
            edgecolors="white",
            linewidths=0.4,
            zorder=11.5,
        )
        for hx, hy, hlabel, hcolor in zip(highlight_x, highlight_y, highlight_labels, highlight_colors):
            lum = 0.299 * hcolor[0] + 0.587 * hcolor[1] + 0.114 * hcolor[2]
            text_color = "black" if lum > 0.62 else "white"
            outline_color = "white" if text_color == "black" else "black"
            ty = min(99.0, hy + 1.6)
            txt = axp.text(
                hx,
                ty,
                str(hlabel),
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color=text_color,
                zorder=12,
            )
            txt.set_clip_on(True)
            txt.set_path_effects(
                [pe.withStroke(linewidth=1.3, foreground=outline_color, alpha=0.8)]
            )
    purity_target_pct = float(args.purity_target) * 100.0
    recovery_target_pct = float(args.recovery_target) * 100.0
    axp.axvline(purity_target_pct, color="#4f83cc", linestyle="--", linewidth=1.6)
    axp.axhline(recovery_target_pct, color="#ffa34d", linestyle="--", linewidth=1.6)
    if 100.0 > purity_target_pct and 100.0 > recovery_target_pct:
        feasible_rect3 = Rectangle(
            (purity_target_pct, recovery_target_pct),
            max(0.0, 100.0 - purity_target_pct),
            max(0.0, 100.0 - recovery_target_pct),
            facecolor="#90caf9",
            alpha=0.12,
            edgecolor="none",
            zorder=0,
        )
        axp.add_patch(feasible_rect3)
    axp.set_xlim(0.0, 100.0)
    axp.set_ylim(0.0, 100.0)
    axp.set_xlabel("Purity (%)", fontsize=14)
    axp.set_ylabel("Recovery (%)", fontsize=14)
    axp.set_title(title_label, fontsize=19, pad=8)
    axp.tick_params(axis="both", labelsize=13)
    axp.grid(True, alpha=0.28)

    legend_handles_3 = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=star_prod_color,
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
            label="First/last + feasible-area points (numbered dots)",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#4f83cc",
            linestyle="--",
            linewidth=1.6,
            label="Purity target line",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#ffa34d",
            linestyle="--",
            linewidth=1.6,
            label="Recovery target line",
        ),
        Patch(
            facecolor="#90caf9",
            edgecolor="none",
            alpha=0.18,
            label="Feasible region",
        ),
    ]
    axp.legend(handles=legend_handles_3, loc="lower left", frameon=False, fontsize=12)

    cbar3 = fig3.colorbar(sc3, ax=axp, pad=0.02, fraction=0.05)
    cbar3.set_label("Productivity", fontsize=13)
    cbar3.set_ticks(np.linspace(float(args.productivity_vmin), float(args.productivity_vmax), 5))
    cbar3.ax.tick_params(labelsize=12)
    fig3.tight_layout()
    fig3.savefig(pr_prod_out_path, dpi=220)
    plt.close(fig3)

    # Additional summary plot: first vs last simulation comparison
    first_last_out_path = out_path.with_name(f"{out_path.stem}_first_vs_last{out_path.suffix}")
    first_row = rows[0]
    last_row = rows[-1]
    first_x = first_row.purity
    first_y = recovery_value(first_row, args.recovery_axis)
    last_x = last_row.purity
    last_y = recovery_value(last_row, args.recovery_axis)

    fig4, (ax4a, ax4b) = plt.subplots(
        1,
        2,
        figsize=(14, 6.4),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    # Left panel: location in Purity-Recovery space
    ax4a.scatter(x, y, s=34, c="#b0b0b0", alpha=0.45, edgecolors="none", label="All simulations")
    ax4a.scatter(
        [first_x],
        [first_y],
        s=190,
        marker="o",
        c="#c62828",
        edgecolors="black",
        linewidths=0.8,
        label=f"First (#{first_row.idx})",
        zorder=5,
    )
    ax4a.scatter(
        [last_x],
        [last_y],
        s=190,
        marker="o",
        c="#2e7d32",
        edgecolors="black",
        linewidths=0.8,
        label=f"Last (#{last_row.idx})",
        zorder=6,
    )
    ax4a.plot([first_x, last_x], [first_y, last_y], color="#616161", linewidth=1.2, linestyle="--", alpha=0.8)
    ax4a.axvline(float(args.purity_target), color="#4f83cc", linestyle="--", linewidth=1.4)
    ax4a.axhline(float(args.recovery_target), color="#ffa34d", linestyle="--", linewidth=1.4)
    if xlim[1] > float(args.purity_target) and ylim[1] > float(args.recovery_target):
        feasible_rect4 = Rectangle(
            (float(args.purity_target), float(args.recovery_target)),
            max(0.0, xlim[1] - float(args.purity_target)),
            max(0.0, ylim[1] - float(args.recovery_target)),
            facecolor="#90caf9",
            alpha=0.10,
            edgecolor="none",
            zorder=0,
        )
        ax4a.add_patch(feasible_rect4)
    ax4a.set_xlim(*xlim)
    ax4a.set_ylim(*ylim)
    ax4a.set_xlabel("Purity", fontsize=13)
    ax4a.set_ylabel(recovery_label, fontsize=13)
    ax4a.set_title("First vs Last in Purity-Recovery Space", fontsize=15)
    ax4a.grid(True, alpha=0.25)
    ax4a.tick_params(axis="both", labelsize=12)
    ax4a.legend(loc="lower left", fontsize=11, frameon=False)

    # Right panel: metric comparison
    labels = ["Purity", recovery_label.replace(" (", "\n("), "Productivity"]
    first_vals = [first_row.purity, first_y, first_row.productivity]
    last_vals = [last_row.purity, last_y, last_row.productivity]
    idxs = np.arange(len(labels))
    w = 0.34
    ax4b.bar(idxs - w / 2, first_vals, width=w, color="#c62828", alpha=0.9, label=f"First (#{first_row.idx})")
    ax4b.bar(idxs + w / 2, last_vals, width=w, color="#2e7d32", alpha=0.9, label=f"Last (#{last_row.idx})")
    ax4b.set_xticks(idxs)
    ax4b.set_xticklabels(labels, fontsize=11)
    ax4b.set_title("Metric Comparison", fontsize=15)
    ax4b.grid(True, axis="y", alpha=0.25)
    ax4b.tick_params(axis="y", labelsize=12)
    ax4b.legend(fontsize=11, frameon=False)

    fig4.suptitle(title_label, fontsize=18, y=0.98)
    fig4.tight_layout()
    fig4.savefig(first_last_out_path, dpi=220)
    plt.close(fig4)

    print(f"Saved: {out_path}")
    print(f"Saved: {proj_out_path}")
    print(f"Saved: {pr_prod_out_path}")
    print(f"Saved: {first_last_out_path}")
    print(f"Rows plotted: {len(rows)}")
    print(annotation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
