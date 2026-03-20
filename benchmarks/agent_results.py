from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple


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

    purity_min = float(os.environ.get("SMB_TARGET_PURITY_EX_MEOH_FREE", "0.60"))
    rga_min = float(os.environ.get("SMB_TARGET_RECOVERY_GA", "0.75"))
    rma_min = float(os.environ.get("SMB_TARGET_RECOVERY_MA", "0.75"))

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


def rank_any_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked = sorted(results, key=search_score, reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank_any"] = idx
    return ranked


def deterministic_select(tasks: List[Dict[str, object]], tried: set) -> int:
    for idx, task in enumerate(tasks):
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key not in tried:
            return idx
    return 0


def bootstrap_reference_select(tasks: List[Dict[str, object]], tried: set) -> int:
    for idx, task in enumerate(tasks):
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key in tried:
            continue
        if is_reference_seed_name(task.get("seed_name")):
            return idx
    return deterministic_select(tasks, tried)


def is_reference_seed_name(seed_name: object) -> bool:
    return str(seed_name or "").strip().lower() == "reference"


def low_fidelity_limits(args: object) -> Dict[str, int]:
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


def is_low_fidelity_result(result: Dict[str, object], args: object) -> bool:
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
    args: object,
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
    args: object,
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
    tried: set,
) -> Optional[int]:
    for idx in ranked_reference_indices(tasks):
        task = tasks[idx]
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key not in tried:
            return idx
    return None


def ranked_reference_indices(tasks: List[Dict[str, object]]) -> List[int]:
    return [idx for idx, task in enumerate(tasks) if str(task.get("seed_name", "")).strip().lower() == "reference"]


def has_any_feasible(results: List[Dict[str, object]]) -> bool:
    return any(bool(item.get("feasible")) for item in results)
