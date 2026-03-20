from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .agent_results import (
    as_float,
    effective_violation,
    extract_metrics_with_validity,
    effective_flow,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def bottleneck_label(result: Dict[str, object]) -> str:
    metrics, _ = extract_metrics_with_validity(result)
    purity = as_float(metrics.get("purity_ex_meoh_free")) or 0.0
    rga = as_float(metrics.get("recovery_ex_GA")) or 0.0
    rma = as_float(metrics.get("recovery_ex_MA")) or 0.0
    status = str(result.get("status", "")).strip().lower()
    if status in {"solver_error", "error", "other"}:
        return "solver_error"
    gaps: List[str] = []
    if purity < 0.60:
        gaps.append("purity")
    if rga < 0.75:
        gaps.append("recovery_ga")
    if rma < 0.75:
        gaps.append("recovery_ma")
    if gaps:
        return ",".join(gaps)
    return "unknown"


def compact_result_record(result: Dict[str, object]) -> Dict[str, object]:
    flow = effective_flow(result) or {}
    metrics, metrics_validated = extract_metrics_with_validity(result)
    return {
        "run_name": str(result.get("run_name", "")),
        "nc": list(result.get("nc", [])),
        "seed_name": str(result.get("seed_name", "")),
        "status": str(result.get("status", "")),
        "feasible": bool(result.get("feasible", False)),
        "j_validated": as_float(result.get("J_validated")),
        "productivity": as_float(metrics.get("productivity_ex_ga_ma")),
        "purity": as_float(metrics.get("purity_ex_meoh_free")),
        "recovery_ga": as_float(metrics.get("recovery_ex_GA")),
        "recovery_ma": as_float(metrics.get("recovery_ex_MA")),
        "normalized_total_violation": effective_violation(result),
        "metrics_validated": metrics_validated,
        "flow": {
            "Ffeed": as_float(flow.get("Ffeed")),
            "F1": as_float(flow.get("F1")),
            "Fdes": as_float(flow.get("Fdes")),
            "Fex": as_float(flow.get("Fex")),
            "Fraf": as_float(flow.get("Fraf")),
            "tstep": as_float(flow.get("tstep")),
        },
    }


def build_evidence_pack(
    results: Sequence[Dict[str, object]],
    recent_limit: int = 5,
    feasible_limit: int = 3,
    infeasible_limit: int = 4,
) -> Dict[str, object]:
    ordered = [item for item in results if isinstance(item, dict)]
    recent_runs = [compact_result_record(item) for item in ordered[-recent_limit:]]

    feasible_rows = [
        item
        for item in ordered
        if bool(item.get("feasible")) and as_float(item.get("J_validated")) is not None
    ]
    feasible_rows = sorted(
        feasible_rows,
        key=lambda item: (
            as_float(item.get("J_validated")) if as_float(item.get("J_validated")) is not None else float("-inf"),
            as_float((item.get("metrics") or {}).get("productivity_ex_ga_ma"))
            if isinstance(item.get("metrics"), dict)
            else float("-inf"),
        ),
        reverse=True,
    )[:feasible_limit]
    top_feasible = [compact_result_record(item) for item in feasible_rows]

    infeasible_rows = [item for item in ordered if not bool(item.get("feasible"))]
    near_infeasible = sorted(
        infeasible_rows,
        key=lambda item: effective_violation(item),
    )[: infeasible_limit // 2 + 1]
    hard_failures = [
        item
        for item in infeasible_rows
        if str(item.get("status", "")).strip().lower() in {"solver_error", "error", "other", "failed"}
    ]
    ranked_hard_failures = sorted(
        hard_failures,
        key=lambda item: (effective_violation(item), str(item.get("run_name", ""))),
    )
    combined: List[Dict[str, object]] = []
    seen_runs: set[str] = set()
    for item in near_infeasible + ranked_hard_failures:
        run_name = str(item.get("run_name", ""))
        if run_name in seen_runs:
            continue
        seen_runs.add(run_name)
        record = compact_result_record(item)
        record["bottleneck"] = bottleneck_label(item)
        combined.append(record)
        if len(combined) >= infeasible_limit:
            break

    run_names = []
    for bucket in (recent_runs, top_feasible, combined):
        for item in bucket:
            run_name = str(item.get("run_name", "")).strip()
            if run_name:
                run_names.append(run_name)

    return {
        "recent_runs": recent_runs,
        "top_feasible": top_feasible,
        "top_infeasible": combined,
        "run_name_catalog": sorted(set(run_names)),
    }


def contains_run_reference(text_items: Sequence[str], run_names: Sequence[str]) -> bool:
    blob = " ".join(str(item) for item in text_items if item is not None)
    if not blob:
        return False
    return any(str(run_name) in blob for run_name in run_names if run_name)


def normalize_evidence_refs(value: object, max_items: int = 8) -> List[str]:
    refs = normalize_text_list(value, max_items=max_items)
    return [item.strip() for item in refs if str(item).strip()]


def build_evidence_fallback_items(evidence_pack: Dict[str, object], max_items: int = 8) -> List[str]:
    rows: List[Dict[str, object]] = []
    for key in ("recent_runs", "top_feasible", "top_infeasible"):
        value = evidence_pack.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append(item)
    items: List[str] = []
    seen_run_names: set[str] = set()
    for row in rows:
        run_name = str(row.get("run_name", "")).strip()
        if not run_name or run_name in seen_run_names:
            continue
        seen_run_names.add(run_name)
        items.append(
            "run_name="
            + run_name
            + f" status={row.get('status')} feasible={row.get('feasible')} "
            + f"prod={row.get('productivity')} purity={row.get('purity')} "
            + f"rGA={row.get('recovery_ga')} rMA={row.get('recovery_ma')} "
            + f"viol={row.get('normalized_total_violation')}"
        )
        if len(items) >= max_items:
            break
    if not items:
        catalog = normalize_text_list(evidence_pack.get("run_name_catalog"), max_items=max_items)
        for run_name in catalog:
            items.append(f"run_name={run_name} catalog_reference")
            if len(items) >= max_items:
                break
    return items


def coerce_evidence_list(
    value: object,
    evidence_pack: Dict[str, object],
    *,
    min_items: int = 2,
    max_items: int = 8,
) -> List[str]:
    evidence = normalize_text_list(value, max_items=max_items)
    fallback = build_evidence_fallback_items(evidence_pack, max_items=max_items)
    for item in fallback:
        if len(evidence) >= min_items:
            break
        if item not in evidence:
            evidence.append(item)
    while len(evidence) < min_items:
        evidence.append("bootstrap_evidence_pending")
    return evidence[:max_items]


def coerce_grounded_evidence_refs(
    value: object,
    run_names: Sequence[str],
    *,
    min_items: int = 1,
    max_items: int = 8,
) -> List[str]:
    refs = normalize_evidence_refs(value, max_items=max_items)
    catalog = [str(item).strip() for item in run_names if str(item).strip()]
    grounded: List[str] = []
    for ref in refs:
        if any((run_name == ref) or (run_name in ref) for run_name in catalog):
            grounded.append(ref)
    for run_name in catalog:
        if len(grounded) >= min_items:
            break
        grounded.append(run_name)
    deduped: List[str] = []
    seen: set[str] = set()
    for ref in grounded:
        if ref in seen:
            continue
        seen.add(ref)
        deduped.append(ref)
    return deduped[:max_items]


def evidence_refs_are_grounded(evidence_refs: Sequence[str], run_names: Sequence[str]) -> bool:
    refs = [str(item).strip() for item in evidence_refs if str(item).strip()]
    catalog = [str(item).strip() for item in run_names if str(item).strip()]
    if not refs or not catalog:
        return False
    for ref in refs:
        if not any((run_name == ref) or (run_name in ref) for run_name in catalog):
            return False
    return True


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


def budget_evidence_pack_json(evidence_pack: Dict[str, object], max_chars: int) -> str:
    """Serialize evidence pack to JSON within max_chars by dropping complete records.

    Unlike compact_prompt_block, this never slices mid-JSON-string — it removes
    whole records from least-important sections first so the output is always
    valid JSON the model can parse.
    """
    ep: Dict[str, object] = {
        "recent_runs": list(evidence_pack.get("recent_runs", [])),
        "top_feasible": list(evidence_pack.get("top_feasible", [])),
        "top_infeasible": list(evidence_pack.get("top_infeasible", [])),
        "run_name_catalog": list(evidence_pack.get("run_name_catalog", [])),
    }
    # Trim order: catalog (redundant), infeasible tail, feasible tail, recent tail
    trim_order = [
        ("run_name_catalog", -1),
        ("top_infeasible", -1),
        ("top_feasible", -1),
        ("recent_runs", 0),
    ]
    for _ in range(40):
        s = json.dumps(ep, separators=(",", ":"))
        if len(s) <= max_chars:
            return s
        trimmed = False
        for section, pop_index in trim_order:
            lst = ep[section]
            if isinstance(lst, list) and lst:
                lst.pop(pop_index)
                trimmed = True
                break
        if not trimmed:
            break
    # Last resort: hard truncate at a JSON-safe char boundary (still valid outer object)
    s = json.dumps(ep, separators=(",", ":"))
    return s[:max_chars]


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


def read_doc_excerpt(path: str, max_chars: int = 4000) -> str:
    p = Path(path)
    if not p.exists():
        return f"Missing file: {path}"
    text = p.read_text(encoding="utf-8")
    return compact_prompt_block(text, max_chars=max_chars, max_lines=200)


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


def hypothesis_matcher(heuristics_context: str, results: Optional[Sequence[Dict[str, object]]] = None) -> str:
    lines: List[str] = []
    capture = False
    for raw_line in (heuristics_context or "").splitlines():
        line = raw_line.rstrip()
        if line.startswith("HYPOTHESES"):
            capture = True
        if capture:
            if line.startswith("FAILURE MODES"):
                break
            lines.append(line)
    if results:
        recent = [item for item in results[-3:] if isinstance(item, dict)]
        if recent:
            lines.append("")
            lines.append("Recent hypothesis-linked outcomes:")
            for item in recent:
                lines.append(
                    f"- run={item.get('run_name')} status={item.get('status')} "
                    f"feasible={item.get('feasible')} viol={effective_violation(item):.6g}"
                )
    if not lines:
        return "No hypothesis context available."
    return compact_prompt_block("\n".join(lines), max_chars=1200, max_lines=36)


def failure_recovery_context(heuristics_context: str, results: Optional[Sequence[Dict[str, object]]] = None) -> str:
    lines: List[str] = []
    capture = False
    for raw_line in (heuristics_context or "").splitlines():
        line = raw_line.rstrip()
        if line.startswith("FAILURE MODES"):
            capture = True
        if capture:
            lines.append(line)
    if results:
        recent = [item for item in results[-3:] if isinstance(item, dict)]
        if recent:
            lines.append("")
            lines.append("Recent recovery signals:")
            for item in recent:
                status = str(item.get("status", "")).strip()
                feasible = bool(item.get("feasible"))
                if feasible and status == "ok":
                    continue
                lines.append(
                    f"- run={item.get('run_name')} status={status} feasible={feasible} "
                    f"viol={effective_violation(item):.6g}"
                )
    if not lines:
        return "No failure recovery context available."
    return compact_prompt_block("\n".join(lines), max_chars=1200, max_lines=36)


def apply_flow_adjustments(base_flow: Dict[str, float], flow_adjustments: Optional[Dict[str, float]]) -> Dict[str, float]:
    keys = ("Ffeed", "F1", "Fdes", "Fex", "Fraf", "tstep")
    adjusted = {key: float(base_flow.get(key, 0.0)) for key in keys}
    if not isinstance(flow_adjustments, dict):
        return adjusted
    for key in keys:
        delta = as_float(flow_adjustments.get(key))
        if delta is None:
            continue
        value = adjusted[key] + float(delta)
        if key == "tstep":
            adjusted[key] = max(1e-6, value)
        else:
            adjusted[key] = max(0.0, value)
    return adjusted


def build_task_from_counterproposal(
    base_task: Dict[str, object],
    counterproposal: Dict[str, object],
    *,
    effective_task: Optional[Dict[str, object]] = None,
    mode: str = "counterproposal",
) -> Dict[str, object]:
    task_mode = str(mode or "counterproposal").strip().lower()
    base_nc = list(base_task.get("nc", [])) if isinstance(base_task.get("nc"), list) else []
    counter_nc = counterproposal.get("nc")
    if isinstance(counter_nc, list) and len(counter_nc) == 4 and all(isinstance(v, (int, float)) for v in counter_nc):
        selected_nc = [int(v) for v in counter_nc] if task_mode != "hybrid" else base_nc
    else:
        selected_nc = base_nc
    base_seed = base_task.get("seed")
    seed_name = str(base_task.get("seed_name", counterproposal.get("seed_name", "")))
    if effective_task and isinstance(effective_task.get("flow"), dict):
        base_flow = {k: float(v) for k, v in effective_task.get("flow", {}).items() if isinstance(v, (int, float))}
    else:
        base_flow = {}
    flow_adjustments = counterproposal.get("flow_adjustments")
    flow_override = apply_flow_adjustments(base_flow, flow_adjustments if isinstance(flow_adjustments, dict) else None)
    return {
        "nc": selected_nc,
        "seed_name": seed_name,
        "seed": base_seed,
        "flow_override": flow_override,
        "counterproposal_run": counterproposal,
        "task_mode": task_mode,
        "source_task": dict(base_task),
    }
