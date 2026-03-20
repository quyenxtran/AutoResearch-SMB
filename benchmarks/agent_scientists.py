from __future__ import annotations

import argparse
import json
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

from . import run_stage as rs
from .agent_results import (
    deterministic_select,
    effective_violation,
    rank_any_results,
    recent_two_run_review_context,
    summarize_result,
)
from .agent_evidence import (
    build_evidence_pack,
    budget_evidence_pack_json,
    compact_prompt_block,
    contains_run_reference,
    coerce_evidence_list,
    coerce_grounded_evidence_refs,
    evidence_refs_are_grounded,
    failure_recovery_context,
    hypothesis_matcher,
    normalize_evidence_refs,
    normalize_text_list,
    review_references_candidate_nc,
    text_mentions_delta_flow_signals,
    text_mentions_delta_metric_signals,
    text_mentions_flow_signals,
    text_mentions_metric_signals,
    text_mentions_numeric_values,
    text_mentions_physics_signals,
    text_mentions_prior_runs,
    text_mentions_required_labels,
    text_mentions_run_name_signals,
    text_mentions_topology_signals,
)
from .agent_llm_client import OpenAICompatClient, request_json_with_single_repair
from .agent_policy import (
    deterministic_review,
    nc_prior_score,
    sqlite_total_records_from_excerpt,
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
        objectives_compact = compact_prompt_block(objectives_excerpt, max_chars=3000, max_lines=80)
        soul_compact = compact_prompt_block(soul_excerpt, max_chars=2500, max_lines=70)
        problem_compact = compact_prompt_block(problem_definition_excerpt, max_chars=2000, max_lines=60)
        skills_compact = compact_prompt_block(skills_excerpt, max_chars=2000, max_lines=60)
        codebase_compact = compact_prompt_block(codebase_excerpt, max_chars=1500, max_lines=50)
        compute_compact = compact_prompt_block(compute_context_excerpt, max_chars=1000, max_lines=35)
        constraint_compact = compact_prompt_block(constraint_context_excerpt, max_chars=1500, max_lines=50)
        sqlite_compact = compact_prompt_block(sqlite_excerpt, max_chars=3000, max_lines=90)
        nc_strategy_compact = compact_prompt_block(nc_strategy_excerpt, max_chars=2000, max_lines=60)
        prompt = textwrap.dedent(
            f"""
            Build a minimal, evidence-first initial plan for a two-scientist SMB campaign.

            Objective context:
            {objectives_compact}

            Operating rules:
            {soul_compact}

            Problem framing:
            {problem_compact}

            SMB physics:
            {skills_compact}

            Codebase context:
            {codebase_compact}

            Runtime compute context:
            {compute_compact}

            Simulation objective/constraint context:
            {constraint_compact}

            SQLite history:
            {sqlite_compact}

            NC strategy board:
            {nc_strategy_compact}

            Requirements:
            - screen all NC layouts before deep seed sweeps
            - be explicit on budget and constraints
            - keep list items short

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
        prompt = compact_prompt_block(prompt, max_chars=15000, max_lines=500)
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


def scientist_a_pick(
    client: OpenAICompatClient,
    candidate_tasks: List[Dict[str, object]],
    results: List[Dict[str, object]],
    tried: set,
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
    shortlist = remaining[: min(len(remaining), 4)]
    default_index = deterministic_select(candidate_tasks, tried)
    if not shortlist:
        return default_index, {"mode": "deterministic", "reason": "No remaining tasks."}

    evidence_pack = build_evidence_pack(results, recent_limit=5, feasible_limit=3, infeasible_limit=4)
    evidence_run_names = [str(item) for item in evidence_pack.get("run_name_catalog", [])]
    evidence_compact = budget_evidence_pack_json(evidence_pack, max_chars=5000)

    best = rank_any_results(results)[0] if results else None
    recent_two_block, recent_two_labels = recent_two_run_review_context(results)
    objectives_compact = compact_prompt_block(objectives_excerpt, max_chars=3000, max_lines=80)
    soul_compact = compact_prompt_block(soul_excerpt, max_chars=2500, max_lines=70)
    codebase_compact = compact_prompt_block(codebase_context_excerpt, max_chars=1200, max_lines=40)
    compute_compact = compact_prompt_block(compute_context_excerpt, max_chars=600, max_lines=20)
    constraint_compact = compact_prompt_block(constraint_context_excerpt, max_chars=1200, max_lines=40)
    heuristics_compact = compact_prompt_block(heuristics_context, max_chars=1500, max_lines=50)
    hypothesis_compact = hypothesis_matcher(heuristics_context, results)
    failure_compact = failure_recovery_context(heuristics_context, results)
    convergence_compact = compact_prompt_block(convergence_context, max_chars=1200, max_lines=40)
    research_compact = compact_prompt_block(research_excerpt, max_chars=2000, max_lines=60)
    nc_strategy_compact = compact_prompt_block(nc_strategy_excerpt, max_chars=2000, max_lines=60)
    sqlite_compact = compact_prompt_block(sqlite_context_excerpt, max_chars=3000, max_lines=90)
    recent_two_compact = compact_prompt_block(recent_two_block, max_chars=1500, max_lines=50)
    priorities_compact = "\n".join(f"- {p}" for p in current_priorities[:6]) or "- none"
    shortlist_brief = [
        {"index": i, "nc": list(item["nc"]), "seed_name": str(item["seed_name"])}
        for i, item in enumerate(shortlist)
    ]
    prompt_warning = ""
    try:
        prompt = textwrap.dedent(
            f"""
            You are Scientist_A for SMB optimization.
            Select one candidate from the shortlist. Return ONLY the JSON below. Do not repeat context.

            Candidate shortlist (select by 0-based index):
            {json.dumps(shortlist_brief, separators=(",", ":"))}

            Respond with this JSON (fill every field with real evidence — no placeholders):
            {{
              "candidate_index": <integer 0..{len(shortlist)-1}>,
              "reason": "<1-2 sentences citing run_name and numeric metric>",
              "acquisition_type": "EXPLORE | EXPLOIT | VERIFY",
              "information_target": "<what does this run answer that prior runs do not?>",
              "coverage_gap": "<untested NC / flow region / hypothesis this fills>",
              "hypothesis_connection": "<hypothesis or failure mode ID this tests>",
              "convergence_assessment": "<improving / stagnating / shift strategy?>",
              "evidence_refs": ["<run_name from evidence pack>"],
              "evidence": ["<metric from a named prior run>", "..."],
              "comparison_to_previous": ["<named run: metric delta + interpretation>", "..."],
              "last_two_run_comparison": ["<R-1: run_name feasible=? prod=? purity=? viol=?>", "<R-2: ...>"],
              "flowrate_comparison": ["<ΔFfeed=..., ΔF1=..., ΔFdes=..., ΔFex=..., ΔFraf=..., Δtstep=...>"],
              "delta_summary": ["<vs R-1: Δprod=..., Δpurity=..., ΔrGA=..., ΔrMA=..., Δviol=...>", "<vs R-2: ...>"],
              "column_topology_comparison": ["<vs R-1: nc=[...]->[...], ΔZ1=..ΔZ4=..>", "<vs R-2: ...>"],
              "physics_rationale": "<zone I-IV mechanics or mass balance argument>",
              "alternatives_considered": ["<rejected index/nc and why>", "..."],
              "priority_updates": ["..."]
            }}

            === CONTEXT (evidence-grounded answers must reference entries below) ===

            Hard project targets: purity_ex_meoh_free>=0.60, recovery_GA>=0.75, recovery_MA>=0.75.
            Mass balance invariant: F1=Ffeed+Fraf=Fdes+Fex (±1%). Feasibility first, then optimize.
            Budget: {args.benchmark_hours:.1f}h total, {args.search_hours:.1f}h search, {budget_used:.4f}h used.
            Min reference probes required before non-reference seeds: {int(getattr(args, "min_probe_reference_runs", 0))}.

            Current best result:
            {summarize_result(best) if best else "None yet."}

            Recent two completed runs (R-1 most recent, R-2 second):
            {recent_two_compact}

            Evidence pack — cite run_name values from here in evidence_refs:
            {evidence_compact}

            NC strategy board:
            {nc_strategy_compact}

            Current priorities:
            {priorities_compact}

            Operating principles:
            {soul_compact}

            Project objectives and constraints:
            {objectives_compact}

            Heuristics and failure modes:
            {heuristics_compact}

            SQLite history:
            {sqlite_compact}

            Research log tail:
            {research_compact}
            """
        ).strip()
        prompt = compact_prompt_block(prompt, max_chars=8000, max_lines=260)
    except Exception as exc:
        prompt_warning = f"Prompt build warning: {type(exc).__name__}: {exc}"
        prompt = (
            "You are Scientist_A for SMB optimization. Return JSON only.\n\n"
            f"Current best result: {summarize_result(best) if best else 'None yet.'}\n"
            f"Recent two completed runs:\n{recent_two_compact}\n\n"
            f"Remaining candidate shortlist:\n{json.dumps(shortlist_brief, separators=(',', ':'))}\n\n"
            "Respond with keys: candidate_index, reason, evidence, comparison_to_previous, "
            "last_two_run_comparison, flowrate_comparison, delta_summary, column_topology_comparison, physics_rationale, nc_competitor_comparison, diagnostic_hypothesis, failure_criteria, fidelity, priority_updates, proposed_followups."
        )
    data, raw, _repaired, repair_error = request_json_with_single_repair(
        client,
        system_prompt="You are an aggressive optimization scientist. Return JSON only and ground claims in evidence.",
        user_prompt=prompt,
        conversation_role="scientist_a_pick",
        metadata={
            "iteration": iteration,
            "search_hours_used": budget_used,
            "shortlist_size": len(shortlist),
            "remaining_count": len(remaining),
            "tried_count": len(tried),
        },
        temperature=0.2,
        required_keys=(
            "candidate_index",
            "reason",
            "acquisition_type",
            "evidence",
            "comparison_to_previous",
            "physics_rationale",
            "evidence_refs",
        ),
    )
    if not isinstance(data, dict):
        return default_index, {
            "mode": "low_quality_recovery",
            "reason": f"Scientist_A JSON failed after repair ({repair_error or 'invalid_output'}). Forcing diagnostic recovery.",
            "low_quality_recovery": True,
            "acquisition_type": "LOW_QUALITY_RECOVERY",
            "priority_updates": [
                "Scientist_A output failed strict JSON/evidence contract; schedule force diagnostic next iteration."
            ],
            "prompt_warning": prompt_warning,
            "raw": raw,
        }
    if data and isinstance(data.get("candidate_index"), int):
        idx = int(data["candidate_index"])
        if 0 <= idx < len(shortlist):
            reason_text = str(data.get("reason", "")).strip()
            evidence = coerce_evidence_list(data.get("evidence"), evidence_pack, min_items=2, max_items=8)
            evidence_refs = coerce_grounded_evidence_refs(data.get("evidence_refs"), evidence_run_names, min_items=1, max_items=8)
            comparisons = normalize_text_list(data.get("comparison_to_previous"), max_items=8)
            last_two_comparisons = normalize_text_list(data.get("last_two_run_comparison"), max_items=4)
            flow_comparisons = normalize_text_list(data.get("flowrate_comparison"), max_items=6)
            delta_summary = normalize_text_list(data.get("delta_summary"), max_items=8)
            topology_comparisons = normalize_text_list(data.get("column_topology_comparison"), max_items=8)
            physics_rationale = str(data.get("physics_rationale", "")).strip()
            nc_comparisons = normalize_text_list(data.get("nc_competitor_comparison"), max_items=8)
            has_history = (len(results) > 0) or (sqlite_total_records_from_excerpt(sqlite_context_excerpt) > 0)
            if evidence_run_names:
                if len(evidence_refs) < 1 or not evidence_refs_are_grounded(evidence_refs, evidence_run_names):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: evidence_refs must contain run_name references from the evidence pack.",
                        "priority_updates": [
                            "Require evidence_refs grounded in run_name_catalog (recent/top feasible/top infeasible runs)."
                        ],
                    }
                if not contains_run_reference(
                    [reason_text] + comparisons + [physics_rationale] + last_two_comparisons,
                    evidence_run_names,
                ):
                    return default_index, {
                        "mode": "deterministic",
                        "reason": "Rejected LLM proposal: reason/comparison/physics text lacks run_name grounding.",
                        "priority_updates": [
                            "Require explicit run_name references in reason/comparison_to_previous/physics_rationale."
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
            data["evidence_refs"] = evidence_refs
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
    soul_excerpt: str = "",
    heuristics_context: str = "",
) -> Dict[str, object]:
    default = deterministic_review(task, best_result)
    prompt_warning = ""
    recent_two_block, recent_two_labels = recent_two_run_review_context(results)
    evidence_pack = build_evidence_pack(results, recent_limit=5, feasible_limit=3, infeasible_limit=4)
    evidence_run_names = [str(item) for item in evidence_pack.get("run_name_catalog", [])]
    evidence_compact = budget_evidence_pack_json(evidence_pack, max_chars=5000)
    codebase_compact = compact_prompt_block(codebase_context_excerpt, max_chars=1200, max_lines=40)
    compute_compact = compact_prompt_block(compute_context_excerpt, max_chars=600, max_lines=20)
    constraint_compact = compact_prompt_block(constraint_context_excerpt, max_chars=1200, max_lines=40)
    nc_strategy_compact = compact_prompt_block(nc_strategy_excerpt, max_chars=2000, max_lines=60)
    research_compact = compact_prompt_block(research_excerpt, max_chars=2000, max_lines=60)
    sqlite_compact = compact_prompt_block(sqlite_context_excerpt, max_chars=3000, max_lines=90)
    recent_two_compact = compact_prompt_block(recent_two_block, max_chars=1500, max_lines=50)
    heuristics_compact = compact_prompt_block(heuristics_context, max_chars=1500, max_lines=50)
    failure_compact = failure_recovery_context(heuristics_context, results)
    soul_compact = compact_prompt_block(soul_excerpt, max_chars=2500, max_lines=70)
    priorities_compact = "\n".join(f"- {p}" for p in current_priorities[:6]) or "- none"
    proposed_task_brief = {"nc": list(task.get("nc", [])), "seed_name": str(task.get("seed_name", ""))}
    effective_task_brief = {
        "nc": list(effective_task.get("nc", [])) if isinstance(effective_task.get("nc"), list) else list(task.get("nc", [])),
        "seed_name": str(effective_task.get("seed_name", task.get("seed_name", ""))),
        "flow": effective_task.get("flow", {}),
    }
    try:
        prompt = textwrap.dedent(
            f"""
            You are Scientist_B. Review the proposed SMB candidate. Return ONLY the JSON below.
            Reject if evidence is generic, mass balance is violated, or the region is proven infeasible.
            If you approve, still provide the strongest counterarguments and required checks.

            Proposed candidate:
            {json.dumps(proposed_task_brief, separators=(",", ":"))}

            Effective bounded candidate (actual flows to be run):
            {json.dumps(effective_task_brief, separators=(",", ":"))}

            Respond with this JSON (fill every field — no placeholders):
            {{
              "decision": "approve" or "reject",
              "reason": "<1-2 sentences citing run_name and numeric evidence>",
              "evidence_refs": ["<run_name from evidence pack>"],
              "comparison_assessment": ["<named run vs proposal: metric delta + interpretation>", "..."],
              "last_two_run_audit": ["<R-1: run_name feasible=? prod=? purity=? viol=?>", "<R-2: ...>"],
              "flowrate_audit": ["<ΔFfeed=..., ΔF1=..., ΔFdes=..., ΔFex=..., ΔFraf=..., Δtstep=...>"],
              "delta_audit": ["<vs R-1: Δprod=..., Δpurity=..., ΔrGA=..., ΔrMA=..., Δviol=...>", "<vs R-2: ...>", "<vs counterproposal: ...>"],
              "column_topology_audit": ["<vs R-1: nc=[...]->[...], ΔZ1..ΔZ4>", "<vs R-2: ...>", "<vs counterproposal topology>"],
              "physics_audit": "<zone I-IV mass balance / selectivity critique>",
              "counterproposal_run": {{
                "nc": [a,b,c,d],
                "flow_adjustments": {{"Ffeed": 0.0, "F1": 0.0, "Fdes": 0.0, "Fex": 0.0, "Fraf": 0.0, "tstep": 0.0}},
                "expected_metric_effect": {{"delta_productivity": 0.0, "delta_purity": 0.0, "delta_recovery_ga": 0.0, "delta_recovery_ma": 0.0, "delta_violation": 0.0}},
                "physics_justification": "<why this counterproposal is better>"
              }},
              "nc_strategy_assessment": ["<candidate nc vs alternatives with evidence>", "..."],
              "compute_assessment": "<budget/time assessment>",
              "counterarguments": ["<strongest objection>", "..."],
              "required_checks": ["<check before trusting result>", "..."],
              "priority_updates": ["..."],
              "risk_flags": ["..."]
            }}

            === CONTEXT ===

            Hard project targets: purity_ex_meoh_free>=0.60, recovery_GA>=0.75, recovery_MA>=0.75.
            Mass balance: F1=Ffeed+Fraf=Fdes+Fex (±1%). Any violation is a Hard Block.

            Current best result:
            {summarize_result(best_result) if best_result else "None yet."}

            Recent two completed runs (R-1 most recent, R-2 second):
            {recent_two_compact}

            Evidence pack — cite run_name values from here in evidence_refs:
            {evidence_compact}

            NC strategy board:
            {nc_strategy_compact}

            Current priorities:
            {priorities_compact}

            Reviewer operating principles:
            {soul_compact}

            Known failure modes:
            {failure_compact}

            Heuristics and hypotheses:
            {heuristics_compact}

            SQLite history:
            {sqlite_compact}
            """
        ).strip()
        prompt = compact_prompt_block(prompt, max_chars=8000, max_lines=260)
    except Exception as exc:
        prompt_warning = f"Prompt build warning: {type(exc).__name__}: {exc}"
        prompt = (
            "You are Scientist_B reviewer. Return JSON only with keys decision, reason, comparison_assessment, "
            "last_two_run_audit, flowrate_audit, delta_audit, column_topology_audit, physics_audit, counterproposal_run, nc_strategy_assessment, compute_assessment, counterarguments, required_checks, priority_updates, risk_flags.\n\n"
            f"Proposed task:\n{json.dumps(proposed_task_brief, separators=(',', ':'))}\n\n"
            f"Effective candidate:\n{json.dumps(effective_task_brief, separators=(',', ':'))}\n\n"
            f"Current best result: {summarize_result(best_result) if best_result else 'None yet.'}\n\n"
            f"Recent two completed runs:\n{recent_two_compact}"
        )
    data, raw, _repaired, repair_error = request_json_with_single_repair(
        client,
        system_prompt="You are a hard-nosed numerical reviewer. Return JSON only and challenge weak proposals.",
        user_prompt=prompt,
        conversation_role="scientist_b_review",
        metadata={
            "iteration": iteration,
            "candidate_nc": task.get("nc"),
            "candidate_seed_name": task.get("seed_name"),
            "effective_flow": effective_task.get("flow", {}),
            "has_best_result": best_result is not None,
        },
        temperature=0.1,
        required_keys=(
            "decision",
            "reason",
            "comparison_assessment",
            "physics_audit",
            "counterproposal_run",
            "evidence_refs",
        ),
    )
    if not isinstance(data, dict):
        return {
            "mode": "low_quality_recovery",
            "decision": "reject",
            "reason": f"Scientist_B JSON failed after repair ({repair_error or 'invalid_output'}). Forcing diagnostic recovery.",
            "low_quality_recovery": True,
            "acquisition_type": "LOW_QUALITY_RECOVERY",
            "priority_updates": [
                "Scientist_B output failed strict JSON/evidence contract; force diagnostic path."
            ],
            "risk_flags": [
                "Scientist_B output invalid after one repair retry."
            ],
            "prompt_warning": prompt_warning,
            "raw": raw,
        }
    if data and str(data.get("decision", "")).lower() in {"approve", "reject"}:
        reason_text = str(data.get("reason", "")).strip()
        evidence_refs = coerce_grounded_evidence_refs(data.get("evidence_refs"), evidence_run_names, min_items=1, max_items=8)
        comparisons = normalize_text_list(data.get("comparison_assessment"), max_items=8)
        last_two_audit = normalize_text_list(data.get("last_two_run_audit"), max_items=4)
        flow_audit = normalize_text_list(data.get("flowrate_audit"), max_items=6)
        delta_audit = normalize_text_list(data.get("delta_audit"), max_items=8)
        topology_audit = normalize_text_list(data.get("column_topology_audit"), max_items=8)
        physics_audit = str(data.get("physics_audit", "")).strip()
        counterproposal = data.get("counterproposal_run")
        nc_assessment = normalize_text_list(data.get("nc_strategy_assessment"), max_items=8)
        has_history = best_result is not None or (sqlite_total_records_from_excerpt(sqlite_context_excerpt) > 0)
        quality_failure = False
        if evidence_run_names:
            if len(evidence_refs) < 1 or not evidence_refs_are_grounded(evidence_refs, evidence_run_names):
                data["decision"] = "reject"
                data["reason"] = "Rejected: evidence_refs must reference run_name entries from the evidence pack."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: evidence_refs are missing or not grounded in evidence rows."
                ]
                quality_failure = True
            if not contains_run_reference([reason_text] + comparisons + [physics_audit], evidence_run_names):
                data["decision"] = "reject"
                data["reason"] = "Rejected: review reason/comparison/physics text lacks run_name grounding."
                data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                    "Review quality risk: missing run_name references in reason/comparison/physics audit."
                ]
                quality_failure = True
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
            quality_failure = True
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
            quality_failure = True
        if has_history and not text_mentions_prior_runs(comparisons):
            data["decision"] = "reject"
            data["reason"] = "Rejected: review comparison lacks concrete prior-run references."
            data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                "Review quality risk: missing run-level evidence in comparison assessment."
            ]
            data["comparison_assessment"] = comparisons or [
                "No run-level prior evidence referenced."
            ]
            quality_failure = True
        if has_history and not text_mentions_metric_signals(comparisons):
            data["decision"] = "reject"
            data["reason"] = "Rejected: review comparison is not metric-grounded."
            data["risk_flags"] = normalize_text_list(data.get("risk_flags"), max_items=6) + [
                "Review quality risk: comparison lacks quantitative metrics."
            ]
            data["comparison_assessment"] = comparisons or [
                "No quantitative metric evidence referenced."
            ]
            quality_failure = True
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
        data["evidence_refs"] = evidence_refs
        data["last_two_run_audit"] = normalize_text_list(data.get("last_two_run_audit"), max_items=4)
        data["flowrate_audit"] = normalize_text_list(data.get("flowrate_audit"), max_items=6)
        data["delta_audit"] = normalize_text_list(data.get("delta_audit"), max_items=8)
        data["column_topology_audit"] = normalize_text_list(data.get("column_topology_audit"), max_items=8)
        data["physics_audit"] = str(data.get("physics_audit", "")).strip()
        if isinstance(data.get("counterproposal_run"), dict):
            data["counterproposal_run"] = data.get("counterproposal_run")
        data["nc_strategy_assessment"] = normalize_text_list(data.get("nc_strategy_assessment"), max_items=8)
        data["compute_assessment"] = str(data.get("compute_assessment", "")).strip()
        reason_final = str(data.get("reason", "")).strip().lower()
        if not quality_failure and str(data.get("decision", "")).lower() == "reject":
            if reason_final.startswith("rejected:"):
                quality_failure = True
        data["low_quality_recovery"] = bool(quality_failure)
        if quality_failure:
            data["acquisition_type"] = "LOW_QUALITY_RECOVERY"
        return {
            "mode": "llm",
            "llm_backend": client.last_backend,
            "prompt_warning": prompt_warning,
            "raw": raw,
            **data,
        }
    return {
        "mode": "low_quality_recovery",
        "decision": "reject",
        "reason": "Scientist_B produced an invalid decision payload after repair; forcing diagnostic recovery.",
        "low_quality_recovery": True,
        "acquisition_type": "LOW_QUALITY_RECOVERY",
        "priority_updates": [
            "Scientist_B output invalid after retry; schedule deterministic diagnostic task."
        ],
        "risk_flags": [
            "Scientist_B review output was not parseable as approve/reject decision."
        ],
        "fallback_review": default,
    }


def scientist_c_arbitrate(
    client: OpenAICompatClient,
    task: Dict[str, object],
    effective_task: Dict[str, object],
    a_note: Dict[str, object],
    b_note: Dict[str, object],
    results: List[Dict[str, object]],
    args: argparse.Namespace,
    heuristics_context: str,
    current_priorities: List[str],
    sqlite_context_excerpt: str,
    iteration: int,
    *,
    revision_count_recent: int = 0,
    force_diagnostic_reason: str = "",
    soul_excerpt: str = "",
    nc_strategy_excerpt: str = "",
) -> Dict[str, object]:
    default_counterproposal = b_note.get("counterproposal_run") if isinstance(b_note.get("counterproposal_run"), dict) else None
    default_decision = "IMPLEMENT_B_COUNTER" if default_counterproposal else "IMPLEMENT_A"
    best_result = rank_any_results(results)[0] if results else None
    recent_two_block, _ = recent_two_run_review_context(results)
    hypothesis_compact = hypothesis_matcher(heuristics_context, results)
    failure_compact = failure_recovery_context(heuristics_context, results)
    recent_two_compact = compact_prompt_block(recent_two_block, max_chars=1500, max_lines=50)
    heuristics_compact = compact_prompt_block(heuristics_context, max_chars=1500, max_lines=50)
    sqlite_compact = compact_prompt_block(sqlite_context_excerpt, max_chars=3000, max_lines=90)
    soul_compact = compact_prompt_block(soul_excerpt, max_chars=2500, max_lines=70)
    nc_strategy_compact = compact_prompt_block(nc_strategy_excerpt, max_chars=2000, max_lines=60)
    evidence_pack = build_evidence_pack(results, recent_limit=5, feasible_limit=3, infeasible_limit=4)
    evidence_run_names = [str(item) for item in evidence_pack.get("run_name_catalog", [])]
    evidence_compact = budget_evidence_pack_json(evidence_pack, max_chars=5000)
    priorities_compact = "\n".join(f"- {p}" for p in current_priorities[:6]) or "- none"
    a_brief = {
        "decision": str(a_note.get("decision", "")).strip(),
        "reason": str(a_note.get("reason", "")).strip(),
        "acquisition_type": str(a_note.get("acquisition_type", "")).strip(),
        "evidence_refs": normalize_evidence_refs(a_note.get("evidence_refs"), max_items=6),
        "candidate_index": a_note.get("candidate_index"),
        "candidate": {"nc": list(task.get("nc", [])), "seed_name": str(task.get("seed_name", ""))},
        "evidence": normalize_text_list(a_note.get("evidence"), max_items=4),
        "coverage_gap": str(a_note.get("coverage_gap", "")).strip(),
        "hypothesis_connection": str(a_note.get("hypothesis_connection", "")).strip(),
        "convergence_assessment": str(a_note.get("convergence_assessment", "")).strip(),
        "diagnostic_hypothesis": str(a_note.get("diagnostic_hypothesis", "")).strip(),
    }
    b_brief = {
        "decision": str(b_note.get("decision", "")).strip(),
        "reason": str(b_note.get("reason", "")).strip(),
        "evidence_refs": normalize_evidence_refs(b_note.get("evidence_refs"), max_items=6),
        "comparison_assessment": normalize_text_list(b_note.get("comparison_assessment"), max_items=4),
        "counterproposal_run": b_note.get("counterproposal_run"),
        "nc_strategy_assessment": normalize_text_list(b_note.get("nc_strategy_assessment"), max_items=4),
        "physics_audit": str(b_note.get("physics_audit", "")).strip(),
        "risk_flags": normalize_text_list(b_note.get("risk_flags"), max_items=4),
    }
    try:
        exec_model = str(getattr(args, "executive_llm_model", "")).strip() or client.model
        exec_client = client
        if exec_model != client.model:
            exec_client = OpenAICompatClient(
                client.base_url,
                exec_model,
                client.enabled,
                api_key=client.api_key,
                fallback_enabled=client.fallback_enabled,
                fallback_base_url=client.fallback_base_url,
                fallback_model=client.fallback_model,
                fallback_api_key=client.fallback_api_key,
                conversation_stream_path=client.conversation_stream_path,
                timeout_seconds=client.timeout_seconds,
                max_tokens=client.max_tokens,
                max_retries=client.max_retries,
                retry_backoff_seconds=client.retry_backoff_seconds,
                conversation_log_mode=client.conversation_log_mode,
                conversation_response_max_chars=client.conversation_response_max_chars,
            )
        prompt = textwrap.dedent(
            f"""
            You are Scientist_C, the executive arbiter for SMB search.
            Rule on the A vs B disagreement. Return ONLY the JSON below.

            Respond with this JSON (no placeholders — cite evidence_refs from the bundle below):
            {{
              "decision": "IMPLEMENT_A | IMPLEMENT_B_COUNTER | IMPLEMENT_HYBRID | RETURN_FOR_REVISION | FORCE_DIAGNOSTIC",
              "reason": "<1-2 sentences citing run_name and physics argument>",
              "selected_task_mode": "A | B_COUNTER | HYBRID | REVISION | DIAGNOSTIC",
              "acquisition_type": "<compact label>",
              "evidence_refs": ["<run_name from evidence bundle>"],
              "diagnostic_focus": "<what to test if FORCE_DIAGNOSTIC>",
              "revision_request": "<what to fix if RETURN_FOR_REVISION>",
              "priority_updates": ["..."]
            }}

            Taxonomy rules:
            - IMPLEMENT_A: A correct, B objection is not a Hard Block
            - IMPLEMENT_B_COUNTER: B's counterproposal is fully specified and physics-grounded
            - IMPLEMENT_HYBRID: merge A's topology with B's flow adjustment (only if both are evidence-grounded)
            - RETURN_FOR_REVISION: Soft Block, one bounded revision warranted (revision_count < max)
            - FORCE_DIAGNOSTIC: circular debate, stuck region, or hard evidence contradiction

            Hard Block (always reject): mass balance violation, proven infeasible NC, fidelity jump without evidence.
            Anti-stall: 3 consecutive RETURN_FOR_REVISION → escalate to FORCE_DIAGNOSTIC.

            === CONTEXT ===

            Candidate: {json.dumps({"nc": list(task.get("nc", [])), "seed_name": str(task.get("seed_name", "")), "flow": effective_task.get("flow", {})}, separators=(",", ":"))}

            Scientist_A summary:
            {json.dumps(a_brief, separators=(",", ":"))}

            Scientist_B summary:
            {json.dumps(b_brief, separators=(",", ":"))}

            Current best result:
            {summarize_result(best_result) if best_result else "None yet."}

            Recent two completed runs:
            {recent_two_compact}

            Evidence bundle — cite run_name values from here:
            {evidence_compact}

            NC strategy board:
            {nc_strategy_compact}

            Arbitration principles:
            {soul_compact}

            Hypotheses and failure context:
            {hypothesis_compact}

            Heuristics:
            {heuristics_compact}

            SQLite context:
            {sqlite_compact}
            """
        ).strip()
        prompt = compact_prompt_block(prompt, max_chars=7000, max_lines=230)
        data, raw, _repaired, _repair_error = request_json_with_single_repair(
            exec_client,
            system_prompt="You are a decisive process executive. Return JSON only.",
            user_prompt=prompt,
            conversation_role="scientist_c_arbitrate",
            metadata={
                "iteration": iteration,
                "candidate_nc": task.get("nc"),
                "candidate_seed_name": task.get("seed_name"),
                "revision_count_recent": revision_count_recent,
                "force_diagnostic_reason": force_diagnostic_reason,
                "executive_model": exec_model,
            },
            temperature=0.1,
            required_keys=(
                "decision",
                "reason",
                "selected_task_mode",
                "acquisition_type",
                "evidence_refs",
            ),
        )
        if exec_client is not client:
            client.conversations.extend(exec_client.conversations)
            if exec_client.last_backend != "none":
                client.last_backend = exec_client.last_backend
        if isinstance(data, dict):
            decision = str(data.get("decision", "")).strip().upper()
            valid = {
                "IMPLEMENT_A",
                "IMPLEMENT_B_COUNTER",
                "IMPLEMENT_HYBRID",
                "RETURN_FOR_REVISION",
                "FORCE_DIAGNOSTIC",
            }
            if decision in valid:
                evidence_refs = coerce_grounded_evidence_refs(data.get("evidence_refs"), evidence_run_names, min_items=1, max_items=8)
                if evidence_run_names and not evidence_refs_are_grounded(evidence_refs, evidence_run_names):
                    data["decision"] = "FORCE_DIAGNOSTIC"
                    data["reason"] = "Executive output missing grounded evidence_refs; forcing diagnostic path."
                    data["selected_task_mode"] = "DIAGNOSTIC"
                    data["acquisition_type"] = "FORCE_DIAGNOSTIC"
                    evidence_refs = evidence_run_names[:2]
                decision = str(data.get("decision", decision)).strip().upper()
                data["decision"] = decision
                data["reason"] = str(data.get("reason", "")).strip()
                data["priority_updates"] = normalize_text_list(data.get("priority_updates"), max_items=8)
                data["diagnostic_focus"] = str(data.get("diagnostic_focus", "")).strip()
                data["revision_request"] = str(data.get("revision_request", "")).strip()
                data["selected_task_mode"] = str(data.get("selected_task_mode", "")).strip().upper()
                data["acquisition_type"] = str(data.get("acquisition_type", "")).strip().upper()
                data["evidence_refs"] = evidence_refs
                data["llm_backend"] = exec_client.last_backend
                data["raw"] = raw
                data["mode"] = "llm"
                if not data["acquisition_type"]:
                    data["acquisition_type"] = decision
                return data
    except Exception as exc:
        fallback = {
            "mode": "deterministic_error",
            "reason": f"Scientist_C exception fallback: {type(exc).__name__}: {exc}",
            "priority_updates": [
                "Executive arbitration failed; use deterministic fallback based on counterproposal quality and diagnostics."
            ],
        }
        if default_counterproposal is not None:
            fallback["decision"] = "IMPLEMENT_B_COUNTER"
            fallback["acquisition_type"] = "IMPLEMENT_B_COUNTER"
        else:
            fallback["decision"] = "IMPLEMENT_A"
            fallback["acquisition_type"] = "IMPLEMENT_A"
        return fallback

    counterproposal_valid = isinstance(default_counterproposal, dict)
    if force_diagnostic_reason:
        return {
            "mode": "deterministic",
            "decision": "FORCE_DIAGNOSTIC",
            "reason": force_diagnostic_reason,
            "priority_updates": [
                "Systematic infeasibility trigger requested an immediate diagnostic execution."
            ],
            "diagnostic_focus": force_diagnostic_reason,
            "selected_task_mode": "DIAGNOSTIC",
            "acquisition_type": "FORCE_DIAGNOSTIC",
        }
    if revision_count_recent < max(0, int(getattr(args, "executive_max_revisions", 1))):
        return {
            "mode": "deterministic",
            "decision": "RETURN_FOR_REVISION",
            "reason": "Deterministic fallback prefers a bounded revision request before forcing an execution choice.",
            "priority_updates": [
                "Return the task for one more revision window before executing a counterproposal."
            ],
            "revision_request": "Tighten the counterproposal and make the diagnostic target more specific.",
            "selected_task_mode": "REVISION",
            "acquisition_type": "RETURN_FOR_REVISION",
        }
    if counterproposal_valid:
        task_nc = list(task.get("nc", []))
        cp_nc = default_counterproposal.get("nc") if isinstance(default_counterproposal, dict) else None
        if isinstance(cp_nc, list) and len(cp_nc) == 4 and cp_nc != task_nc:
            decision = "IMPLEMENT_B_COUNTER"
        elif best_result and effective_violation(best_result) > 1e-3:
            decision = "IMPLEMENT_HYBRID"
        else:
            decision = default_decision
        return {
            "mode": "deterministic",
            "decision": decision,
            "reason": "Deterministic fallback selected the strongest available non-revision action.",
            "priority_updates": [
                "Use the best available executable task when arbitration is unavailable."
            ],
            "selected_task_mode": "B_COUNTER" if decision == "IMPLEMENT_B_COUNTER" else ("HYBRID" if decision == "IMPLEMENT_HYBRID" else "A"),
            "acquisition_type": decision,
        }
    return {
        "mode": "deterministic",
        "decision": "IMPLEMENT_A",
        "reason": "Deterministic fallback keeps the original proposal because no valid counterproposal is available.",
        "priority_updates": [
            "Fallback to Scientist_A proposal when no usable counterproposal exists."
        ],
        "selected_task_mode": "A",
        "acquisition_type": "IMPLEMENT_A",
    }
