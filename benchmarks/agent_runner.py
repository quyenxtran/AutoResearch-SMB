#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib import error, request

from . import run_stage as rs


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-scientist SMB agent benchmark.")
    parser.add_argument("--run-name", default=os.environ.get("SMB_EXPERIMENT_NAME", "qwen_smb_two_scientists"))
    parser.add_argument("--artifact-dir", default=str(REPO_ROOT / "artifacts" / "agent_runs"))
    parser.add_argument("--nc-library", default=os.environ.get("SMB_NC_LIBRARY", "1,2,3,2;2,2,2,2;1,3,2,2"))
    parser.add_argument("--seed-library", default=os.environ.get("SMB_SEED_LIBRARY", "notebook"))
    parser.add_argument("--solver-name", default=os.environ.get("SMB_SOLVER_NAME", "auto"))
    parser.add_argument("--linear-solver", default=os.environ.get("SMB_LINEAR_SOLVER", "mumps"))
    parser.add_argument("--benchmark-hours", type=float, default=float(os.environ.get("SMB_BENCHMARK_HOURS", "5.0")))
    parser.add_argument("--search-hours", type=float, default=float(os.environ.get("SMB_SEARCH_BUDGET_HOURS", "4.0")))
    parser.add_argument("--validation-hours", type=float, default=float(os.environ.get("SMB_VALIDATION_BUDGET_HOURS", "1.0")))
    parser.add_argument("--max-search-evals", type=int, default=int(os.environ.get("SMB_AGENT_MAX_SEARCH_EVALS", "18")))
    parser.add_argument("--max-validations", type=int, default=int(os.environ.get("SMB_AGENT_MAX_VALIDATIONS", "3")))
    parser.add_argument("--tee", action="store_true", default=os.environ.get("SMB_AGENT_TEE", "0") == "1")
    parser.add_argument("--llm-enabled", action="store_true", default=os.environ.get("SMB_AGENT_LLM_ENABLED", "1") == "1")
    parser.add_argument("--llm-base-url", default=os.environ.get("OLLAMA_BASE_URL", ""))
    parser.add_argument("--llm-model", default=os.environ.get("OLLAMA_MODEL", os.environ.get("SMB_LOCAL_LLM_MODEL", "qwen3.5:9b")))
    parser.add_argument("--objectives-file", default=os.environ.get("SMB_OBJECTIVES_FILE", str(REPO_ROOT / "Objectives.md")))
    parser.add_argument("--llm-soul-file", default=os.environ.get("SMB_LLM_SOUL_FILE", str(REPO_ROOT / "LLM_SOUL.md")))
    parser.add_argument("--ipopt-resource-file", default=os.environ.get("SMB_IPOPT_RESOURCE_FILE", str(REPO_ROOT / "IPOPT_SOLVER_RESOURCES.md")))
    return parser


def make_stage_args(stage: str) -> argparse.Namespace:
    return rs.build_parser().parse_args(["--stage", stage])


def env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in {None, ""} else default


def configure_stage_args(base: argparse.Namespace, args: argparse.Namespace) -> argparse.Namespace:
    stage_args = argparse.Namespace(**vars(base))
    stage_args.solver_name = args.solver_name
    stage_args.linear_solver = args.linear_solver
    stage_args.tee = args.tee
    stage_args.nc_library = args.nc_library
    stage_args.seed_library = args.seed_library
    stage_args.max_iter = 5000
    stage_args.tol = 1e-6
    stage_args.acceptable_tol = 1e-5
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
    stage_args.meoh_max_raff_wt = float(env_or_default("SMB_MEOH_MAX_RAFF_WT", str(stage_args.meoh_max_raff_wt)))
    stage_args.water_max_ex_wt = float(env_or_default("SMB_WATER_MAX_EX_WT", str(stage_args.water_max_ex_wt)))
    stage_args.water_max_zone1_entry_wt = float(
        env_or_default("SMB_WATER_MAX_ZONE1_ENTRY_WT", str(stage_args.water_max_zone1_entry_wt))
    )
    return stage_args


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, enabled: bool) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled and bool(self.base_url) and bool(self.model)

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.enabled:
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (error.URLError, error.HTTPError, KeyError, json.JSONDecodeError, TimeoutError):
            return None

    @staticmethod
    def extract_json(text: Optional[str]) -> Optional[Dict[str, object]]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def read_doc_excerpt(path: str, max_chars: int = 4000) -> str:
    p = Path(path)
    if not p.exists():
        return f"Missing file: {path}"
    return p.read_text(encoding="utf-8")[:max_chars]


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


def effective_violation(result: Dict[str, object]) -> float:
    slacks = result.get("constraint_slacks")
    if isinstance(slacks, dict) and "normalized_total_violation" in slacks:
        return float(slacks["normalized_total_violation"])
    provisional = result.get("provisional")
    if isinstance(provisional, dict):
        metrics = provisional.get("metrics") or {}
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


def deterministic_select(tasks: List[Dict[str, object]], tried: set[Tuple[Tuple[int, ...], str]]) -> int:
    for idx, task in enumerate(tasks):
        key = (tuple(task["nc"]), str(task["seed_name"]))
        if key not in tried:
            return idx
    return 0


def deterministic_review(candidate: Dict[str, object], best_result: Optional[Dict[str, object]]) -> Dict[str, object]:
    if best_result and candidate["nc"] == best_result.get("nc") and candidate["seed_name"] == best_result.get("seed_name"):
        return {"decision": "reject", "reason": "Already evaluated this layout and seed."}
    return {"decision": "approve", "reason": "Candidate is within current bounds and still untested."}


def rank_any_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked = sorted(results, key=search_score, reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank_any"] = idx
    return ranked


def build_search_tasks(args: argparse.Namespace) -> List[Dict[str, object]]:
    nc_library = rs.parse_nc_library(args.nc_library)
    seed_library = rs.parse_seed_library(args.seed_library)
    tasks: List[Dict[str, object]] = []
    for nc in nc_library:
        for seed in seed_library:
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
    budget_used: float,
) -> Tuple[int, Dict[str, object]]:
    remaining = [task for task in candidate_tasks if (tuple(task["nc"]), str(task["seed_name"])) not in tried]
    shortlist = remaining[: min(len(remaining), 8)]
    default_index = deterministic_select(candidate_tasks, tried)
    if not shortlist:
        return default_index, {"mode": "deterministic", "reason": "No remaining tasks."}

    best = rank_any_results(results)[0] if results else None
    prompt = textwrap.dedent(
        f"""
        You are Scientist_A for an SMB optimization benchmark.
        Objective summary:
        {objectives_excerpt}

        Scientist rules summary:
        {soul_excerpt}

        Counted benchmark budget is 5.0 SMB hours with 4.0 search hours and 1.0 validation hour.
        Search wall-hours used so far: {budget_used:.4f}

        Current best result:
        {summarize_result(best) if best else "None yet."}

        Remaining candidate shortlist:
        {json.dumps(shortlist, indent=2)}

        Respond with JSON only:
        {{"candidate_index": <0-based index into shortlist>, "reason": "<brief reason>", "fidelity": "medium"}}
        """
    ).strip()
    raw = client.chat("You are a concise optimization scientist. Return JSON only.", prompt)
    data = client.extract_json(raw)
    if data and isinstance(data.get("candidate_index"), int):
        idx = int(data["candidate_index"])
        if 0 <= idx < len(shortlist):
            chosen = shortlist[idx]
            absolute_idx = candidate_tasks.index(chosen)
            return absolute_idx, {"mode": "llm", "raw": raw, **data}
    return default_index, {"mode": "deterministic", "reason": "Falling back to deterministic candidate choice."}


def scientist_b_review(
    client: OpenAICompatClient,
    task: Dict[str, object],
    best_result: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    default = deterministic_review(task, best_result)
    prompt = textwrap.dedent(
        f"""
        You are Scientist_B. Review this proposed SMB medium-fidelity optimization attempt.
        Proposed task:
        {json.dumps(task, indent=2)}

        Current best result:
        {summarize_result(best_result) if best_result else "None yet."}

        Hard bounds include:
        - F1 in {args.f1_bounds}
        - Ffeed, Fdes, Fex, Fraf in {args.ffeed_bounds}, {args.fdes_bounds}, {args.fex_bounds}, {args.fraf_bounds}
        - tstep in {args.tstep_bounds}

        Respond with JSON only:
        {{"decision": "approve" or "reject", "reason": "<brief reason>"}}
        """
    ).strip()
    raw = client.chat("You are a skeptical numerical scientist. Return JSON only.", prompt)
    data = client.extract_json(raw)
    if data and str(data.get("decision", "")).lower() in {"approve", "reject"}:
        return {"mode": "llm", "raw": raw, **data}
    return {"mode": "deterministic", **default}


def execute_search_task(args: argparse.Namespace, task: Dict[str, object]) -> Dict[str, object]:
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
    candidate_args.run_name = f"{args.run_name}_search_nc_{'-'.join(str(v) for v in task['nc'])}_{candidate_args.seed_name}"
    return rs.evaluate_optimized_layout(candidate_args, tuple(task["nc"]))


def build_validation_candidates(results: List[Dict[str, object]], max_items: int) -> List[Dict[str, object]]:
    ranked = rank_any_results(results)
    selected: List[Dict[str, object]] = []
    seen: set[Tuple[Tuple[int, ...], float, float, float, float, float]] = set()
    for item in ranked:
        flow = effective_flow(item)
        if flow is None:
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
    return selected


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


def write_artifact(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    artifact = artifact_path(args)
    objectives_excerpt = read_doc_excerpt(args.objectives_file, max_chars=5000)
    soul_excerpt = read_doc_excerpt(args.llm_soul_file, max_chars=4000)
    ipopt_excerpt = read_doc_excerpt(args.ipopt_resource_file, max_chars=2500)
    client = OpenAICompatClient(args.llm_base_url, args.llm_model, args.llm_enabled)

    search_results: List[Dict[str, object]] = []
    validation_results: List[Dict[str, object]] = []
    scientist_a_log: List[Dict[str, object]] = []
    scientist_b_log: List[Dict[str, object]] = []
    ledger: List[Dict[str, object]] = []
    tried: set[Tuple[Tuple[int, ...], str]] = set()

    try:
        solver_check = rs.run_solver_check(configure_stage_args(make_stage_args("solver-check"), args))
        search_tasks = build_search_tasks(args)
        search_hours_used = 0.0
        validation_hours_used = 0.0

        while (
            len(tried) < len(search_tasks)
            and len(search_results) < args.max_search_evals
            and search_hours_used < args.search_hours
        ):
            idx, a_note = scientist_a_pick(
                client,
                search_tasks,
                search_results,
                tried,
                configure_stage_args(make_stage_args("optimize-layouts"), args),
                objectives_excerpt,
                soul_excerpt,
                search_hours_used,
            )
            task = search_tasks[idx]
            task_key = (tuple(task["nc"]), str(task["seed_name"]))
            if task_key in tried:
                break
            scientist_a_log.append({"task": task, "decision": a_note})

            best_so_far = rank_any_results(search_results)[0] if search_results else None
            b_note = scientist_b_review(
                client,
                task,
                best_so_far,
                configure_stage_args(make_stage_args("optimize-layouts"), args),
            )
            scientist_b_log.append({"task": task, "decision": b_note})
            tried.add(task_key)
            if str(b_note.get("decision", "approve")).lower() != "approve":
                continue

            result = execute_search_task(args, task)
            search_results.append(result)
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

        validation_pool = build_validation_candidates(search_results, args.max_validations)
        for ordinal, candidate in enumerate(validation_pool, start=1):
            if validation_hours_used >= args.validation_hours:
                break
            validation = execute_validation(args, candidate, ordinal)
            validation_results.append(validation)
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

        payload = {
            "status": "ok",
            "run_name": args.run_name,
            "mode": "llm-assisted" if client.enabled else "deterministic-fallback",
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
                "ipopt_resource_file": args.ipopt_resource_file,
                "objectives_excerpt": objectives_excerpt,
                "llm_soul_excerpt": soul_excerpt,
                "ipopt_resource_excerpt": ipopt_excerpt,
            },
            "scientist_a_log": scientist_a_log,
            "scientist_b_log": scientist_b_log,
            "search_results": search_results,
            "validation_results": validation_results,
            "ranked_search_results": ranked_search,
            "ranked_validation_results": ranked_validation,
            "best_result": final_best,
            "ledger": ledger,
        }
        write_artifact(artifact, payload)
        print(json.dumps({"artifact": str(artifact), "status": "ok", "run_name": args.run_name}, indent=2))
        return 0
    except Exception as exc:
        payload = {
            "status": "error",
            "run_name": args.run_name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_artifact(artifact, payload)
        print(json.dumps({"artifact": str(artifact), "status": "error", "run_name": args.run_name}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
