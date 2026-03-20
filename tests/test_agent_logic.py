from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "Agent-Driven-NLP-Opimizer"
IMPORT_ROOT = PROJECT_ROOT if PROJECT_ROOT.exists() else REPO_ROOT
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

# Keep unit tests independent from optional solver deps only when pyomo is
# genuinely unavailable. Avoid poisoning the full test session when real pyomo
# is installed for integration-style tests.
if "pyomo.environ" not in sys.modules and importlib.util.find_spec("pyomo.environ") is None:
    pyomo_mod = types.ModuleType("pyomo")
    pyomo_env_mod = types.ModuleType("pyomo.environ")
    pyomo_env_mod.value = lambda x: x
    pyomo_mod.environ = pyomo_env_mod
    sys.modules["pyomo"] = pyomo_mod
    sys.modules["pyomo.environ"] = pyomo_env_mod

from benchmarks import agent_runner as ar


class QueueClient:
    def __init__(self, responses: list[str], model: str = "stub-model") -> None:
        self.responses = list(responses)
        self.last_backend = "stub"
        self.conversations: list[dict[str, object]] = []
        self.model = model
        self.base_url = "http://stub.local/v1"
        self.enabled = True
        self.api_key = "stub"
        self.fallback_enabled = False
        self.fallback_base_url = ""
        self.fallback_model = ""
        self.fallback_api_key = ""
        self.conversation_stream_path = None
        self.timeout_seconds = 30.0
        self.max_tokens = 256
        self.max_retries = 1
        self.retry_backoff_seconds = 0.0
        self.conversation_log_mode = "compact"
        self.conversation_response_max_chars = 1200

    def chat(self, *args: object, **kwargs: object) -> str:
        if self.responses:
            return self.responses.pop(0)
        return "{}"

    def extract_json(self, raw: str) -> dict[str, object]:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def make_args(**overrides: object) -> argparse.Namespace:
    base = {
        "run_name": "unit",
        "benchmark_hours": 12.0,
        "search_hours": 10.0,
        "validation_hours": 2.0,
        "min_probe_reference_runs": 0,
        "probe_low_fidelity_enabled": 1,
        "probe_nfex": 5,
        "probe_nfet": 2,
        "probe_ncp": 1,
        "finalization_hard_gate_enabled": 1,
        "executive_controller_enabled": True,
        "executive_trigger_rejects": 2,
        "executive_force_after_rejects": 3,
        "executive_top_k_lock": 1,
        "single_scientist_mode": 0,
        "executive_arbitration_enabled": 1,
        "executive_max_revisions": 1,
        "executive_llm_model": "stub-model",
        "systematic_infeasibility_k": 5,
        "random_search_mode": 0,
        "method": "agent_v2",
        "nc_library": "1,2,3,2;2,2,2,2",
        "seed_library": "notebook",
        "f1_bounds": "0.5,5.0",
        "ffeed_bounds": "0.5,2.5",
        "fdes_bounds": "0.5,2.5",
        "fex_bounds": "0.5,2.5",
        "fraf_bounds": "0.5,2.5",
        "tstep_bounds": "8.0,12.0",
        "max_pump_flow": 2.5,
        "f1_max_flow": 5.0,
        "purity_min": 0.6,
        "recovery_ga_min": 0.75,
        "recovery_ma_min": 0.75,
        "project_purity_min": 0.6,
        "project_recovery_ga_min": 0.75,
        "project_recovery_ma_min": 0.75,
        "meoh_max_raff_wt": 1.0,
        "water_max_ex_wt": 1.0,
        "water_max_zone1_entry_wt": 1.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def sample_result(
    run_name: str,
    *,
    feasible: bool,
    status: str,
    j: float | None,
    productivity: float,
    purity: float,
    rga: float,
    rma: float,
    violation: float,
    nc: list[int] | None = None,
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "nc": list(nc or [2, 2, 2, 2]),
        "seed_name": "reference",
        "status": status,
        "feasible": feasible,
        "J_validated": j,
        "metrics": {
            "productivity_ex_ga_ma": productivity,
            "purity_ex_meoh_free": purity,
            "recovery_ex_GA": rga,
            "recovery_ex_MA": rma,
        },
        "constraint_slacks": {"normalized_total_violation": violation},
        "flow": {"Ffeed": 1.2, "F1": 2.0, "Fdes": 1.1, "Fex": 0.9, "Fraf": 0.8, "tstep": 9.2},
        "timing": {"wall_seconds": 1.0, "cpu_hours_accounted": 0.001},
    }


def candidate_tasks() -> list[dict[str, object]]:
    seed = {"name": "reference", "Ffeed": 1.2, "F1": 2.0, "Fdes": 1.1, "Fex": 0.9, "Fraf": 0.8, "tstep": 9.2}
    return [
        {"nc": [2, 2, 2, 2], "seed_name": "reference", "seed": dict(seed)},
        {"nc": [1, 2, 3, 2], "seed_name": "reference", "seed": dict(seed)},
    ]


def test_build_evidence_pack_includes_required_feasible_and_infeasible_windows() -> None:
    results = [
        sample_result("run_01", feasible=False, status="solver_error", j=None, productivity=0.7, purity=0.5, rga=0.6, rma=0.6, violation=0.5),
        sample_result("run_02", feasible=False, status="infeasible", j=None, productivity=0.8, purity=0.58, rga=0.7, rma=0.69, violation=0.18),
        sample_result("run_03", feasible=True, status="ok", j=0.9, productivity=1.0, purity=0.7, rga=0.8, rma=0.81, violation=0.0),
        sample_result("run_04", feasible=True, status="ok", j=1.1, productivity=1.2, purity=0.78, rga=0.86, rma=0.84, violation=0.0),
        sample_result("run_05", feasible=False, status="solver_error", j=None, productivity=0.65, purity=0.45, rga=0.55, rma=0.52, violation=0.7),
        sample_result("run_06", feasible=True, status="ok", j=1.3, productivity=1.4, purity=0.82, rga=0.9, rma=0.88, violation=0.0),
        sample_result("run_07", feasible=False, status="infeasible", j=None, productivity=0.95, purity=0.62, rga=0.73, rma=0.72, violation=0.05),
    ]
    pack = ar.build_evidence_pack(results, recent_limit=5, feasible_limit=3, infeasible_limit=4)
    assert [row["run_name"] for row in pack["recent_runs"]] == ["run_03", "run_04", "run_05", "run_06", "run_07"]
    assert len(pack["top_feasible"]) == 3
    assert [row["run_name"] for row in pack["top_feasible"]] == ["run_06", "run_04", "run_03"]
    assert 1 <= len(pack["top_infeasible"]) <= 4
    assert any(str(row["status"]).lower() == "solver_error" for row in pack["top_infeasible"])
    assert any(float(row["normalized_total_violation"] or 1.0) <= 0.18 for row in pack["top_infeasible"])
    assert "run_06" in pack["run_name_catalog"]
    assert "run_01" in pack["run_name_catalog"]


def test_coerce_evidence_list_falls_back_to_evidence_pack_rows() -> None:
    pack = ar.build_evidence_pack(
        [
            sample_result("run_01", feasible=False, status="solver_error", j=None, productivity=0.7, purity=0.5, rga=0.6, rma=0.6, violation=0.5),
            sample_result("run_02", feasible=True, status="ok", j=1.2, productivity=1.4, purity=0.8, rga=0.9, rma=0.88, violation=0.0),
        ],
        recent_limit=5,
        feasible_limit=3,
        infeasible_limit=3,
    )
    evidence = ar.coerce_evidence_list([], pack, min_items=2, max_items=6)
    assert len(evidence) >= 2
    assert any("run_name=run_01" in item or "run_name=run_02" in item for item in evidence)


def test_bootstrap_reference_select_prefers_reference_seed() -> None:
    tasks = [
        {"nc": [2, 2, 2, 2], "seed_name": "parameter_search", "seed": {"name": "parameter_search"}},
        {"nc": [1, 2, 3, 2], "seed_name": "reference", "seed": {"name": "reference"}},
    ]
    idx = ar.bootstrap_reference_select(tasks, tried=set())
    assert idx == 1


def test_request_json_with_single_repair_requests_json_mode() -> None:
    class CaptureClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat(self, *args: object, **kwargs: object) -> str:
            self.calls.append(dict(kwargs))
            return '{"candidate_index": 0}'

        def extract_json(self, raw: str) -> dict[str, object]:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}

    client = CaptureClient()
    data, _raw, _repaired, _err = ar.request_json_with_single_repair(
        client,
        system_prompt="sys",
        user_prompt="user",
        conversation_role="scientist_a_pick",
        metadata={"iteration": 1},
        temperature=0.2,
        required_keys=("candidate_index",),
    )
    assert isinstance(data, dict)
    assert client.calls
    assert client.calls[0].get("require_json_output") is True


def test_scientist_a_invalid_output_retries_once_then_forces_diagnostic_mode() -> None:
    bad_payload = {
        "candidate_index": 0,
        "reason": "Looks good.",
        "acquisition_type": "EXPLOIT",
        "evidence": ["good trend"],
        "comparison_to_previous": ["better than before"],
        "physics_rationale": "mass balance",
    }
    client = QueueClient([json.dumps(bad_payload), json.dumps(bad_payload)])
    idx, note = ar.scientist_a_pick(
        client,
        candidate_tasks(),
        [
            sample_result("run_10", feasible=False, status="solver_error", j=None, productivity=0.8, purity=0.58, rga=0.7, rma=0.68, violation=0.2),
            sample_result("run_11", feasible=True, status="ok", j=1.15, productivity=1.22, purity=0.8, rga=0.87, rma=0.84, violation=0.0),
        ],
        tried=set(),
        args=make_args(),
        objectives_excerpt="",
        soul_excerpt="",
        codebase_context_excerpt="",
        compute_context_excerpt="",
        constraint_context_excerpt="",
        nc_strategy_excerpt="",
        research_excerpt="",
        current_priorities=[],
        sqlite_context_excerpt="SQLite context: total_records=2, feasible_records=1",
        budget_used=0.0,
        iteration=1,
        heuristics_context="",
        convergence_context="",
    )
    assert idx == 0
    assert note["mode"] == "low_quality_recovery"
    assert note["low_quality_recovery"] is True
    assert note["acquisition_type"] == "LOW_QUALITY_RECOVERY"


def test_scientist_b_invalid_output_retries_once_then_forces_diagnostic_mode() -> None:
    client = QueueClient(['{"decision":"reject"}', '{"decision":"reject"}'])
    note = ar.scientist_b_review(
        client,
        task={"nc": [2, 2, 2, 2], "seed_name": "reference"},
        effective_task={
            "nc": [2, 2, 2, 2],
            "seed_name": "reference",
            "flow": {"Ffeed": 1.2, "F1": 2.0, "Fdes": 1.1, "Fex": 0.9, "Fraf": 0.8, "tstep": 9.2},
        },
        best_result=None,
        results=[sample_result("run_21", feasible=False, status="solver_error", j=None, productivity=0.6, purity=0.5, rga=0.6, rma=0.6, violation=0.4)],
        args=make_args(),
        codebase_context_excerpt="",
        compute_context_excerpt="",
        constraint_context_excerpt="",
        nc_strategy_excerpt="",
        research_excerpt="",
        current_priorities=[],
        sqlite_context_excerpt="SQLite context: total_records=1, feasible_records=0",
        iteration=1,
    )
    assert note["mode"] == "low_quality_recovery"
    assert note["low_quality_recovery"] is True
    assert note["acquisition_type"] == "LOW_QUALITY_RECOVERY"


def test_scientist_c_arbitration_coerces_grounded_evidence_refs() -> None:
    response = json.dumps(
        {
            "decision": "IMPLEMENT_A",
            "reason": "A is better.",
            "priority_updates": ["continue"],
            "diagnostic_focus": "",
            "revision_request": "",
            "selected_task_mode": "A",
            "acquisition_type": "IMPLEMENT_A",
            "evidence_refs": ["unknown_run_name"],
        }
    )
    client = QueueClient([response], model="stub-model")
    decision = ar.scientist_c_arbitrate(
        client,
        task={"nc": [2, 2, 2, 2], "seed_name": "reference", "seed": {"name": "reference"}},
        effective_task={
            "nc": [2, 2, 2, 2],
            "seed_name": "reference",
            "flow": {"Ffeed": 1.2, "F1": 2.0, "Fdes": 1.1, "Fex": 0.9, "Fraf": 0.8, "tstep": 9.2},
        },
        a_note={"decision": "propose", "reason": "based on run_31", "acquisition_type": "EXPLORE", "evidence_refs": ["run_31"]},
        b_note={
            "decision": "reject",
            "reason": "counter from run_31",
            "comparison_assessment": ["run_name=run_31 productivity=1.0"],
            "counterproposal_run": {
                "nc": [2, 2, 2, 2],
                "flow_adjustments": {"Ffeed": 0.0, "F1": 0.0, "Fdes": 0.0, "Fex": 0.0, "Fraf": 0.0, "tstep": 0.0},
                "expected_metric_effect": {
                    "delta_productivity": 0.0,
                    "delta_purity": 0.0,
                    "delta_recovery_ga": 0.0,
                    "delta_recovery_ma": 0.0,
                    "delta_violation": 0.0,
                },
                "physics_justification": "mass balance check",
            },
        },
        results=[sample_result("run_31", feasible=False, status="solver_error", j=None, productivity=0.6, purity=0.5, rga=0.6, rma=0.6, violation=0.4)],
        args=make_args(executive_llm_model="stub-model"),
        heuristics_context="",
        current_priorities=[],
        sqlite_context_excerpt="SQLite context: total_records=1, feasible_records=0",
        iteration=2,
        revision_count_recent=0,
        force_diagnostic_reason="",
    )
    assert decision["decision"] in {
        "IMPLEMENT_A",
        "IMPLEMENT_B_COUNTER",
        "IMPLEMENT_HYBRID",
        "RETURN_FOR_REVISION",
        "FORCE_DIAGNOSTIC",
    }
    assert decision["acquisition_type"] in {
        "IMPLEMENT_A",
        "IMPLEMENT_B_COUNTER",
        "IMPLEMENT_HYBRID",
        "RETURN_FOR_REVISION",
        "FORCE_DIAGNOSTIC",
    }
    assert decision.get("evidence_refs")
    assert "run_31" in decision.get("evidence_refs", [])


def test_live_results_jsonl_emits_expected_fields_and_order(tmp_path: Path) -> None:
    out = tmp_path / "live_results.jsonl"
    ar.initialize_live_results_stream(out)
    ar.append_live_results_event(
        out,
        {"event": "iteration_start", "job_id": "123", "run_name": "unit", "iteration": 1, "role": "main_loop"},
    )
    ar.append_live_results_event(
        out,
        {"event": "scientist_a_decision", "job_id": "123", "run_name": "unit", "iteration": 1, "role": "scientist_a_pick", "decision": "EXPLORE"},
    )
    ar.append_live_results_event(
        out,
        {
            "event": "simulation_complete",
            "job_id": "123",
            "run_name": "unit",
            "iteration": 1,
            "role": "simulation",
            "decision": "EXECUTE",
            "solver_status": "ok",
            "feasible": True,
            "productivity": 1.2,
        },
    )
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [line["event"] for line in lines] == ["iteration_start", "scientist_a_decision", "simulation_complete"]
    for line in lines:
        assert "timestamp_utc" in line
        assert "job_id" in line
        assert "run_name" in line
        assert "iteration" in line
        assert "role" in line


def test_malformed_responses_do_not_enter_llm_execution_path() -> None:
    tasks = candidate_tasks()
    bad_client = QueueClient(["not-json", "still-not-json"])
    _, a_note = ar.scientist_a_pick(
        bad_client,
        tasks,
        [sample_result("run_41", feasible=False, status="solver_error", j=None, productivity=0.7, purity=0.5, rga=0.6, rma=0.6, violation=0.45)],
        tried=set(),
        args=make_args(),
        objectives_excerpt="",
        soul_excerpt="",
        codebase_context_excerpt="",
        compute_context_excerpt="",
        constraint_context_excerpt="",
        nc_strategy_excerpt="",
        research_excerpt="",
        current_priorities=[],
        sqlite_context_excerpt="SQLite context: total_records=1, feasible_records=0",
        budget_used=0.0,
        iteration=1,
        heuristics_context="",
        convergence_context="",
    )
    b_note = ar.scientist_b_review(
        bad_client,
        task=tasks[0],
        effective_task={
            "nc": [2, 2, 2, 2],
            "seed_name": "reference",
            "flow": {"Ffeed": 1.2, "F1": 2.0, "Fdes": 1.1, "Fex": 0.9, "Fraf": 0.8, "tstep": 9.2},
        },
        best_result=None,
        results=[sample_result("run_41", feasible=False, status="solver_error", j=None, productivity=0.7, purity=0.5, rga=0.6, rma=0.6, violation=0.45)],
        args=make_args(),
        codebase_context_excerpt="",
        compute_context_excerpt="",
        constraint_context_excerpt="",
        nc_strategy_excerpt="",
        research_excerpt="",
        current_priorities=[],
        sqlite_context_excerpt="SQLite context: total_records=1, feasible_records=0",
        iteration=1,
    )
    assert a_note["mode"] != "llm"
    assert b_note["mode"] != "llm"
    assert a_note["acquisition_type"] == "LOW_QUALITY_RECOVERY"
    assert b_note["acquisition_type"] == "LOW_QUALITY_RECOVERY"
