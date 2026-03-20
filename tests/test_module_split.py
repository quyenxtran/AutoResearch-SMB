"""
Tests verifying that the benchmarks module split is correct:
- Each new module (agent_results, agent_evidence, agent_llm_client, agent_db,
  agent_policy, agent_scientists) is importable.
- Key public symbols are present and callable/correct type.
- Smoke tests for pure-Python functions that require no external dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path (mirrors how benchmarks/run_stage.py does it)
REPO_ROOT = Path(__file__).resolve().parents[1]
SMB_SRC = str(REPO_ROOT / "src")
if SMB_SRC not in sys.path:
    sys.path.insert(0, SMB_SRC)


class TestAgentResultsImports:
    def test_module_importable(self):
        from benchmarks import agent_results  # noqa: F401

    def test_as_float(self):
        from benchmarks.agent_results import as_float
        assert callable(as_float)
        assert as_float(1.5) == 1.5
        assert as_float(None) is None
        assert as_float("bad") is None

    def test_layout_text(self):
        from benchmarks.agent_results import layout_text
        assert layout_text([1, 2, 3, 2]) == "1,2,3,2"
        assert layout_text(None) == ""

    def test_is_reference_seed_name(self):
        from benchmarks.agent_results import is_reference_seed_name
        assert is_reference_seed_name("reference") is True
        assert is_reference_seed_name("Reference") is True
        assert is_reference_seed_name("notebook") is False
        assert is_reference_seed_name(None) is False

    def test_layout_text_nc_key_equiv(self):
        # layout_text in agent_results does the same joining as nc_key in agent_policy
        from benchmarks.agent_results import layout_text
        assert layout_text([1, 2, 3, 2]) == "1,2,3,2"

    def test_effective_violation_returns_float(self):
        from benchmarks.agent_results import effective_violation
        result = {"status": "ok", "feasible": True}
        assert isinstance(effective_violation(result), float)

    def test_search_score_returns_tuple(self):
        from benchmarks.agent_results import search_score
        result = {"feasible": True, "metrics": {"productivity_ex_ga_ma": 1.5}, "status": "ok"}
        score = search_score(result)
        assert isinstance(score, tuple)
        assert len(score) == 3

    def test_has_any_feasible(self):
        from benchmarks.agent_results import has_any_feasible
        assert has_any_feasible([{"feasible": True}]) is True
        assert has_any_feasible([{"feasible": False}]) is False
        assert has_any_feasible([]) is False

    def test_rank_any_results(self):
        from benchmarks.agent_results import rank_any_results
        results = [
            {"feasible": False, "status": "ok"},
            {"feasible": True, "status": "ok", "metrics": {"productivity_ex_ga_ma": 2.0}},
        ]
        ranked = rank_any_results(results)
        assert ranked[0].get("feasible") is True

    def test_linear_slope_basic(self):
        from benchmarks.agent_results import linear_slope
        slope = linear_slope([0, 1, 2], [0, 2, 4])
        assert slope is not None
        assert abs(slope - 2.0) < 1e-9

    def test_deterministic_select(self):
        from benchmarks.agent_results import deterministic_select
        tasks = [
            {"nc": [1, 2, 3, 2], "seed_name": "reference"},
            {"nc": [2, 2, 2, 2], "seed_name": "reference"},
        ]
        tried = set()
        idx = deterministic_select(tasks, tried)
        assert idx == 0

    def test_low_fidelity_limits(self):
        from benchmarks.agent_results import low_fidelity_limits
        import argparse
        args = argparse.Namespace(
            finalization_low_fidelity_nfex=5,
            finalization_low_fidelity_nfet=2,
            finalization_low_fidelity_ncp=1,
        )
        limits = low_fidelity_limits(args)
        assert limits["nfex"] == 5
        assert limits["nfet"] == 2
        assert limits["ncp"] == 1


class TestAgentEvidenceImports:
    def test_module_importable(self):
        from benchmarks import agent_evidence  # noqa: F401

    def test_normalize_text_list(self):
        from benchmarks.agent_evidence import normalize_text_list
        result = normalize_text_list(["hello", "world"], max_items=5)
        assert result == ["hello", "world"]
        result_str = normalize_text_list("line1\nline2", max_items=5)
        assert "line1" in result_str

    def test_compact_prompt_block(self):
        from benchmarks.agent_evidence import compact_prompt_block
        text = "hello\nhello\nworld"
        result = compact_prompt_block(text, max_chars=200)
        # Duplicate lines should be removed
        assert result.count("hello") == 1

    def test_compact_prompt_block_truncation(self):
        from benchmarks.agent_evidence import compact_prompt_block
        text = "a" * 1000
        result = compact_prompt_block(text, max_chars=50)
        assert len(result) <= 51  # allow for ellipsis char

    def test_build_evidence_pack(self):
        from benchmarks.agent_evidence import build_evidence_pack
        results = [
            {"run_name": "run1", "feasible": True, "status": "ok", "J_validated": 1.5,
             "metrics": {"productivity_ex_ga_ma": 1.5}},
            {"run_name": "run2", "feasible": False, "status": "ok"},
        ]
        pack = build_evidence_pack(results)
        assert "recent_runs" in pack
        assert "top_feasible" in pack
        assert "top_infeasible" in pack
        assert "run_name_catalog" in pack

    def test_budget_evidence_pack_json(self):
        from benchmarks.agent_evidence import build_evidence_pack, budget_evidence_pack_json
        results = [{"run_name": "run1", "feasible": True, "status": "ok"}]
        pack = build_evidence_pack(results)
        result = budget_evidence_pack_json(pack, max_chars=500)
        assert isinstance(result, str)
        assert len(result) <= 500

    def test_contains_run_reference(self):
        from benchmarks.agent_evidence import contains_run_reference
        assert contains_run_reference(["run1 is best"], ["run1"]) is True
        assert contains_run_reference(["nothing here"], ["run1"]) is False

    def test_text_mentions_flow_signals(self):
        from benchmarks.agent_evidence import text_mentions_flow_signals
        assert text_mentions_flow_signals(["Ffeed=3.0, F1=4.0"]) is True
        # "flow" matches the regex (case-insensitive); use a string with no flow keywords
        assert text_mentions_flow_signals(["no rate information here"]) is False

    def test_extract_nc_mentions(self):
        from benchmarks.agent_evidence import extract_nc_mentions
        result = extract_nc_mentions("candidate nc=[1,2,3,2] is proposed")
        assert (1, 2, 3, 2) in result

    def test_apply_flow_adjustments(self):
        from benchmarks.agent_evidence import apply_flow_adjustments
        base = {"Ffeed": 3.0, "F1": 5.0, "Fdes": 2.0, "Fex": 1.5, "Fraf": 0.5, "tstep": 10.0}
        adjusted = apply_flow_adjustments(base, {"Ffeed": 0.5, "tstep": -1.0})
        assert abs(adjusted["Ffeed"] - 3.5) < 1e-9
        assert abs(adjusted["tstep"] - 9.0) < 1e-9

    def test_markdown_focused_excerpt_missing_file(self):
        from benchmarks.agent_evidence import markdown_focused_excerpt
        result = markdown_focused_excerpt("/nonexistent/path.md", ["foo"], max_chars=100)
        assert "Missing" in result


class TestAgentLlmClientImports:
    def test_module_importable(self):
        from benchmarks import agent_llm_client  # noqa: F401

    def test_openai_compat_client_disabled(self):
        from benchmarks.agent_llm_client import OpenAICompatClient
        client = OpenAICompatClient(
            base_url="http://localhost:11434/v1",
            model="test-model",
            enabled=False,
        )
        assert not client.enabled
        result = client.chat("system", "user")
        assert result is None

    def test_extract_json_basic(self):
        from benchmarks.agent_llm_client import OpenAICompatClient
        result = OpenAICompatClient.extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extract_json_with_think_block(self):
        from benchmarks.agent_llm_client import OpenAICompatClient
        text = '<think>reasoning here</think>{"answer": 42}'
        result = OpenAICompatClient.extract_json(text)
        assert result == {"answer": 42}

    def test_extract_json_none_returns_none(self):
        from benchmarks.agent_llm_client import OpenAICompatClient
        assert OpenAICompatClient.extract_json(None) is None
        assert OpenAICompatClient.extract_json("") is None

    def test_required_keys_missing(self):
        from benchmarks.agent_llm_client import required_keys_missing
        data = {"a": 1, "b": 2}
        missing = required_keys_missing(data, ["a", "b", "c"])
        assert missing == ["c"]

    def test_utc_now_text(self):
        from benchmarks.agent_llm_client import utc_now_text
        text = utc_now_text()
        assert "UTC" in text
        assert len(text) > 10


class TestAgentDbImports:
    def test_module_importable(self):
        from benchmarks import agent_db  # noqa: F401

    def test_open_sqlite_db_in_memory(self, tmp_path):
        from benchmarks.agent_db import open_sqlite_db
        db_path = str(tmp_path / "test.sqlite")
        conn = open_sqlite_db(db_path)
        assert conn is not None
        conn.close()

    def test_sqlite_record_count_empty(self, tmp_path):
        from benchmarks.agent_db import open_sqlite_db, sqlite_record_count
        db_path = str(tmp_path / "test.sqlite")
        conn = open_sqlite_db(db_path)
        count = sqlite_record_count(conn)
        assert count == 0
        conn.close()

    def test_sqlite_history_context_empty(self, tmp_path):
        from benchmarks.agent_db import open_sqlite_db, sqlite_history_context
        db_path = str(tmp_path / "test.sqlite")
        conn = open_sqlite_db(db_path)
        context = sqlite_history_context(conn)
        assert isinstance(context, str)
        assert "total_records=0" in context
        conn.close()

    def test_sqlite_layout_trend_table_empty(self, tmp_path):
        from benchmarks.agent_db import open_sqlite_db, sqlite_layout_trend_table
        db_path = str(tmp_path / "test.sqlite")
        conn = open_sqlite_db(db_path)
        result = sqlite_layout_trend_table(conn)
        assert isinstance(result, str)
        conn.close()

    def test_sqlite_total_records_from_excerpt(self):
        from benchmarks.agent_policy import sqlite_total_records_from_excerpt
        assert sqlite_total_records_from_excerpt("total_records=42, feasible=3") == 42
        assert sqlite_total_records_from_excerpt("no match here") == 0

    def test_merge_priority_board(self):
        from benchmarks.agent_db import merge_priority_board
        current = ["priority A", "priority B"]
        note = {"priority_updates": ["priority C"]}
        merged = merge_priority_board(current, note)
        assert "priority A" in merged
        assert "priority C" in merged

    def test_read_research_tail_missing(self, tmp_path):
        from benchmarks.agent_db import read_research_tail
        path = tmp_path / "nonexistent.md"
        result = read_research_tail(path, max_chars=500)
        assert "No research log" in result


class TestAgentPolicyImports:
    def test_module_importable(self):
        from benchmarks import agent_policy  # noqa: F401

    def test_nc_key(self):
        from benchmarks.agent_policy import nc_key
        assert nc_key([1, 2, 3, 2]) == "1,2,3,2"
        assert nc_key([2, 2, 2, 2]) == "2,2,2,2"

    def test_nc_prior_score_symmetric(self):
        from benchmarks.agent_policy import nc_prior_score
        score_sym = nc_prior_score([2, 2, 2, 2])
        score_asym = nc_prior_score([1, 1, 1, 5])
        assert score_sym > score_asym

    def test_sqlite_total_records_from_excerpt(self):
        from benchmarks.agent_policy import sqlite_total_records_from_excerpt
        assert sqlite_total_records_from_excerpt("SQLite context: total_records=7, feasible=2") == 7

    def test_deterministic_review_approve(self):
        from benchmarks.agent_policy import deterministic_review
        candidate = {"nc": [1, 2, 3, 2], "seed_name": "reference"}
        result = deterministic_review(candidate, None)
        assert result["decision"] == "approve"

    def test_deterministic_review_reject_duplicate(self):
        from benchmarks.agent_policy import deterministic_review
        candidate = {"nc": [1, 2, 3, 2], "seed_name": "reference"}
        best = {"nc": [1, 2, 3, 2], "seed_name": "reference", "run_name": "run1"}
        result = deterministic_review(candidate, best)
        assert result["decision"] == "reject"

    def test_single_scientist_policy_review_mode(self):
        from benchmarks.agent_policy import single_scientist_policy_review
        candidate = {"nc": [1, 2, 3, 2], "seed_name": "reference"}
        result = single_scientist_policy_review(candidate, None)
        assert result["mode"] == "single_scientist_policy"

    def test_check_systematic_infeasibility_not_triggered(self):
        from benchmarks.agent_policy import check_systematic_infeasibility
        result = check_systematic_infeasibility([], k=5)
        assert result["triggered"] is False

    def test_check_systematic_infeasibility_triggered(self):
        from benchmarks.agent_policy import check_systematic_infeasibility
        bad_results = [
            {"run_name": f"run{i}", "feasible": False, "status": "solver_error"}
            for i in range(5)
        ]
        result = check_systematic_infeasibility(bad_results, k=5)
        assert result["triggered"] is True

    def test_physics_informed_select_no_tasks(self):
        from benchmarks.agent_policy import physics_informed_select
        tasks = [{"nc": [1, 2, 3, 2], "seed_name": "reference"}]
        tried = {(tuple([1, 2, 3, 2]), "reference")}
        idx, info = physics_informed_select(tasks, tried, [])
        # Falls back when all tried
        assert idx == 0
        assert info["mode"] == "physics_informed_fallback"

    def test_build_search_tasks_returns_list(self):
        from benchmarks.agent_policy import build_search_tasks
        import argparse
        # seed_library="notebook" selects the NOTEBOOK_SEEDS set; "reference" is a seed name not a library
        args = argparse.Namespace(
            nc_library="1,2,3,2;2,2,2,2",
            seed_library="notebook",
        )
        tasks = build_search_tasks(args)
        assert isinstance(tasks, list)
        # At minimum one task per nc layout
        assert len(tasks) >= 2

    def test_probe_reference_runs_required(self):
        from benchmarks.agent_policy import probe_reference_runs_required, build_search_tasks
        import argparse
        args = argparse.Namespace(
            nc_library="1,2,3,2",
            seed_library="notebook",
            min_probe_reference_runs=1,
        )
        tasks = build_search_tasks(args)
        required = probe_reference_runs_required(args, tasks)
        assert required >= 0

    def test_executive_controller_decide_approve(self):
        from benchmarks.agent_policy import executive_controller_decide
        import argparse
        args = argparse.Namespace(
            executive_controller_enabled=True,
            executive_trigger_rejects=2,
            executive_force_after_rejects=3,
            executive_top_k_lock=5,
        )
        tasks = [{"nc": [1, 2, 3, 2], "seed_name": "reference"}]
        b_note = {"decision": "approve"}
        result = executive_controller_decide(
            args, tasks, set(), 0, tasks[0], b_note, [], 0
        )
        assert result["decision"] == "not_needed"


class TestAgentScientistsImports:
    def test_module_importable(self):
        from benchmarks import agent_scientists  # noqa: F401

    def test_default_initial_priority_plan_structure(self):
        from benchmarks.agent_scientists import default_initial_priority_plan
        import argparse
        args = argparse.Namespace(
            nc_library="1,2,3,2;2,2,2,2",
            seed_library="reference",
            solver_name="ipopt",
            linear_solver="mumps",
        )
        plan = default_initial_priority_plan(args)
        assert "priorities" in plan
        assert "proposed_simulations" in plan
        assert "risks" in plan
        assert "nc_screening_strategy" in plan
        assert len(plan["priorities"]) > 0

    def test_scientist_a_pick_no_tasks(self):
        from benchmarks.agent_scientists import scientist_a_pick
        from benchmarks.agent_llm_client import OpenAICompatClient
        import argparse
        client = OpenAICompatClient("", "", enabled=False)
        args = argparse.Namespace(
            benchmark_hours=12.0,
            search_hours=10.0,
            validation_hours=2.0,
            min_probe_reference_runs=0,
        )
        idx, note = scientist_a_pick(
            client=client,
            candidate_tasks=[],
            results=[],
            tried=set(),
            args=args,
            objectives_excerpt="",
            soul_excerpt="",
            codebase_context_excerpt="",
            compute_context_excerpt="",
            constraint_context_excerpt="",
            nc_strategy_excerpt="",
            research_excerpt="",
            current_priorities=[],
            sqlite_context_excerpt="",
            budget_used=0.0,
            iteration=1,
        )
        assert isinstance(idx, int)
        assert isinstance(note, dict)

    def test_scientist_b_review_deterministic_fallback(self):
        from benchmarks.agent_scientists import scientist_b_review
        from benchmarks.agent_llm_client import OpenAICompatClient
        import argparse
        client = OpenAICompatClient("", "", enabled=False)
        args = argparse.Namespace()
        task = {"nc": [1, 2, 3, 2], "seed_name": "reference"}
        result = scientist_b_review(
            client=client,
            task=task,
            effective_task=task,
            best_result=None,
            results=[],
            args=args,
            codebase_context_excerpt="",
            compute_context_excerpt="",
            constraint_context_excerpt="",
            nc_strategy_excerpt="",
            research_excerpt="",
            current_priorities=[],
            sqlite_context_excerpt="",
            iteration=1,
        )
        assert "decision" in result

    def test_scientist_c_arbitrate_deterministic_fallback(self):
        from benchmarks.agent_scientists import scientist_c_arbitrate
        from benchmarks.agent_llm_client import OpenAICompatClient
        import argparse
        client = OpenAICompatClient("", "", enabled=False)
        args = argparse.Namespace(
            executive_llm_model="",
            executive_max_revisions=1,
        )
        task = {"nc": [1, 2, 3, 2], "seed_name": "reference"}
        a_note = {"decision": "approve", "reason": "test", "acquisition_type": "EXPLORE"}
        b_note = {"decision": "reject", "reason": "test"}
        result = scientist_c_arbitrate(
            client=client,
            task=task,
            effective_task=task,
            a_note=a_note,
            b_note=b_note,
            results=[],
            args=args,
            heuristics_context="",
            current_priorities=[],
            sqlite_context_excerpt="",
            iteration=1,
        )
        assert "decision" in result
        assert result["decision"] in {
            "IMPLEMENT_A", "IMPLEMENT_B_COUNTER", "IMPLEMENT_HYBRID",
            "RETURN_FOR_REVISION", "FORCE_DIAGNOSTIC",
        }


class TestCrossModuleConsistency:
    """Verify that the same function names appearing in multiple modules produce the same results."""

    def test_nc_key_matches_layout_text(self):
        from benchmarks.agent_results import layout_text
        from benchmarks.agent_policy import nc_key
        nc = [1, 2, 3, 2]
        # Both produce the same "1,2,3,2" representation
        assert layout_text(nc) == nc_key(nc)

    def test_normalize_text_list_consistent(self):
        from benchmarks.agent_evidence import normalize_text_list as ntl_evidence
        from benchmarks.agent_db import normalize_text_list as ntl_db
        value = ["a", "b", "c"]
        assert ntl_evidence(value) == ntl_db(value)

    def test_effective_violation_consistent(self):
        from benchmarks.agent_results import effective_violation as ev_results
        from benchmarks.agent_evidence import effective_violation as ev_evidence
        result = {"status": "ok", "feasible": True}
        assert ev_results(result) == ev_evidence(result)

    def test_compact_prompt_block_consistent(self):
        from benchmarks.agent_evidence import compact_prompt_block as cpb_evidence
        from benchmarks.agent_db import compact_prompt_block as cpb_db
        text = "hello world\nhello world"
        assert cpb_evidence(text) == cpb_db(text)


class TestAgentRunnerDelegation:
    def test_agent_runner_uses_split_scientist_entrypoints(self):
        from benchmarks import agent_runner
        from benchmarks import agent_scientists

        assert agent_runner.default_initial_priority_plan is agent_scientists.default_initial_priority_plan
        assert agent_runner.initial_priority_plan is agent_scientists.initial_priority_plan
        assert agent_runner.scientist_a_pick is agent_scientists.scientist_a_pick
        assert agent_runner.scientist_b_review is agent_scientists.scientist_b_review
        assert agent_runner.scientist_c_arbitrate is agent_scientists.scientist_c_arbitrate

    def test_agent_runner_uses_split_policy_entrypoints(self):
        from benchmarks import agent_runner
        from benchmarks import agent_policy

        assert agent_runner.configure_stage_args is agent_policy.configure_stage_args
        assert agent_runner.build_search_tasks is agent_policy.build_search_tasks
        assert agent_runner.apply_probe_reference_gate is agent_policy.apply_probe_reference_gate
        assert agent_runner.probe_reference_runs_required is agent_policy.probe_reference_runs_required
        assert agent_runner.search_execution_policy is agent_policy.search_execution_policy
        assert agent_runner.single_scientist_policy_review is agent_policy.single_scientist_policy_review
        assert agent_runner.executive_controller_decide is agent_policy.executive_controller_decide
        assert agent_runner.physics_informed_select is agent_policy.physics_informed_select
        assert agent_runner.check_systematic_infeasibility is agent_policy.check_systematic_infeasibility

    def test_agent_runner_uses_split_db_evidence_results_and_llm_entrypoints(self):
        from benchmarks import agent_db
        from benchmarks import agent_evidence
        from benchmarks import agent_llm_client
        from benchmarks import agent_results
        from benchmarks import agent_runner

        assert agent_runner.open_sqlite_db is agent_db.open_sqlite_db
        assert agent_runner.persist_result_to_sqlite is agent_db.persist_result_to_sqlite
        assert agent_runner.record_convergence_snapshot is agent_db.record_convergence_snapshot
        assert agent_runner.sqlite_history_context is agent_db.sqlite_history_context
        assert agent_runner.sqlite_layout_trend_table is agent_db.sqlite_layout_trend_table
        assert agent_runner.append_iteration_research is agent_db.append_iteration_research
        assert agent_runner.append_result_research is agent_db.append_result_research
        assert agent_runner.merge_priority_board is agent_db.merge_priority_board

        assert agent_runner.build_evidence_pack is agent_evidence.build_evidence_pack
        assert agent_runner.coerce_evidence_list is agent_evidence.coerce_evidence_list
        assert agent_runner.coerce_grounded_evidence_refs is agent_evidence.coerce_grounded_evidence_refs
        assert agent_runner.compact_prompt_block is agent_evidence.compact_prompt_block
        assert agent_runner.failure_recovery_context is agent_evidence.failure_recovery_context
        assert agent_runner.hypothesis_matcher is agent_evidence.hypothesis_matcher

        assert agent_runner.as_float is agent_results.as_float
        assert agent_runner.effective_flow is agent_results.effective_flow
        assert agent_runner.effective_violation is agent_results.effective_violation
        assert agent_runner.rank_any_results is agent_results.rank_any_results
        assert agent_runner.summarize_result is agent_results.summarize_result

        assert agent_runner.OpenAICompatClient is agent_llm_client.OpenAICompatClient
        assert agent_runner.request_json_with_single_repair is agent_llm_client.request_json_with_single_repair
