from __future__ import annotations

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

# run_stage.py imports pyomo.environ at module import time. Stub the minimum
# surface only when pyomo is genuinely unavailable so the full suite can still
# exercise real pyomo modules later in the same session.
if "pyomo.environ" not in sys.modules and importlib.util.find_spec("pyomo.environ") is None:
    pyomo_mod = types.ModuleType("pyomo")
    pyomo_env_mod = types.ModuleType("pyomo.environ")
    pyomo_env_mod.value = lambda x: x
    pyomo_mod.environ = pyomo_env_mod
    sys.modules["pyomo"] = pyomo_mod
    sys.modules["pyomo.environ"] = pyomo_env_mod

import run_stage as rs  # noqa: E402


def test_slurm_defaults_export_parallel_ipopt_controls():
    slurm_path = REPO_ROOT / "slurm" / "pace_smb_two_scientists_qwen.slurm"
    text = slurm_path.read_text(encoding="utf-8")
    assert "SMB_IPOPT_WORKERS" in text
    assert "SMB_IPOPT_THREADS_PER_WORKER" in text
    assert 'SMB_IPOPT_WORKERS="${SMB_IPOPT_WORKERS:-2}"' in text
    assert 'SMB_IPOPT_THREADS_PER_WORKER="${SMB_IPOPT_THREADS_PER_WORKER:-2}"' in text


def test_resolve_ipopt_parallel_profile_uses_configured_workers_and_threads(monkeypatch: pytest.MonkeyPatch):
    if not hasattr(rs, "resolve_ipopt_parallel_profile"):
        pytest.skip("run_stage.resolve_ipopt_parallel_profile is unavailable")

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    monkeypatch.setenv("SMB_IPOPT_WORKERS", "2")
    monkeypatch.setenv("SMB_IPOPT_THREADS_PER_WORKER", "2")

    profile = rs.resolve_ipopt_parallel_profile(Namespace(ipopt_workers=2, ipopt_threads_per_worker=2))

    assert profile["workers"] == 2
    assert profile["threads_per_worker"] == 2
    assert profile["cpu_budget"] == 6


def test_ipopt_accounting_cpus_uses_threads_when_explicit(monkeypatch: pytest.MonkeyPatch):
    if not hasattr(rs, "ipopt_accounting_cpus"):
        pytest.skip("run_stage.ipopt_accounting_cpus is unavailable")

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    args = Namespace(ipopt_workers=2, ipopt_threads_per_worker=2)

    assert rs.ipopt_accounting_cpus(args) == 2


def test_ordering_helper_preserves_submission_order(monkeypatch: pytest.MonkeyPatch):
    if not hasattr(rs, "run_parallel_stage_tasks"):
        pytest.skip("run_stage.run_parallel_stage_tasks is unavailable")

    class FakeFuture:
        def __init__(self, value: dict[str, int]) -> None:
            self._value = value

        def result(self):
            return self._value

    class FakeExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.submitted: list[dict[str, int]] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, payload):
            self.submitted.append(payload)
            return FakeFuture({"status": "ok", "idx": payload["idx"]})

    def fake_as_completed(futures):
        return list(reversed(list(futures)))

    payloads = [
        {"stage": "candidate", "args": {"run_name": "r0"}, "nc": [1, 2, 3, 2], "idx": 0},
        {"stage": "candidate", "args": {"run_name": "r1"}, "nc": [2, 2, 2, 2], "idx": 1},
        {"stage": "candidate", "args": {"run_name": "r2"}, "nc": [1, 3, 2, 2], "idx": 2},
    ]

    monkeypatch.setattr(rs.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(rs.concurrent.futures, "as_completed", fake_as_completed)

    result = rs.run_parallel_stage_tasks(payloads, workers=2, threads_per_worker=2)
    assert [item["idx"] for item in result] == [0, 1, 2]
