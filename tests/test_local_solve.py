#!/usr/bin/env python3
"""Local quick-test for the SMB optimization pipeline.

Run this before submitting to PACE to verify:
1. The solver is available and the model builds correctly
2. A low-fidelity solve converges (not just "infeasible")
3. The two-phase feasibility-first approach finds a feasible point
4. Warm-start from Phase 1 improves Phase 2 convergence

Usage:
    python -m pytest tests/test_local_solve.py -v -s
    python tests/test_local_solve.py              # standalone
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# Ensure src/ is on the path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_test_model(nc=(2, 2, 2, 2), nfex=4, nfet=2, ncp=1):
    """Build a low-fidelity model for quick local testing."""
    from sembasmb import SMBConfig, FlowRates, build_inputs, build_model, apply_discretization

    config = SMBConfig(nc=nc, nfex=nfex, nfet=nfet, ncp=ncp)
    flow = FlowRates(
        F1=2.2, Fdes=1.2, Fex=0.9, Ffeed=1.3,
        Fraf=1.6, tstep=9.4,
    )
    inputs = build_inputs(config, flow)
    m = build_model(config, inputs)
    apply_discretization(m, config, inputs)
    return m, inputs, config


def require_available_solver() -> str:
    """Skip environment-dependent solve tests when IPOPT is unavailable."""
    from sembasmb import check_solver_available

    for solver in ("ipopt", "ipopt_sens"):
        if check_solver_available(solver):
            return solver
    pytest.skip("Neither ipopt nor ipopt_sens found on PATH")


def test_solver_available():
    """Check that IPOPT is available."""
    from sembasmb import check_solver_available
    for solver in ['ipopt', 'ipopt_sens']:
        available = check_solver_available(solver)
        print(f"  {solver}: {'available' if available else 'NOT FOUND'}")
    require_available_solver()


def test_model_builds():
    """Verify model builds without errors at low fidelity."""
    m, inputs, config = build_test_model()
    print(f"  Model built: {sum(1 for _ in m.component_objects())} components")
    assert hasattr(m, 'C'), "Model missing concentration variable C"
    assert hasattr(m, 'UF'), "Model missing feed velocity UF"


def test_direct_solve_low_fidelity():
    """Test direct (single-phase) solve at low fidelity."""
    from sembasmb import add_optimization, solve_model, default_ipopt_options
    from sembasmb import compute_outlet_averages, compute_purity_recovery

    solver_name = require_available_solver()
    m, inputs, config = build_test_model()
    add_optimization(m, inputs, purity_min=0.60, recovery_min_ga=0.75, recovery_min_ma=0.75)

    options = default_ipopt_options()
    options['linear_solver'] = 'mumps'  # Most portable
    options['max_iter'] = 1000
    options['print_level'] = 5  # Full IPOPT iteration table

    start = time.perf_counter()
    results = solve_model(m, solver_name=solver_name, options=options, tee=True)
    elapsed = time.perf_counter() - start

    term = str(results.solver.termination_condition)
    print(f"  Direct solve: {term} in {elapsed:.1f}s")
    print(f"  Solver status: {results.solver.status}")

    # Collect metrics even if infeasible
    try:
        outlets = compute_outlet_averages(m, inputs)
        metrics = compute_purity_recovery(m, inputs, outlets)
        print(f"  Purity: {metrics.get('purity_ex_meoh_free', 'N/A')}")
        print(f"  Recovery GA: {metrics.get('recovery_ex_GA', 'N/A')}")
        print(f"  Recovery MA: {metrics.get('recovery_ex_MA', 'N/A')}")
        print(f"  Productivity: {metrics.get('productivity_ex_ga_ma', 'N/A')}")
    except Exception as exc:
        print(f"  Metrics extraction failed: {exc}")

    return term


def test_two_phase_solve():
    """Test two-phase (feasibility-first) solve at low fidelity.

    Phase 1: Minimize constraint violation (find a feasible point).
    Phase 2: Maximize productivity from the feasible point with warm-start.
    """
    from sembasmb import (
        add_optimization, add_feasibility_objective, restore_productivity_objective,
        solve_model, default_ipopt_options, feasibility_restoration_options, warm_start_options,
        compute_outlet_averages, compute_purity_recovery,
    )

    solver_name = require_available_solver()
    m, inputs, config = build_test_model()
    add_optimization(m, inputs, purity_min=0.60, recovery_min_ga=0.75, recovery_min_ma=0.75)

    # --- Phase 1: Feasibility ---
    add_feasibility_objective(m, inputs)

    phase1_options = default_ipopt_options()
    phase1_options.update(feasibility_restoration_options())
    phase1_options['linear_solver'] = 'mumps'
    phase1_options['print_level'] = 5  # Full IPOPT iteration table

    start = time.perf_counter()
    phase1_results = solve_model(m, solver_name=solver_name, options=phase1_options, tee=True)
    phase1_elapsed = time.perf_counter() - start

    phase1_term = str(phase1_results.solver.termination_condition)
    print(f"  Phase 1 (feasibility): {phase1_term} in {phase1_elapsed:.1f}s")

    # Check slack values
    try:
        slack_ga = float(m.slack_recovery_ga.value) if m.slack_recovery_ga.value is not None else None
        slack_ma = float(m.slack_recovery_ma.value) if m.slack_recovery_ma.value is not None else None
        slack_purity = float(m.slack_purity.value) if m.slack_purity.value is not None else None
        total_slack = (slack_ga or 0.0) + (slack_ma or 0.0) + (slack_purity or 0.0)
        print(f"  Slacks: recovery_GA={slack_ga}, recovery_MA={slack_ma}, purity={slack_purity}")
        print(f"  Total slack: {total_slack:.6f} {'(FEASIBLE)' if total_slack < 1e-4 else '(still infeasible)'}")
    except Exception as exc:
        print(f"  Slack extraction failed: {exc}")
        total_slack = None

    # --- Phase 2: Productivity optimization ---
    restore_productivity_objective(m)

    phase2_options = default_ipopt_options()
    phase2_options.update(warm_start_options())
    phase2_options['linear_solver'] = 'mumps'
    phase2_options['print_level'] = 5  # Full IPOPT iteration table

    start = time.perf_counter()
    phase2_results = solve_model(m, solver_name=solver_name, options=phase2_options, tee=True)
    phase2_elapsed = time.perf_counter() - start

    phase2_term = str(phase2_results.solver.termination_condition)
    print(f"  Phase 2 (optimize): {phase2_term} in {phase2_elapsed:.1f}s")

    # Collect final metrics
    try:
        outlets = compute_outlet_averages(m, inputs)
        metrics = compute_purity_recovery(m, inputs, outlets)
        print(f"  Final purity: {metrics.get('purity_ex_meoh_free', 'N/A')}")
        print(f"  Final recovery GA: {metrics.get('recovery_ex_GA', 'N/A')}")
        print(f"  Final recovery MA: {metrics.get('recovery_ex_MA', 'N/A')}")
        print(f"  Final productivity: {metrics.get('productivity_ex_ga_ma', 'N/A')}")
    except Exception as exc:
        print(f"  Final metrics extraction failed: {exc}")

    return phase1_term, phase2_term


def test_multiple_layouts():
    """Test a few NC layouts to see which ones are feasible at low fidelity."""
    from sembasmb import (
        add_optimization, solve_model, default_ipopt_options,
        compute_outlet_averages, compute_purity_recovery,
    )

    solver_name = require_available_solver()
    layouts = [(1, 2, 3, 2), (2, 2, 2, 2), (1, 3, 2, 2), (2, 1, 3, 2)]
    results = []

    for nc in layouts:
        m, inputs, config = build_test_model(nc=nc)
        add_optimization(m, inputs, purity_min=0.60, recovery_min_ga=0.75, recovery_min_ma=0.75)

        options = default_ipopt_options()
        options['linear_solver'] = 'mumps'
        options['max_iter'] = 500
        options['print_level'] = 5  # Full IPOPT iteration table

        start = time.perf_counter()
        try:
            res = solve_model(m, solver_name=solver_name, options=options, tee=True)
            term = str(res.solver.termination_condition)
        except Exception as exc:
            term = f"error: {exc}"
        elapsed = time.perf_counter() - start

        purity = None
        prod = None
        try:
            outlets = compute_outlet_averages(m, inputs)
            metrics = compute_purity_recovery(m, inputs, outlets)
            purity = metrics.get('purity_ex_meoh_free')
            prod = metrics.get('productivity_ex_ga_ma')
        except Exception:
            pass

        results.append((nc, term, elapsed, purity, prod))
        print(f"  nc={nc}: {term} ({elapsed:.1f}s) purity={purity} prod={prod}")
    assert len(results) == len(layouts)


def test_warm_start_from_reference():
    """Test that a reference-eval solve can warm-start an optimization run.

    1. Run a fixed-flow reference solve (no optimization).
    2. Capture the solved model state.
    3. Build a new optimization model and initialize it from the reference state.
    4. Run two-phase optimization from the warm-started point.
    5. Compare convergence to a cold-start optimization.
    """
    from sembasmb import (
        SMBConfig, FlowRates, build_inputs, build_model, apply_discretization,
        add_optimization, add_feasibility_objective, restore_productivity_objective,
        solve_model, default_ipopt_options, feasibility_restoration_options, warm_start_options,
        compute_outlet_averages, compute_purity_recovery,
    )

    solver_name = require_available_solver()
    nc = (2, 2, 2, 2)
    nfex, nfet, ncp = 4, 2, 1

    # --- Step 1: Reference-eval (fixed-flow solve) ---
    config = SMBConfig(nc=nc, nfex=nfex, nfet=nfet, ncp=ncp)
    flow = FlowRates(F1=2.2, Fdes=1.2, Fex=0.9, Ffeed=1.3, Fraf=1.6, tstep=9.4)
    inputs = build_inputs(config, flow)
    m_ref = build_model(config, inputs)
    apply_discretization(m_ref, config, inputs)

    options = default_ipopt_options()
    options['linear_solver'] = 'mumps'
    options['max_iter'] = 1000
    options['print_level'] = 5  # Full IPOPT iteration table

    start = time.perf_counter()
    ref_results = solve_model(m_ref, solver_name=solver_name, options=options, tee=True)
    ref_elapsed = time.perf_counter() - start
    ref_term = str(ref_results.solver.termination_condition)
    print(f"  Reference solve: {ref_term} in {ref_elapsed:.1f}s")

    # --- Step 2: Capture model state ---
    # Import the extraction function from benchmarks
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    from run_stage import extract_model_state, apply_warm_start_state

    state = extract_model_state(m_ref, inputs)
    n_c_vals = len(state.get("C", {}))
    print(f"  Captured state: {n_c_vals} C values, UF={state['UF']:.4f}, tstep={state['tstep']:.2f}")
    assert n_c_vals > 0, "No concentration values captured"

    # --- Step 3: Build optimization model and warm-start ---
    config2 = SMBConfig(nc=nc, nfex=nfex, nfet=nfet, ncp=ncp)
    flow2 = FlowRates(F1=2.2, Fdes=1.2, Fex=0.9, Ffeed=1.3, Fraf=1.6, tstep=9.4)
    inputs2 = build_inputs(config2, flow2)
    m_warm = build_model(config2, inputs2)
    apply_discretization(m_warm, config2, inputs2)
    add_optimization(m_warm, inputs2, purity_min=0.60, recovery_min_ga=0.75, recovery_min_ma=0.75)

    # Apply warm-start state from reference eval
    apply_warm_start_state(m_warm, state)

    # --- Step 4: Two-phase solve from warm-started point ---
    add_feasibility_objective(m_warm, inputs2)
    p1_options = default_ipopt_options()
    p1_options.update(feasibility_restoration_options())
    p1_options['linear_solver'] = 'mumps'
    p1_options['print_level'] = 5  # Full IPOPT iteration table

    start = time.perf_counter()
    p1_results = solve_model(m_warm, solver_name=solver_name, options=p1_options, tee=True)
    p1_elapsed = time.perf_counter() - start
    p1_term = str(p1_results.solver.termination_condition)
    print(f"  Warm-start Phase 1: {p1_term} in {p1_elapsed:.1f}s")

    restore_productivity_objective(m_warm)
    p2_options = default_ipopt_options()
    p2_options.update(warm_start_options())
    p2_options['linear_solver'] = 'mumps'
    p2_options['print_level'] = 5  # Full IPOPT iteration table

    start = time.perf_counter()
    p2_results = solve_model(m_warm, solver_name=solver_name, options=p2_options, tee=True)
    p2_elapsed = time.perf_counter() - start
    p2_term = str(p2_results.solver.termination_condition)
    print(f"  Warm-start Phase 2: {p2_term} in {p2_elapsed:.1f}s")

    try:
        outlets = compute_outlet_averages(m_warm, inputs2)
        metrics = compute_purity_recovery(m_warm, inputs2, outlets)
        print(f"  Warm-start purity: {metrics.get('purity_ex_meoh_free', 'N/A')}")
        print(f"  Warm-start productivity: {metrics.get('productivity_ex_ga_ma', 'N/A')}")
    except Exception as exc:
        print(f"  Warm-start metrics failed: {exc}")

    return ref_term, p1_term, p2_term


if __name__ == "__main__":
    print("\n=== 1. Solver Availability ===")
    try:
        test_solver_available()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("\n=== 2. Model Build ===")
    test_model_builds()

    print("\n=== 3. Direct Solve (single-phase) ===")
    direct_term = test_direct_solve_low_fidelity()

    print("\n=== 4. Two-Phase Solve (feasibility-first) ===")
    phase1_term, phase2_term = test_two_phase_solve()

    print("\n=== 5. Warm-Start from Reference Eval ===")
    ref_term, ws_p1_term, ws_p2_term = test_warm_start_from_reference()

    print("\n=== 6. Multiple Layout Screening ===")
    layout_results = test_multiple_layouts()

    print("\n=== Summary ===")
    print(f"  Direct solve: {direct_term}")
    print(f"  Two-phase: Phase1={phase1_term}, Phase2={phase2_term}")
    print(f"  Warm-start: Ref={ref_term}, Phase1={ws_p1_term}, Phase2={ws_p2_term}")
    feasible_layouts = sum(1 for _, term, _, _, _ in layout_results if term == "optimal")
    print(f"  Layout screening: {feasible_layouts}/{len(layout_results)} converged to optimal")

    if ws_p2_term in ("optimal", "acceptable"):
        print("\n  Warm-start optimization succeeded — ready for PACE deployment.")
    elif phase2_term in ("optimal", "acceptable"):
        print("\n  Two-phase solve succeeded but warm-start didn't improve.")
    elif direct_term in ("optimal", "acceptable"):
        print("\n  Direct solve succeeded but two-phase didn't improve — check Phase 1 slack values.")
    else:
        print("\n  Both approaches infeasible at low fidelity — investigate initial point or constraint targets.")
