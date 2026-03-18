# Problem Definition

## Core Question

Find high-productivity SMB operating conditions for the Kraton-feed case while satisfying project quality constraints and hardware limits, and do so with compute-efficient search.

This project asks two linked questions:

1. What is the best feasible operating point?
2. What search policy finds it most reliably under fixed budget?

## Optimization Problem

Objective:

- maximize `productivity_ex_ga_ma = (CE_GA + CE_MA) * UE * area * eb`

Hard constraints:

- quality targets (runtime-configurable project thresholds)
- flow bounds and pump limits
- flow consistency (`F1 = Fdes + Fex = Ffeed + Fraf`)
- fixed hardware (`8` columns total), layout as admissible `nc` partitions

Main decision variables:

- `nc`, `Ffeed`, `F1`, `Fdes`, `Fex`, `tstep`

Derived (normally):

- `Fraf`

## What Kind of Optimization Problem Is This?

With fixed `nc`: large nonconvex NLP from discretized dynamic SMB equations.  
With variable `nc`: mixed discrete + continuous optimization (MINLP-style benchmark in practice).

Global optimality is generally not provable with local NLP solvers alone.  
Target is a high-confidence, validated, globally competitive solution under finite budget.

## Direct vs Agentic Strategy

Direct-only strategy:

- simpler and easier to benchmark
- can stall in local basins or infeasible regions

Agentic strategy:

- adapts fidelity and search region using evidence
- can improve sample efficiency
- needs strict review/guardrails to avoid heuristic drift

## Practical Benchmark Framing

Keep fixed across methods:

- same feed and model assumptions
- same admissible NC library
- same bounds and hard constraints
- same final high-fidelity validator

Compare by:

- feasibility rate
- best validated productivity
- time/simulations to best feasible
- robustness of final candidate

## Fixed-Budget Rule

Use the actual job budget from environment.  
In benchmark mode, enforce a counted SMB numerical budget (not "infinite retries").

Typical split:

- search/refinement budget
- reserved validation budget

Do not consume reserved validation budget early without explicit justification.

## Five-Hour Benchmark Protocol

When explicitly running 5-hour benchmark mode:

- 4h search/refinement
- 1h final validation reserve

Policy requirements:

- same admissible NC list for all methods
- same variable bounds
- same reporting metrics
- same final high-fidelity check

## Recommended Success Criteria

A run is strong if it delivers:

- feasible candidate meeting project targets
- high validated productivity
- evidence-backed comparisons to alternatives and recent failures
- reproducible metadata and solver settings

## Minimal Evidence For Strong Final Claim

Minimum claim package:

- final candidate metrics from high-fidelity run
- at least two meaningful competitor comparisons
- explicit failure-mode discussion for nearby rejected regions
- sensitivity or perturbation evidence around final point

## Bottom Line

Treat this as constrained, evidence-driven optimization under finite resources:

- find feasible region first
- optimize inside it
- validate rigorously
- claim only what data supports
