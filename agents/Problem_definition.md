# Problem Definition

## Core Question

We want to solve the SMB productivity optimization problem under hard product-quality constraints:

- maximize SMB productivity of the organic acids in the extract
- require extract purity `> 90%`
- require recovery to extract `> 90%`

for the Kraton-feed case defined in `Objectives.md`, using the local `SembaSMB` model as the physical reference.

The more interesting research question is not only "what is the best operating point?" but also:

- how should the problem be searched efficiently
- how much model fidelity is needed at each stage
- how much confidence can we have that the reported solution is close to a global optimum rather than just a local optimum

## Optimization Problem

### Objective

Maximize extract productivity of the organic acids, consistent with the local code:

- `productivity_ex_ga_ma = (CE_GA + CE_MA) * UE * area * eb`

### Hard constraints

- `purity_ex_meoh_free >= 0.90`
- `recovery_ex_GA >= 0.90`
- `recovery_ex_MA >= 0.90`
- `0.5 <= Ffeed <= 2.5 mL/min`
- external pump-limited streams `Fdes`, `Fex`, `Ffeed`, and `Fraf` must stay `<= 2.5 mL/min`
- internal circulation flow `F1` may vary up to `5.0 mL/min`
- physically consistent internal flow splits

### Most important inputs

The most important problem inputs are:

- inlet and outlet flowrates:
  - `F1`
  - `Fdes`
  - `Fex`
  - `Ffeed`
  - `Fraf`
  - `tstep`
- column configuration:
  - `nc`
  - `L`
  - `d`
  - `eb`
  - `ep`
  - `nfex`
  - `nfet`
  - `ncp`

For the Kraton-feed benchmark, the physical SMB hardware should be treated as fixed rather than redesignable:

- `L = 20.0`
- `d = 1.0`
- total number of physical columns is fixed at `8`
- the section allocation `nc` may vary, but only as an admissible integer partition of `8` columns across the four SMB zones
- `eb`, `ep`, and the adsorption model remain fixed unless the study is explicitly reformulated as a hardware or model-uncertainty study

Examples of admissible layouts include:

- `(1, 2, 3, 2)`
- `(2, 2, 2, 2)`

Non-admissible layouts are those that violate the fixed-hardware rule, for example any `nc` tuple whose total is not `8`.

For this problem, the most important free operating decisions are expected to be:

- `Ffeed`
- `F1`
- `Fdes`
- `Fex`
- `tstep`

with `Fraf` treated as flow-consistent and derived unless there is a strong reason to reformulate.

## What Kind of Optimization Problem Is This?

Strictly speaking, the current implementation is not a simple algebraic optimization problem. It is closer to:

- a nonlinear dynamic optimization problem
- built from a discretized DAE/PDE SMB model
- with nonconvex adsorption equilibrium terms
- with nonlinear recovery and purity constraints

In practice, after discretization with Pyomo DAE, the solve becomes a large sparse nonlinear program. If integer design decisions are introduced later, such as discrete column configurations or discrete fidelity choices, then it becomes a mixed-integer nonlinear optimization problem.

So the practical challenge is:

- nonconvexity
- possible infeasibility
- sensitivity to initialization
- high solve cost at high fidelity

This means that "global optimum" is a strong claim and should not be used casually.

### When Is It Really A MINLP?

This point matters for the benchmark design.

If the section configuration `nc` is fixed, then the current problem is not truly a MINLP. It is a large nonconvex NLP built from a discretized dynamic SMB model.

If `nc` is allowed to vary over a discrete set of admissible section layouts on the same 8-column SMB hardware, then the problem becomes mixed-discrete:

- discrete design choice: `nc`
- continuous operating choices: `Ffeed`, `F1`, `Fdes`, `Fex`, `tstep`, and any other continuous flow variables

In that formulation, `nc` is exactly the ingredient that makes the problem a practical MINLP.

More precisely:

- fixed geometry: the hardware still has `8` physical columns, fixed `L`, and fixed `d`
- variable discrete structure: the 8 columns are reassigned among the four SMB zones through `nc`

So there are really two benchmark modes:

- fixed-`nc` benchmark: compare continuous-search methods on one hardware layout
- variable-`nc` benchmark: compare discrete-plus-continuous search methods on a true MINLP-style problem with fixed total hardware

## Two Ways To Approach The Problem

## 1. Direct optimization approach

This means treating the SMB problem as a constrained nonlinear optimization problem and solving it directly with a chosen formulation and solver.

### Advantages

- conceptually clean
- easier to benchmark
- easier to compare one solver/profile against another
- simpler experimental design

### Risks

- may converge to a local optimum
- may fail because the target region is infeasible
- may be very sensitive to initialization
- high-fidelity solves may be too expensive to use naively for global exploration

### Best use case

This is the right approach when:

- the feasible region is already reasonably understood
- the number of free operating decisions is small
- we want a strong local optimum near a well-chosen starting region

## 2. Agentic optimization approach

This means giving the scientists autonomy to choose:

- which solver resource to use
- which fidelity to use
- which variables to search first
- when to screen broadly versus refine locally
- when to challenge a candidate as likely local, infeasible, or numerically weak

### Advantages

- better use of compute budget
- can adapt search strategy to observed model behavior
- can use low fidelity for scouting and high fidelity for confirmation
- can combine scientific reasoning with solver evidence

### Risks

- harder to benchmark cleanly
- more degrees of freedom in the search process
- can become overly heuristic if not disciplined
- may give false confidence if it is not forced to validate final candidates carefully

### Best use case

This is the right approach when:

- the search space is expensive to explore
- fidelity matters
- the problem may be infeasible or strongly nonconvex
- we want a system that allocates compute intelligently rather than solving everything at maximum cost

## Can We Confirm A Global Optimum?

Not in a strong mathematical sense with the current workflow unless we add a much more rigorous global-optimization framework.

With the current `Pyomo + Ipopt` style workflow, what we can realistically aim for is:

- a well-tested high-quality feasible solution
- evidence that nearby alternatives are worse
- evidence that multiple starting points or search paths lead to the same region
- evidence that the result survives stricter fidelity and stricter solver settings

So the realistic research target should be:

- not "prove the global optimum"
- but "build the strongest possible evidence that the reported optimum is globally competitive under the available model and compute budget"

## What A Smart Agentic System Should Actually Do

A smart system should not just "run optimization." It should decide:

- whether the problem appears feasible under the current constraints
- whether low fidelity is enough for screening
- when a candidate deserves high-fidelity confirmation
- whether a solver failure is a numerical issue or a physical infeasibility signal
- when to spend compute on wider search versus tighter local refinement

The key idea is:

- use cheap computation to identify structure
- use expensive computation only where it changes the scientific conclusion

## Practical Experimental Framing

The problem can be framed as two nested questions.

### Question A: process optimum

Given a fixed physical SMB model and fixed quality constraints:

- what operating point gives the highest productivity?

### Question B: optimization-system design

Given limited compute and multiple solver/fidelity choices:

- what search strategy most reliably finds the best high-confidence solution?

Question A is the chemistry/process problem.

Question B is the agentic optimization problem.

Both matter, but they should not be confused.

## How To Compare Direct MINLP Versus Agent-Assisted Optimization

If the cost of intelligence is assumed to be zero, then the comparison should not reward or penalize the LLM itself. It should compare how effectively each method uses simulation and optimization effort to deliver a strong final operating point.

The comparison should be done under the same fixed physical setup:

- same Kraton-feed composition
- same fixed SMB geometry
- same admissible `nc` library with total columns fixed at `8`
- same hard purity, recovery, feed, and pump constraints
- same final high-fidelity validation model

Under those rules, the most useful comparison metrics are the following.

### Benchmark objective

Use a fixed evaluator with this lexicographic benchmark objective:

- first priority: feasibility
- second priority: validated high-fidelity productivity

More explicitly:

- if a candidate is feasible at final validation, set `J_validated = productivity_ex_ga_ma`
- if a candidate is infeasible at final validation, it loses to any feasible candidate
- among infeasible candidates, rank by smaller normalized total constraint violation

This keeps the scientific objective simple while avoiding arbitrary penalty tuning.

### Primary performance metrics

- best feasible high-fidelity productivity found
- feasibility of the reported final point
- minimum constraint slack at the final point:
  - `purity_ex_meoh_free - 0.90`
  - `recovery_ex_GA - 0.90`
  - `recovery_ex_MA - 0.90`

These tell us whether one method actually found a better process solution and how safely it satisfies the quality constraints.

### Search-efficiency metrics

- number of SMB solves required to reach the final reported solution
- number of high-fidelity solves required
- wall time spent on SMB simulation and numerical optimization only
- CPU-hours consumed by SMB solves
- time or solve count to first feasible point
- time or solve count to the best-known solution

Because intelligence cost is assumed zero, these are better fairness metrics than token counts or LLM latency.

### Robustness metrics

- repeatability of the final solution across multiple restarts or seeds
- fraction of runs that reach a feasible solution
- spread of final productivity across repeated trials
- sensitivity of the final solution to initialization
- sensitivity of the final answer to solver profile changes

These distinguish a genuinely strong method from one that gets lucky once.

### Fidelity-management metrics

- agreement between low-, medium-, and high-fidelity rankings of the top candidates
- number of false positives from low fidelity that fail at high fidelity
- number of false negatives where low fidelity wrongly rejects candidates that later validate well
- fraction of total compute spent at each fidelity level

These are especially important for the agent-assisted system, because "smartness" should show up as better fidelity allocation rather than just more search.

## Fixed-Budget Rule

To compare direct MINLP versus agent-driven search fairly, both methods should receive the same SMB numerical budget.

Do not budget by LLM tokens or reasoning time. Budget only the SMB numerical work:

- NLP and MINLP solve time
- simulation time
- validation time

The cleanest rule is:

- both methods receive the same maximum SMB-only wall time
- both methods receive the same maximum SMB-only CPU-hours
- both methods must reserve part of the budget for final high-fidelity validation

In practice, wall time should be set long enough that the direct MINLP baseline can complete at least one serious run instead of timing out trivially.

## Five-Hour Benchmark Protocol

Use a counted SMB numerical budget of `5.0` hours for each method.

This budget is intended to compare:

- the direct MINLP-style baseline
- the agent-assisted search system

Only SMB numerical work counts against this budget.

### Counted work

Count these against the `5.0` hours:

- model build plus nonlinear solve for candidate screening
- layout optimization solves
- flow-screen or feasibility-screen solves
- final validation solves
- repeated confirmation solves used to accept or reject a candidate

Do not count these against the `5.0` hours:

- solver installation
- one-time environment setup
- one-time smoke tests such as `solver-check`
- editing prompts, notebooks, or code

### Budget split

Use the same split for both methods:

- `4.0` hours for search and refinement
- `1.0` hour reserved for final validation

The reserved validation hour should not be spent during early exploration unless the method documents a reason.

### Direct baseline under the 5-hour rule

For the current `Pyomo + Ipopt` workflow, the direct baseline should be treated as a practical MINLP-style benchmark:

- admissible `nc` enumeration
- continuous NLP optimization within each selected layout
- notebook-seeded multistart for the continuous variables

The direct baseline should follow a fixed policy:

1. use the same admissible `nc` library as the agent benchmark
2. spend the `4.0` search hours on medium-fidelity search and refinement
3. use the `1.0` reserved hour on high-fidelity validation of the best incumbent layouts
4. do not adapt the search logic mid-run except through its predefined solver fallback policy

### Agent-assisted baseline under the 5-hour rule

The agent-assisted system receives the same counted `5.0` hours, but it may choose:

- which admissible `nc` layouts to screen first
- which fidelity to use at each stage
- when to stop exploring a weak region
- when to spend the reserved validation hour

The agent is still bound by:

- the same admissible `nc` library
- the same flow and time bounds
- the same hard purity and recovery constraints
- the same final high-fidelity validation model
- the same total counted SMB budget

### Recommended benchmark sequence

For each method:

1. start with the same admissible `nc` library and the same hard constraints
2. explore and refine candidates under the `4.0` hour search budget
3. log every SMB numerical call in the same ledger format
4. transition to final validation before the `1.0` reserved hour is exhausted
5. report the best validated candidate obtained within the total counted `5.0` hours

### Required outputs of the 5-hour benchmark

For each method, report:

- best validated `J_validated`
- final `nc`
- final `Ffeed`, `F1`, `Fdes`, `Fex`, `tstep`
- final purity and recoveries
- time to first feasible candidate, if any
- total SMB-only wall time used
- total SMB-only CPU-hours used
- total solve count
- high-fidelity solve count
- whether the final point was validated inside the reserved hour

### Meaning of the comparison

Under this protocol, the fair question is:

- with the same counted `5.0` SMB hours, which method reaches the best validated feasible solution, and how quickly does it get there?

## How To Account For Agent Compute Fairly

The agent will usually run many NLPs at different fidelities, so the fair accounting unit should be SMB numerical cost, not the number of agent decisions.

Use a run ledger in which every SMB numerical call records:

- method label
- `nc`
- `Ffeed`, `F1`, `Fdes`, `Fex`, `tstep`
- fidelity level
- solver profile
- start time
- end time
- wall time
- CPU count used
- CPU-hours used
- solver status
- whether the call was exploratory or final validation

Then define:

- `total_wall_time = sum(wall time of all SMB numerical calls)`
- `total_cpu_hours = sum(wall time * allocated CPU count over all SMB numerical calls)`
- `high_fidelity_cost = sum(cost of validation-tier calls)`

For parallel agent runs, the fair quantity is usually CPU-hours, because wall time alone can hide the fact that the agent used many CPUs at once.

So the recommended reporting is:

- SMB-only wall time
- SMB-only CPU-hours
- total solve count
- high-fidelity solve count

If the agent uses multiple fidelities, also report normalized cost units:

- choose the average cost of one high-fidelity validation solve as `1.0`
- express each low- or medium-fidelity call as a fraction of that cost
- sum those normalized costs across the whole run

This makes heterogeneous search paths directly comparable.

### Confidence metrics

- whether multiple search paths converge to the same region of the design space
- local stability of the final point under small perturbations in `F1`, `Fdes`, `Fex`, and `tstep`
- for the variable-`nc` case, whether multiple search paths identify the same section layout
- whether the final candidate survives stricter solver tolerances and validation settings
- critic agreement that the final point is both physically credible and numerically credible

These do not prove global optimality, but they do measure how much evidence supports the final claim.

## The Real Research Debate

The central debate is not just "MINLP solver versus LLM." It is:

- can explicit scientific reasoning choose the right discrete layout, fidelity, and search order faster than a direct optimization pipeline
- and can that reasoning reach the same or a better best-known validated solution with fewer expensive SMB solves

That is the persuasive version of the question.

If `nc` is part of the decision space, the agent-assisted system may have a real advantage because it can:

- reject weak `nc` layouts early using lower-cost evidence
- avoid spending high-fidelity solves on clearly poor regions
- switch between broad discrete exploration and local continuous refinement

A direct MINLP approach may have a real advantage when:

- the admissible `nc` set is small and well defined
- the solver formulation is tight and stable
- the global search machinery is strong enough to exploit the structure directly

So this should be treated as an empirical question, not an assumption.

## How To Make The Case Persuasively

The strongest argument is not a narrative argument. It is an experimental one.

### 1. Build a fair reference benchmark

For every method:

- use the same admissible set of `nc` layouts, each satisfying `sum(nc) = 8`
- use the same continuous variable bounds
- use the same hard purity, recovery, feed, and pump constraints
- use the same final high-fidelity validation model
- use the same stopping rule or the same SMB-only compute budget
- reserve the same fraction of budget for final validation

Without this, the comparison is not persuasive.

### 2. Distinguish proof from evidence

For the full high-fidelity SMB model, a formal proof of global optimality is unlikely.

So use two standards:

- tractable benchmark proof:
  - on a reduced problem, a coarse-fidelity model, or a restricted `nc` library, use exhaustive enumeration or a deterministic global method to establish the true optimum if possible
- full-model evidence:
  - on the real high-fidelity problem, report the best validated solution and the evidence that it is globally competitive

This lets you say:

- "we proved optimality on the reduced benchmark"
- "we showed the agent transfers that advantage to the real problem"

That is much more persuasive than claiming absolute proof on the full model.

### 3. Use regret against the best-known validated solution

For each method, report:

- best-known validated objective value found
- regret relative to the best validated value across all methods
- time-to-best-known
- solve-count-to-best-known
- CPU-hours-to-best-known
- probability of reaching the best-known region over repeated trials

This directly answers whether reasoning gets to the global basin faster.

### 4. Separate discrete-search skill from continuous-refinement skill

For the variable-`nc` benchmark, report:

- whether the method identifies the best `nc` layout
- how many layouts it had to evaluate before reaching that layout
- whether the chosen layout satisfies the fixed-hardware rule `sum(nc) = 8`
- how much effort it spends refining a good layout once found

This is important because a system can be strong at continuous local refinement but weak at choosing the right discrete structure, or vice versa.

### 5. Show robustness, not just one win

To make the case convincingly, do repeated trials with different:

- initial continuous guesses
- discrete search orderings
- solver profiles
- fidelity schedules

Then report:

- success rate
- distribution of final objective values
- distribution of time-to-feasible
- distribution of time-to-best-known
- distribution of CPU-hours-to-best-known

If the agent-assisted method wins once, that is anecdotal. If it wins repeatedly, that is evidence.

## A Realistic Claim You Can Defend

The most defensible claim is not:

- "the agent proved the global optimum"

It is:

- "for the SMB problem where `nc` introduces discrete structure, the agent-assisted system reached the best-known validated solution faster and more reliably than the direct MINLP baseline under the same simulation budget"

That claim is strong, measurable, and scientifically defensible.

## Recommended Benchmark Summary

For a clean study, report each method with the same summary table:

- benchmark objective `J_validated`
- best validated productivity
- final purity and recoveries
- minimum constraint slack
- total SMB solve count
- total high-fidelity solve count
- SMB-only wall time
- SMB-only CPU-hours
- success rate across repeated trials
- validation status of the final point

This makes the comparison interpretable even if the underlying search logic is very different.

## Recommended Success Criteria

The study is successful if it can produce:

1. a feasible high-fidelity operating point satisfying purity and recovery constraints
2. a justified explanation of why the chosen solver and fidelity path were appropriate
3. evidence that the result is not just a fragile local artifact
4. a clear statement of whether the constraints appear feasible, marginal, or likely infeasible

## Minimal Evidence For A Strong Final Claim

Before calling a result "best found" or "near-global," the system should be able to show:

- repeated convergence from multiple starting conditions or search paths
- agreement between lower- and higher-fidelity rankings in the final region
- acceptable solver termination behavior
- no hidden violation of purity, recovery, or pump constraints
- critic approval that the result is scientifically and numerically credible

## Bottom Line

The real task is not just solving one nonlinear program.

It is designing a disciplined search-and-validation system for a nonconvex SMB optimization problem where:

- the main knobs are operating flowrates and switching time
- the physical column geometry is fixed for the Kraton-feed benchmark
- the section layout `nc` may be fixed for an NLP benchmark or varied for a true MINLP benchmark, but it must remain an admissible integer partition of `8` total columns
- the objective is productivity
- the constraints are purity and recovery
- fidelity is expensive
- and certainty about global optimality is limited

The best version of this project is therefore:

- direct SMB optimization inside a larger agentic system that chooses solver resources, search strategy, and fidelity intelligently
- while staying honest that the final result is a high-confidence optimum candidate, not a formal proof of global optimality
