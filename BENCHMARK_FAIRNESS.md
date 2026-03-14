# SMB Benchmark Fairness Policy

## Purpose

Define an apples-to-apples comparison rule when evaluating:

- serial SMB stage campaigns
- parallel candidate-screening campaigns

for the same SMB optimization problem.

## Core Rule

Do not compare methods by wall-clock time alone unless concurrency is identical.

Use one of these valid comparison protocols:

1. Fixed resource envelope (recommended)
2. Strict serial-only comparison

## Protocol A: Fixed Resource Envelope (Recommended)

Both methods must use the same total compute budget:

- same total CPU core-hours
- same memory and solver stack constraints
- same search space (NC library, bounds, fidelity rules, constraints)
- same final validation fidelity

Then compare:

- best feasible objective found within budget
- feasibility rate
- constraint violation statistics
- total core-hours consumed

This allows parallel candidate evaluation and is fair for campaign-level optimization.

## Protocol B: Strict Serial-Only

For algorithm-only comparison:

- force concurrency = 1 for all methods
- keep same solver options and search scope

Then wall-clock and search trajectory are directly comparable.

## Required Reporting (Every Benchmark)

Report all of the following:

- experiment ID and date
- protocol used (A or B)
- solver stack (example: `ipopt + mumps`)
- NC library and parameter bounds
- fidelity policy
- total wall time
- total CPU core-hours
- number of attempted runs
- number of feasible runs
- best feasible productivity
- best near-feasible normalized violation

## Not Fair (Avoid)

- comparing serial wall-time vs parallel wall-time without budget normalization
- changing solver stack between methods without disclosure
- changing bounds or NC library between methods
- comparing provisional infeasible points as final winners

## Recommended Default for This Project

Use Protocol A with a fixed budget per campaign (example: 12 CPU cores for 5 hours), and keep:

- same `nc_library`
- same hard constraints
- same final high-fidelity validation rule

This preserves fairness while allowing practical parallel screening across independent candidate runs.
