# SKILLS.md — SMB Physics and Operational Fundamentals

This document captures fundamental SMB physics, mass transfer principles, solver behavior, and operational knowledge for the Kraton-feed case. It describes how the system works — not which parameter values are optimal (those must be discovered by experiment and logged in `hypotheses.json`).

---

## 1. SMB Zone Functions

In a 4-zone SMB the four zones have distinct roles:

| Zone | Position | Function |
|------|----------|----------|
| Z1 | Desorbent inlet → Extract outlet | Desorption: strips strongly-adsorbed components into extract |
| Z2 | Extract outlet → Feed inlet | Purification: re-adsorbs impurities that leaked into extract side |
| Z3 | Feed inlet → Raffinate outlet | Adsorption: retains target components from feed; lets weaker-adsorbed species pass |
| Z4 | Raffinate outlet → Desorbent inlet | Regeneration: partially strips weakly-adsorbed species before returning to Z1 |

The column allocation `nc = (n1, n2, n3, n4)` distributes physical columns across zones and therefore determines the volume available for each function. Different allocations shift the balance between desorption capacity, purification efficiency, adsorption efficiency, and throughput. Whether more columns in a zone improves or degrades performance depends on the operating flows and isotherm — it must be verified by simulation.

---

## 2. Flow Mass Balance (Critical Invariant)

```
F1 = Fdes + Fex       (desorbent/extract zone balance)
F1 = Ffeed + Fraf     (feed/raffinate zone balance)
```

`Fraf` is **derived**, not an independent decision variable. Any operating point must satisfy both equations within numerical tolerance. Violations produce unphysical concentration profiles and solver failures.

Internal zone flows follow directly from these:

```
F2 = F1 - Fex
F3 = F2 + Ffeed
F4 = F1 - Fdes
```

Check: `F3 - F4 = Ffeed + Fdes - Fex = Fraf`. Any result that violates `|F1 - Fdes - Fex| / F1 > 0.01` must be rejected.

---

## 3. Equilibrium Theory and Feasibility Conditions

For **linear isotherms**, triangle theory gives analytic zone flow conditions for complete separation. For the **nonlinear MLL isotherm** used here, these conditions become concentration-dependent and cannot be evaluated analytically — but the qualitative logic holds:

- Z1 must carry both target components (flow high enough to desorb all adsorbed mass)
- Z2 must carry only the strongly-adsorbed component
- Z3 must carry only the weakly-adsorbed component
- Z4 must carry neither (flow low enough to re-adsorb remaining impurities)

If any zone fails its function, purity or recovery will be infeasible regardless of optimization.

---

## 4. MLL Isotherm — What the Parameters Mean

The Modified Langmuir (MLL) isotherm is **competitive and nonlinear**:

- `H` (Henry's constant): linear adsorption affinity at infinite dilution. Ratio H_GA/H_MA determines intrinsic selectivity.
- `qm` (saturation capacity): maximum loading on the sorbent. Larger qm = more capacity.
- `K` (Langmuir constant): nonlinearity coefficient. High K means strong curvature even at low concentrations.
- `kapp` (apparent mass transfer coefficient): controls how fast equilibrium is approached in the column.

Components compete for adsorption sites — increasing one component's concentration can desorb another. At the Kraton-feed concentrations (wt0 ≈ 0.3–0.4% for GA/MA), the system is near-dilute, so Henry's law is approximately valid but nonlinear effects still matter.

---

## 5. Switching Time and Cyclic Steady State

The switching time `tstep` controls how far concentration wave fronts move per switch period. It must be matched to zone flow rates and column volumes:

- Too short: fronts don't travel far enough — poor separation efficiency
- Too long: fronts overshoot zone boundaries — cross-contamination

The cyclic steady-state constraint (CSSC in the code) links the concentration profile at the end of one column's period to the start of the next. This is a hard periodicity constraint baked into the model — it is not optional.

`tstep` interacts with flow rates because the wave velocity in each zone depends on both the interstitial velocity and the effective isotherm slope. The appropriate range of `tstep` for a given layout and flow regime is a simulation result, not a fixed prior.

---

## 6. Multi-Fidelity Discretization

The DAE model is discretized in space (nfex finite elements, CENTRAL scheme) and time (nfet finite elements, ncp collocation points per element). Coarser grids:

- Solve faster and require less memory
- May miss fine-scale transport effects, especially near constraint boundaries
- Generally preserve rank ordering between candidates, but absolute metric values differ from high-fidelity

The fidelity ladder is: **low → medium → high**. Never jump directly from low to high — the warm-start from medium fidelity is necessary for numerical stability. The high-fidelity result is the only one suitable for final reporting.

---

## 7. Solver Status Interpretation

| Status | Meaning | Action |
|--------|---------|--------|
| `optimal` / `ok` | Converged to local optimum | Check metrics for constraint satisfaction |
| `infeasible` | IPOPT found no feasible point from this start | Try different initial point, or relax bounds temporarily |
| `solver_error` | Numerical failure | May be bad initial point, extreme flows, or model issue — not proof of physics infeasibility |
| `acceptable` | Converged to weaker tolerance | Treat as approximate; re-solve at tighter tolerance before accepting |
| `maxIterations` | Iteration limit hit | Increase `max_iter`, or check if problem is near-infeasible |

A `solver_error` means IPOPT could not find a solution from the given start — it does not mean the problem is infeasible. A different initial point, a lower-fidelity warm start, or looser tolerances may succeed.

---

## 8. Purity and Recovery — Exact Definitions

These definitions are hard-coded in `metrics.py`. Do not substitute alternatives.

- **Purity (MeOH-free)**: `(CE_GA + CE_MA) / (CE_GA + CE_MA + CE_Water)` — time-averaged extract concentrations, MeOH excluded from denominator
- **Recovery GA**: fraction of GA entering in feed that exits in extract
- **Recovery MA**: fraction of MA entering in feed that exits in extract
- **Productivity**: `(CE_GA + CE_MA) × UE × area × eb` — volumetric throughput of target acids in extract stream

The extract purity is on a MeOH-free basis because MeOH is the desorbent and its presence is expected — what matters is the acid/water separation.

---

## 9. Physical Hardware Constraints

These bounds reflect the actual experimental hardware on which results will be validated:

- `Fdes, Fex, Ffeed, Fraf ≤ 2.5 mL/min` (external pump maximum)
- `F1 ≤ 5.0 mL/min` (internal recirculation, separate higher limit)
- All flows must remain strictly positive
- Total columns fixed at 8; each zone must have ≥ 1 column

Violating these makes a result experimentally irreproducible even if the simulation converges.

---

## 10. What the Agents Should Discover, Not Be Told

The following are **not** in this document because they are experimental findings that the agents must discover and validate:

- Which `nc` layout performs best for this feed/isotherm system
- Optimal ranges for `F1`, `Fdes`, `Fex`, `Ffeed`, `tstep`
- How purity responds quantitatively to specific flow changes
- Which zone allocation (Z1, Z2, Z3, Z4 column count) maximizes productivity
- Coupling ratios between flow variables

Evidence from simulation runs should be logged in `hypotheses.json` (`simulation_results[]`) and used to update hypothesis confidence. Do not treat any specific value range as "known good" unless it appears in a validated high-fidelity run logged there.
