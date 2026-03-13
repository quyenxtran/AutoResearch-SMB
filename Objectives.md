# AutoResearch-SMB Objective

## Mission

Use the existing `AutoResearch-SMB/SembaSMB` model as the reference SMB implementation. The agent must learn the SMB formulation from the local code and notebook, then build, run, test, troubleshoot, and optimize the process for the Kraton-feed case.

The interesting part of this study is not only the final optimum, but also whether an agentic system can reason about SMB design choices, choose a good section layout and fidelity path, and validate its decisions with simulation efficiently.

The reference notebook is:

- `AutoResearch-SMB/SembaSMB/SMB_KratonFeed_updated_config copy 6.ipynb`

The reference source modules are:

- `AutoResearch-SMB/SembaSMB/src/smb_config.py`
- `AutoResearch-SMB/SembaSMB/src/smb_model.py`
- `AutoResearch-SMB/SembaSMB/src/smb_discretization.py`
- `AutoResearch-SMB/SembaSMB/src/smb_optimization.py`
- `AutoResearch-SMB/SembaSMB/src/smb_metrics.py`
- `AutoResearch-SMB/SembaSMB/src/smb_solver.py`

Do not treat this as a blank-sheet problem. Reuse the local SembaSMB physics and only change the optimization workflow, bounds, fidelity, and search strategy as needed.

## Optimization Goal

Maximize extract productivity of the organic acids, defined consistently with the current code as:

- `productivity_ex_ga_ma = (CE_GA + CE_MA) * UE * area * eb`

subject to all hard process constraints below.

## Components and Basis

Use this component order everywhere:

1. `GA`
2. `MA`
3. `Water`
4. `MeOH`

Use the Kraton-feed composition from the reference notebook:

- Feed mass fractions `wt0 = (0.003, 0.004, 0.990, 0.003)`
- Component densities `rho = (1.5, 1.6, 1.0, 0.79)`

Interpretation of feed mass fractions:

- `GA = 0.3 wt%`
- `MA = 0.4 wt%`
- `Water = 99.0 wt%`
- `MeOH = 0.3 wt%`

The model's internal feed concentration vector is computed exactly as in `build_inputs()`:

- `CF = wt0 / sum(wt0 / rho)`
- Approximate values on the code's internal concentration basis:
  - `GA = 0.003005`
  - `MA = 0.004007`
  - `Water = 0.991688`
  - `MeOH = 0.003005`

## Desorbent Composition

Use pure methanol as the desorbent.

- Desorbent mass fractions: `(0.0, 0.0, 0.0, 1.0)`
- In the current code, the internal desorbent concentration vector is:
  - `CD = (0.0, 0.0, 0.0, 0.79)`

Do not add water, GA, or MA to the desorbent unless explicitly instructed later.

## SMB Configuration

Use the Kraton-feed notebook configuration as the default reference point and default high-fidelity validation target:

- total physical columns fixed at `8`
- reference section layout `nc = (1, 2, 3, 2)`
- `nfex = 10`
- `nfet = 5`
- `ncp = 2`
- `L = 20.0`
- `d = 1.0`
- `eb = 0.44`
- `ep = 0.66`
- `Pe = 1000.0`
- `isoth = "MLL"`
- `xscheme = "CENTRAL"`

Use the current fixed transport and isotherm parameters unless a local sensitivity check proves a reason to change them:

- `kapp = (0.8, 1.22, 1.0, 0.69)`
- `qm = (0.084, 0.117, 0.02, 0.05)`
- `K = (254.0, 1208.0, 1e-3, 79.0)`
- `H = (0.61, 0.52, 1e-3, 0.06)`

For this study, the physical geometry is fixed but the section allocation may be optimized:

- `L` is fixed
- `d` is fixed
- total columns are fixed at `8`
- `nc` may vary only as an admissible integer 4-tuple satisfying `sum(nc) = 8`

Examples of admissible layouts:

- `(1, 2, 3, 2)`
- `(2, 2, 2, 2)`

Do not treat this as a variable-total-column design problem.

## Input Parameter Inventory

Treat the SMB problem inputs as four groups.

## Most Important Inputs

The scientists should treat the following as the highest-priority inputs for this problem.

### 1. Input and output flowrates

These are the most important operating inputs because they directly control SMB separation behavior and productivity:

- `F1`
- `Fdes`
- `Fex`
- `Ffeed`
- `Fraf`
- `tstep`

For this optimization problem, the primary operating focus should be on:

- inlet and outlet flowrates
- switching time

### 2. Column configuration

These are the most important structural inputs because they define the physical SMB layout and numerical fidelity target:

- `nc`
- total number of columns
- `L`
- `d`
- `eb`
- `ep`
- `nfex`
- `nfet`
- `ncp`

Unless a study is explicitly about sensitivity to chemistry or isotherm assumptions, the scientists should treat flowrates and column configuration as the first-order inputs and treat the remaining model parameters as supporting fixed context.

For this benchmark, interpret "column configuration" carefully:

- fixed physical geometry: total columns `= 8`, fixed `L`, fixed `d`
- variable discrete section allocation: `nc`

### 1. Fixed experiment inputs

These define the physical SMB model for this Kraton-feed case and should be treated as fixed unless a documented study explicitly changes them:

- `comps = ('GA', 'MA', 'Water', 'MeOH')`
- `wt0 = (0.003, 0.004, 0.990, 0.003)`
- `rho = (1.5, 1.6, 1.0, 0.79)`
- desorbent composition `CD = (0.0, 0.0, 0.0, 0.79)`
- total physical columns `= 8`
- `L = 20.0`
- `d = 1.0`
- `eb = 0.44`
- `ep = 0.66`
- `Pe = 1000.0`
- `isoth = "MLL"`
- `kapp = (0.8, 1.22, 1.0, 0.69)`
- `qm = (0.084, 0.117, 0.02, 0.05)`
- `K = (254.0, 1208.0, 1e-3, 79.0)`
- `H = (0.61, 0.52, 1e-3, 0.06)`

### 2. Numerical and fidelity inputs

These control discretization and numerical fidelity rather than the underlying process chemistry:

- `nfex`
- `nfet`
- `ncp`
- `xscheme`
- solver executable
- linear solver choice
- solver option profile

These are valid inputs to the computational experiment, but they are not physical decision variables of the SMB process itself.

### 3. Operating-point inputs

These define the SMB operating point. In code terms they correspond to `FlowRates` and related velocity inputs:

- `F1`
- `Fdes`
- `Fex`
- `Ffeed`
- `Fraf`
- `tstep`
- optional direct velocity inputs:
  - `u_f`
  - `u_d`
  - `u_e`
  - `u_r`

For this problem, the scientists should think of the main operating-point inputs as:

- `F1`
- `Fdes`
- `Fex`
- `Ffeed`
- `tstep`

and treat `Fraf` as flow-consistent and derived unless a carefully documented reason exists to parameterize it differently.

### 3A. Discrete structural inputs

These are discrete design choices over the fixed 8-column hardware:

- `nc`

For this study, `nc` is allowed to vary like a design variable, but only within an admissible library of integer 4-tuples satisfying:

- `sum(nc) = 8`
- each SMB zone must remain physically meaningful

### 4. User-fixed optimization requirements

These are not free design variables. They are fixed requirements of the problem statement:

- `0 < Ffeed <= 2.5 mL/min`
- all pump-relevant flows `<= 2.5 mL/min`
- total physical columns fixed at `8`
- `purity_ex_meoh_free >= 0.90`
- `recovery_ex_GA >= 0.90`
- `recovery_ex_MA >= 0.90`

## Decision Variables vs Derived Quantities

Unless the scientists justify a different formulation, the preferred free operating decision variables are:

- `nc`
- `Ffeed`
- `F1`
- `Fdes`
- `Fex`
- `tstep`

The following should normally be treated as derived quantities, not independent decisions:

- `F2 = F1 - Fex`
- `F3 = F2 + Ffeed`
- `F4 = F1 - Fdes`
- `Fraf = Ffeed + Fdes - Fex`
- `area = pi * d^2 / 4`
- `u_f`, `u_d`, `u_e`, `u_r`
- `dict_CF`
- `dict_CD`
- `dict_U`
- `dict_Dax`

In other words:

- `nc` is a discrete structural decision on fixed hardware
- `Ffeed`, `F1`, `Fdes`, `Fex`, and `tstep` are the main continuous operating decisions
- `Fraf` and internal section flows are derived from the model equations

## Initial Guesses vs True Inputs

The default values in `SMBConfig` such as:

- `F1_init`
- `Fdes_init`
- `Fex_init`
- `Ffeed_init`
- `Fraf_init`
- `tstep_init`

should be treated as initialization values or notebook defaults, not as mandatory settings for this optimization problem.

## Flow Variables and Physical Meaning

Treat the following as the main flow variables:

- `F1`: section-1 internal circulation flow
- `Fdes`: desorbent flow
- `Fex`: extract flow
- `Ffeed`: feed flow
- `Fraf`: raffinate flow
- `tstep`: switching time

Use the flow relationships already encoded in the model:

- `F2 = F1 - Fex`
- `F3 = F2 + Ffeed`
- `F4 = F1 - Fdes`
- `Fraf = F3 - F4 = Ffeed + Fdes - Fex`

Do not treat `Fraf` as an independent design variable if that would violate the model flow equations. It is a derived flow that still must satisfy physical bounds.

## Hard Operating Constraints

### User-imposed constraints

- No SMB pump may run above `2.5 mL/min`
- Keep the total number of physical columns fixed at `8`

Treat feed as a free operating variable subject to the same pump cap:

- `0 < Ffeed <= 2.5`

If `nc` is varied, enforce:

- `sum(nc) = 8`
- `nc` must be an admissible integer section allocation

Enforce the `2.5 mL/min` cap on every commanded or derived pump-relevant stream:

- `F1 <= 2.5`
- `Fdes <= 2.5`
- `Fex <= 2.5`
- `Ffeed <= 2.5`
- `Fraf <= 2.5`

Also enforce positivity and physically consistent flow splits:

- `F1 > 0`
- `Fdes > 0`
- `Fex > 0`
- `Fraf > 0`
- `F2 = F1 - Fex > 0`
- `F4 = F1 - Fdes > 0`

### Product-quality constraints

Use the extract-purity definition already implemented by the code:

- `purity_ex_meoh_free = (CE_GA + CE_MA) / (CE_GA + CE_MA + CE_Water)`

The target is:

- `purity_ex_meoh_free >= 0.90`

Use per-component recovery-to-extract constraints:

- `recovery_ex_GA >= 0.90`
- `recovery_ex_MA >= 0.90`

Also retain the reference process-quality safeguards unless a documented study proves they must change:

- `water_max_ex_wt <= 0.05`
- `water_max_zone1_entry_wt <= 0.01`
- `meoh_max_raff_wt <= 0.10`

### Time and search bounds

Use these as the default starting bounds for optimization unless the agent documents a better justified range:

- `tstep in [4.0, 8.0]`
- `Ffeed in (0.0, 2.5]`
- `F1 in [1.5, 2.5]`
- `Fdes in (0.0, 2.5]`
- `Fex in (0.0, 2.5]`

## Required Workflow

The agent must not jump directly to a full optimization run. It must reason, test, and validate its choices using the local SMB model.

Minimum required behavior:

1. Reproduce a local baseline solve from the reference SembaSMB notebook logic.
2. Verify solver availability and convergence behavior.
3. Confirm the model reproduces the same metrics as the local notebook on the same inputs.
4. Choose and justify its own search strategy for:
   - discrete layout choice `nc`
   - continuous operating variables
   - fidelity escalation
   - validation of final candidates
5. Distinguish exploratory results from validated results.
6. Re-check final candidates at the highest chosen fidelity before reporting a result.

If the constraints appear infeasible, the agent must say so explicitly and provide:

- the best near-feasible point found
- the violated constraints
- the amount of violation
- whether the failure appears to come from model physics, bounds, or numerical issues

## Final Deliverables

The final result must include:

- the final chosen `nc`
- the final chosen fidelity level
- the final optimized flowrates
- the final `tstep`
- `purity_ex_meoh_free`
- `recovery_ex_GA`
- `recovery_ex_MA`
- `recovery_ex`
- `productivity_ex_ga_ma`
- `Fraf`
- solver termination status
- a short explanation of why the solution is considered trustworthy
