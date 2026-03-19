from pyomo.environ import Constraint, Objective, Var, maximize, minimize, NonNegativeReals

from .config import SMBInputs


def add_optimization(
    m,
    inputs: SMBInputs,
    purity_min: float = 0.8,
    recovery_min_ga: float = 0.8,
    recovery_min_ma: float = 0.9,
    meoh_max_raff_wt: float = 0.05,
    water_max_ex_wt: float = 0.05,
    water_max_zone1_entry_wt: float = 0.01,
    tstep_bounds=(8.0, 12.0),
    ffeed_bounds=(0.5, 2.5),
    fdes_bounds=(0.5, 2.5),
    fex_bounds=(0.5, 2.5),
    fraf_bounds=(0.5, 2.5),
    f1_bounds=(0.5, 5.0),
    fex_fixed=None,
    f1_min: float = 0.5,
    f1_max: float = 5.0,
):
    m.UF.free()
    m.UD.free()
    m.UE.free()
    m.UR.unfix()
    m.U[inputs.ncols].free()
    m.tstep.unfix()
    if tstep_bounds is not None:
        m.tstep.setlb(tstep_bounds[0])
        m.tstep.setub(tstep_bounds[1])

    if ffeed_bounds is not None:
        uf_lb = ffeed_bounds[0] / (inputs.area * inputs.eb)
        uf_ub = ffeed_bounds[1] / (inputs.area * inputs.eb)
        m.UF.setlb(uf_lb)
        m.UF.setub(uf_ub)

    if fdes_bounds is not None:
        ud_lb = fdes_bounds[0] / (inputs.area * inputs.eb)
        ud_ub = fdes_bounds[1] / (inputs.area * inputs.eb)
        m.UD.setlb(ud_lb)
        m.UD.setub(ud_ub)

    if fex_bounds is not None:
        ue_lb = fex_bounds[0] / (inputs.area * inputs.eb)
        ue_ub = fex_bounds[1] / (inputs.area * inputs.eb)
        m.UE.setlb(ue_lb)
        m.UE.setub(ue_ub)

    if fraf_bounds is not None:
        ur_lb = fraf_bounds[0] / (inputs.area * inputs.eb)
        ur_ub = fraf_bounds[1] / (inputs.area * inputs.eb)
        m.UR.setlb(ur_lb)
        m.UR.setub(ur_ub)

    if fex_fixed is not None:
        ue_val = fex_fixed / (inputs.area * inputs.eb)
        m.UE.fix(ue_val)

    # For the benchmark problem, raffinate is flow-consistent and derived:
    # q_raf = q_feed + q_des - q_ex. Enforce the same relation here instead of
    # letting UR float independently, which can create artificial feasibility
    # artifacts relative to the documented process constraints.
    m.RaffinateConsistency = Constraint(expr=m.UR == m.UF + m.UD - m.UE)

    m.CE = Var(m.comp)
    m.CR = Var(m.comp)

    def CE_rule(m, j):
        return m.CE[j] == sum(m.C[inputs.nc[0], j, k, 1.0] for k in m.t) / len(m.t)

    def CR_rule(m, j):
        return m.CR[j] == sum(m.C[inputs.nc[0] + inputs.nc[1] + inputs.nc[2], j, k, 1.0] for k in m.t) / len(m.t)

    m.CE_cons = Constraint(m.comp, rule=CE_rule)
    m.CR_cons = Constraint(m.comp, rule=CR_rule)

    try:
        ga_idx = inputs.comps.index('GA') + 1
        ma_idx = inputs.comps.index('MA') + 1
        meoh_idx = inputs.comps.index('MeOH') + 1
        water_idx = inputs.comps.index('Water') + 1
    except ValueError as exc:
        raise ValueError("inputs.comps must include 'GA', 'MA', 'MeOH', and 'Water'") from exc

    ce_acid = sum(m.CE[i] for i in m.acid)
    cf_ga = inputs.dict_CF[ga_idx]
    cf_ma = inputs.dict_CF[ma_idx]

    m.RecoveryExGA = Constraint(expr=(m.CE[ga_idx] * m.UE) / (cf_ga * m.UF) >= recovery_min_ga)
    m.RecoveryExMA = Constraint(expr=(m.CE[ma_idx] * m.UE) / (cf_ma * m.UF) >= recovery_min_ma)

    m.PurityExMeohFree = Constraint(
        expr=ce_acid >= purity_min * (sum(m.CE[i] for i in m.comp) - m.CE[meoh_idx])
    )

    col_ex = inputs.nc[0]
    col_raff = inputs.nc[0] + inputs.nc[1] + inputs.nc[2]

    def RaffMeoh_rule(m, t):
        total = sum(m.C[col_raff, j, t, 1.0] for j in m.comp)
        return m.C[col_raff, meoh_idx, t, 1.0] <= meoh_max_raff_wt * total

    m.RaffMeoh = Constraint(m.t, rule=RaffMeoh_rule)

    def ExtractWater_rule(m, t):
        total = sum(m.C[col_ex, j, t, 1.0] for j in m.comp)
        return m.C[col_ex, water_idx, t, 1.0] <= water_max_ex_wt * total

    m.ExtractWater = Constraint(m.t, rule=ExtractWater_rule)

    x_entry = m.x.at(1)

    def Zone1EntryWater_rule(m, t):
        total = sum(m.C[1, j, t, x_entry] for j in m.comp)
        return m.C[1, water_idx, t, x_entry] <= water_max_zone1_entry_wt * total

    m.Zone1EntryWater = Constraint(m.t, rule=Zone1EntryWater_rule)

    if f1_bounds is not None:
        u1_lb = f1_bounds[0] / (inputs.area * inputs.eb)
        u1_ub = f1_bounds[1] / (inputs.area * inputs.eb)
        m.U[1].setlb(u1_lb)
        m.U[1].setub(u1_ub)
    else:
        if f1_min is not None:
            u1_lb = f1_min / (inputs.area * inputs.eb)
            m.U[1].setlb(u1_lb)
        if f1_max is not None:
            u1_ub = f1_max / (inputs.area * inputs.eb)
            m.U[1].setub(u1_ub)

    m.obj = Objective(expr=ce_acid * m.UE * inputs.area * inputs.eb, sense=maximize)

    return m


def add_feasibility_objective(
    m,
    inputs: SMBInputs,
    *,
    purity_min: float = 0.60,
    recovery_min_ga: float = 0.75,
    recovery_min_ma: float = 0.75,
):
    """Replace the productivity objective with a feasibility objective.

    Adds slack variables to the hard constraints (recovery, purity) and
    minimizes total slack. After solving Phase 1, if total slack is near zero,
    the model is at a feasible point suitable for warm-starting Phase 2
    (productivity maximization).

    Call this INSTEAD of add_optimization() for Phase 1, or call it AFTER
    add_optimization() to temporarily swap the objective.
    """
    try:
        ga_idx = inputs.comps.index('GA') + 1
        ma_idx = inputs.comps.index('MA') + 1
    except ValueError as exc:
        raise ValueError("inputs.comps must include 'GA' and 'MA'") from exc

    # Add slack variables for each hard constraint
    m.slack_recovery_ga = Var(within=NonNegativeReals, initialize=0.0)
    m.slack_recovery_ma = Var(within=NonNegativeReals, initialize=0.0)
    m.slack_purity = Var(within=NonNegativeReals, initialize=0.0)

    # Relax the hard constraints by adding slack
    cf_ga = inputs.dict_CF[ga_idx]
    cf_ma = inputs.dict_CF[ma_idx]

    # Deactivate original hard constraints if they exist
    for cname in ('RecoveryExGA', 'RecoveryExMA', 'PurityExMeohFree'):
        if hasattr(m, cname):
            getattr(m, cname).deactivate()

    # Add relaxed versions
    m.RecoveryExGA_relaxed = Constraint(
        expr=(m.CE[ga_idx] * m.UE) / (cf_ga * m.UF) + m.slack_recovery_ga >= recovery_min_ga
    )
    m.RecoveryExMA_relaxed = Constraint(
        expr=(m.CE[ma_idx] * m.UE) / (cf_ma * m.UF) + m.slack_recovery_ma >= recovery_min_ma
    )

    meoh_idx = inputs.comps.index('MeOH') + 1
    ce_acid = sum(m.CE[i] for i in m.acid)
    m.PurityExMeohFree_relaxed = Constraint(
        expr=ce_acid + m.slack_purity >= purity_min * (sum(m.CE[i] for i in m.comp) - m.CE[meoh_idx])
    )

    # Deactivate productivity objective
    if hasattr(m, 'obj'):
        m.obj.deactivate()

    # Minimize total slack (= find the most feasible point)
    m.feasibility_obj = Objective(
        expr=m.slack_recovery_ga + m.slack_recovery_ma + m.slack_purity,
        sense=minimize,
    )

    return m


def restore_productivity_objective(m):
    """After Phase 1 feasibility solve, restore the productivity objective for Phase 2.

    Fixes slack variables to zero, reactivates original constraints, and
    switches back to the productivity objective.
    """
    # Fix slacks to zero — we're now demanding true feasibility
    if hasattr(m, 'slack_recovery_ga'):
        m.slack_recovery_ga.fix(0.0)
    if hasattr(m, 'slack_recovery_ma'):
        m.slack_recovery_ma.fix(0.0)
    if hasattr(m, 'slack_purity'):
        m.slack_purity.fix(0.0)

    # Deactivate relaxed constraints
    for cname in ('RecoveryExGA_relaxed', 'RecoveryExMA_relaxed', 'PurityExMeohFree_relaxed'):
        if hasattr(m, cname):
            getattr(m, cname).deactivate()

    # Reactivate original hard constraints
    for cname in ('RecoveryExGA', 'RecoveryExMA', 'PurityExMeohFree'):
        if hasattr(m, cname):
            getattr(m, cname).activate()

    # Swap objectives
    if hasattr(m, 'feasibility_obj'):
        m.feasibility_obj.deactivate()
    if hasattr(m, 'obj'):
        m.obj.activate()

    return m
