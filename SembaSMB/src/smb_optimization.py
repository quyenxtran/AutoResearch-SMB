from pyomo.environ import Constraint, Objective, Var, maximize

from .smb_config import SMBInputs


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
    ffeed_bounds=(1.0, 3.0),
    fdes_bounds=None,
    fex_fixed=None,
    f1_min: float = 1.5,
):
    m.UF.free()
    m.UD.free()
    m.UE.free()
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

    if fex_fixed is not None:
        ue_val = fex_fixed / (inputs.area * inputs.eb)
        m.UE.fix(ue_val)

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

    if f1_min is not None:
        u1_lb = f1_min / (inputs.area * inputs.eb)
        m.F1Min = Constraint(expr=m.U[1] >= u1_lb)

    m.obj = Objective(expr=ce_acid * m.UE * inputs.area * inputs.eb, sense=maximize)

    return m
