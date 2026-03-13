from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeReals,
    Param,
    PositiveReals,
    RangeSet,
    Var,
)

from .smb_config import SMBConfig, SMBInputs


def build_model(config: SMBConfig, inputs: SMBInputs) -> ConcreteModel:
    m = ConcreteModel()

    m.x = ContinuousSet(bounds=(0.0, 1.0))
    m.t = ContinuousSet(bounds=(0.0, 1.0))
    m.col = RangeSet(inputs.ncols)
    m.comp = RangeSet(inputs.ncomp)
    m.acid = RangeSet(inputs.ncomp - 2)

    m.L = Param(initialize=config.L)
    m.area = Param(initialize=inputs.area)
    m.Dax = Param(m.col, initialize=inputs.dict_Dax)
    m.eb = Param(initialize=config.eb)

    m.kapp = Var(m.comp, initialize=inputs.dict_kapp, bounds=(1e-8, 100))
    m.qm = Var(m.comp, initialize=inputs.dict_qm, bounds=(1e-8, 10.0))
    m.K = Var(m.comp, initialize=inputs.dict_K, bounds=(1e-8, 1500))

    h_bounds = (1e-8, 1.0) if config.isoth.upper() == 'MLLE' else (1e-8, 1e3)
    m.H = Var(m.comp, initialize=inputs.dict_H, bounds=h_bounds)

    for i in m.comp:
        m.kapp[i].fix(inputs.dict_kapp[i])
        m.qm[i].fix(inputs.dict_qm[i])
        m.K[i].fix(inputs.dict_K[i])
        m.H[i].fix(inputs.dict_H[i])

    m.C = Var(m.col, m.comp, m.t, m.x, within=NonNegativeReals)
    m.Q = Var(m.col, m.comp, m.t, m.x)
    m.Cp = Var(m.col, m.comp, m.t, m.x)

    m.CF = Param(m.comp, initialize=inputs.dict_CF)
    m.CD = Param(m.comp, initialize=inputs.dict_CD)

    m.U = Var(m.col, within=PositiveReals, initialize=inputs.dict_U, bounds=(0.1, 12.0))

    m.UF = Var(initialize=inputs.u_f, bounds=(0.01, 10.0))
    m.UD = Var(initialize=inputs.u_d, bounds=(0.01, 10.0))
    m.UE = Var(initialize=inputs.u_e, bounds=(0.01, 10.0))
    m.UR = Var(initialize=inputs.u_r, bounds=(0.01, 10.0))

    m.Frec = Expression(expr=(m.U[1] - m.UD) * m.area * m.eb)

    m.UF.fix(inputs.u_f)
    m.UD.fix(inputs.u_d)
    m.UE.fix(inputs.u_e)

    m.tstep = Var(initialize=inputs.tstep, bounds=(0.1, 14.0))
    m.tstep.fix(inputs.tstep)

    m.dCdx = DerivativeVar(m.C, wrt=m.x)
    m.dC2dx2 = DerivativeVar(m.C, wrt=(m.x, m.x))
    m.dQdt = DerivativeVar(m.Q, wrt=m.t)
    m.dCdt = DerivativeVar(m.C, wrt=m.t)
    m.dCpdt = DerivativeVar(m.Cp, wrt=m.t)

    def MassBalanceLiquid_rule(m, i, j, k, l):
        if l == 0:
            return Constraint.Skip
        return (
            m.dCdt[i, j, k, l] / m.tstep
            + m.U[i] * m.dCdx[i, j, k, l] / m.L
            + (1 - m.eb) / m.eb * m.kapp[j] * (m.C[i, j, k, l] - m.Cp[i, j, k, l])
            - m.Dax[i] * m.dC2dx2[i, j, k, l] / m.L / m.L
            == 0
        )

    def MassBalanceSolid_rule(m, i, j, k, l):
        return (
            (m.dQdt[i, j, k, l] + config.ep * m.dCpdt[i, j, k, l]) / m.tstep
            == m.kapp[j] * (m.C[i, j, k, l] - m.Cp[i, j, k, l])
        )

    isoth = config.isoth.upper()

    def Equilibrium_rule(m, i, j, k, l):
        if isoth == 'MLLE':
            return (
                m.Q[i, j, k, l]
                == m.H[j]
                + m.qm[j]
                * m.K[j]
                * m.Cp[i, j, k, l]
                / (1 + sum(m.K[v] * m.Cp[i, v, k, l] for v in m.comp))
            )
        if isoth == 'MLL':
            return (
                m.Q[i, j, k, l]
                == m.H[j] * m.Cp[i, j, k, l]
                + m.qm[j]
                * m.K[j]
                * m.Cp[i, j, k, l]
                / (1 + sum(m.K[v] * m.Cp[i, v, k, l] for v in m.comp))
            )
        if isoth == 'L':
            return (
                m.Q[i, j, k, l]
                == m.qm[j]
                * m.K[j]
                * m.Cp[i, j, k, l]
                / (1 + sum(m.K[v] * m.Cp[i, v, k, l] for v in m.comp))
            )
        raise ValueError(f"Unknown isotherm '{isoth}'")

    m.MassBalanceLiquid = Constraint(m.col, m.comp, m.t, m.x, rule=MassBalanceLiquid_rule)
    m.MassBalanceSolid = Constraint(m.col, m.comp, m.t, m.x, rule=MassBalanceSolid_rule)
    m.Equilibrium = Constraint(m.col, m.comp, m.t, m.x, rule=Equilibrium_rule)

    def FlowCondition_rule(m, i):
        if i == 1:
            return m.U[inputs.ncols] + m.UD == m.U[i]
        if i == inputs.nc[0] + 1:
            return m.U[i - 1] == m.U[i] + m.UE
        if i == inputs.nc[0] + inputs.nc[1] + 1:
            return m.U[i - 1] + m.UF == m.U[i]
        if i == inputs.nc[0] + inputs.nc[1] + inputs.nc[2] + 1:
            return m.U[i - 1] == m.U[i] + m.UR
        return m.U[i - 1] == m.U[i]

    m.U[inputs.ncols].fix(inputs.dict_U[inputs.ncols])
    m.FlowCondition = Constraint(m.col, rule=FlowCondition_rule)

    return m
