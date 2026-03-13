from pyomo.environ import Constraint, TransformationFactory, Var

from .smb_config import SMBConfig, SMBInputs


def apply_discretization(m, config: SMBConfig, inputs: SMBInputs):
    discretizet = TransformationFactory('dae.collocation')
    discretizet.apply_to(m, nfe=config.nfet, ncp=config.ncp, wrt=m.t, scheme='LAGRANGE-RADAU')

    discretizex = TransformationFactory('dae.finite_difference')
    discretizex.apply_to(m, nfe=config.nfex, wrt=m.x, scheme=config.xscheme)

    def C_AxialDerivativeConstraintBeginning_rule(m, i, j, k):
        return (
            m.dCdx[i, j, k, 0]
            == (-3 * m.C[i, j, k, 0] + 4 * m.C[i, j, k, m.x.at(2)] - m.C[i, j, k, m.x.at(3)])
            / 2
            / m.x.at(2)
        )

    m.C_AxialDerivativeConstraintBeginning = Constraint(m.col, m.comp, m.t, rule=C_AxialDerivativeConstraintBeginning_rule)

    def C_Axial2ndDerivativeConstraintBeginning_rule(m, i, j, k):
        return (
            m.dC2dx2[i, j, k, 0]
            == (
                2 * m.C[i, j, k, 0]
                - 5 * m.C[i, j, k, m.x.at(2)]
                + 4 * m.C[i, j, k, m.x.at(3)]
                - m.C[i, j, k, m.x.at(4)]
            )
            / (m.x.at(2)) ** 2
        )

    m.C_Axial2ndDerivativeConstraintBeginning = Constraint(m.col, m.comp, m.t, rule=C_Axial2ndDerivativeConstraintBeginning_rule)

    xscheme = config.xscheme.upper()
    if xscheme == 'CENTRAL':
        def C_AxialDerivativeConstraintEnd_rule(m, i, j, k):
            return (
                m.dCdx[i, j, k, m.x.at(-1)]
                == (3 * m.C[i, j, k, m.x.at(-1)] - 4 * m.C[i, j, k, m.x.at(-2)] + m.C[i, j, k, m.x.at(-3)])
                / 2
                / (m.x.at(-1) - m.x.at(-2))
            )

        m.C_AxialDerivativeConstraintEnd = Constraint(m.col, m.comp, m.t, rule=C_AxialDerivativeConstraintEnd_rule)

        def C_Axial2ndDerivativeConstraintEnd_rule(m, i, j, k):
            return (
                m.dC2dx2[i, j, k, m.x.at(-1)]
                == (
                    2 * m.C[i, j, k, m.x.at(-1)]
                    - 5 * m.C[i, j, k, m.x.at(-2)]
                    + 4 * m.C[i, j, k, m.x.at(-3)]
                    - m.C[i, j, k, m.x.at(-4)]
                )
                / (m.x.at(-1) - m.x.at(-2)) ** 2
            )

        m.C_Axial2ndDerivativeConstraintEnd = Constraint(m.col, m.comp, m.t, rule=C_Axial2ndDerivativeConstraintEnd_rule)
    elif xscheme == 'BACKWARD':
        def C_Axial2ndDerivativeConstraintBeginning_rule2(m, i, j, k):
            return (
                m.dC2dx2[i, j, k, m.x.at(2)]
                == (
                    2 * m.C[i, j, k, m.x.at(2)]
                    - 5 * m.C[i, j, k, m.x.at(3)]
                    + 4 * m.C[i, j, k, m.x.at(4)]
                    - m.C[i, j, k, m.x.at(5)]
                )
                / (m.x.at(2)) ** 2
            )

        m.C_Axial2ndDerivativeConstraintBeginning2 = Constraint(m.col, m.comp, m.t, rule=C_Axial2ndDerivativeConstraintBeginning_rule2)

    m.C0 = Var(m.col, m.comp, m.t)

    def MassBalance_rule(m, i, j, k):
        if k == 0:
            return Constraint.Skip
        if i == 1:
            return m.C[inputs.ncols, j, k, 1.0] * m.U[inputs.ncols] + m.CD[j] * m.UD == m.C0[i, j, k] * m.U[i]
        if i == inputs.nc[0] + inputs.nc[1] + 1:
            return m.C[i - 1, j, k, 1.0] * m.U[i - 1] + m.CF[j] * m.UF == m.C0[i, j, k] * m.U[i]
        return m.C[i - 1, j, k, 1.0] == m.C0[i, j, k]

    def BoundaryConditionC_rule(m, i, j, k):
        return m.C0[i, j, k] == m.C[i, j, k, 0] - m.Dax[i] / m.U[i] * m.dCdx[i, j, k, 0] / m.L

    m.MassBalance = Constraint(m.col, m.comp, m.t, rule=MassBalance_rule)
    m.BoundaryConditionC = Constraint(m.col, m.comp, m.t, rule=BoundaryConditionC_rule)

    def CSSC_rule(m, i, j, l):
        if i == inputs.ncols:
            return m.C[inputs.ncols, j, 0, l] == m.C[1, j, 1, l]
        return m.C[i, j, 0, l] == m.C[i + 1, j, 1, l]

    def CSSCp_rule(m, i, j, l):
        if i == inputs.ncols:
            return m.Cp[inputs.ncols, j, 0, l] == m.Cp[1, j, 1, l]
        return m.Cp[i, j, 0, l] == m.Cp[i + 1, j, 1, l]

    def CSSQ_rule(m, i, j, l):
        if i == inputs.ncols:
            return m.Q[inputs.ncols, j, 0, l] == m.Q[1, j, 1, l]
        return m.Q[i, j, 0, l] == m.Q[i + 1, j, 1, l]

    m.CSSC = Constraint(m.col, m.comp, m.x, rule=CSSC_rule)
    m.CSSCp = Constraint(m.col, m.comp, m.x, rule=CSSCp_rule)
    m.CSSQ = Constraint(m.col, m.comp, m.x, rule=CSSQ_rule)

    return m
