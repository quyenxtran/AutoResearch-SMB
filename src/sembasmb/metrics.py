from typing import Dict, List

from pyomo.environ import value

from .config import SMBInputs


def compute_outlet_averages(m, inputs: SMBInputs) -> Dict[str, List[float]]:
    col_ex = inputs.nc[0]
    col_raff = inputs.nc[0] + inputs.nc[1] + inputs.nc[2]
    t_points = list(m.t)

    ce = []
    cr = []
    for j in range(inputs.ncomp):
        ce_val = sum(value(m.C[col_ex, j + 1, t, 1.0]) for t in t_points) / len(t_points)
        cr_val = sum(value(m.C[col_raff, j + 1, t, 1.0]) for t in t_points) / len(t_points)
        ce.append(ce_val)
        cr.append(cr_val)

    return {'CE': ce, 'CR': cr}


def compute_purity_recovery(m, inputs: SMBInputs, outlets: Dict[str, List[float]]) -> Dict[str, float]:
    ce = outlets['CE']
    cr = outlets['CR']
    acid_idx = range(inputs.ncomp - 2)

    ce_acid = sum(ce[i] for i in acid_idx)
    cf_acid = sum(inputs.dict_CF[i + 1] for i in acid_idx)
    q_feed = value(m.UF)
    q_ex = value(m.UE)
    q_raff = value(m.UR)

    purity_ex_meoh_free = ce_acid / (sum(ce) - ce[-1])
    purity_ex_overall = ce_acid / sum(ce)

    # Keep recovery definitions consistent with optimization constraints:
    # RecoveryExGA / RecoveryExMA in smb_optimization.py use CE * UE / (CF * UF).
    recovery_ex = ce_acid * q_ex / (cf_acid * q_feed)
    recovery_raff = sum(cr[i] for i in acid_idx) * q_raff / (cf_acid * q_feed)

    q_ex_ml_min = q_ex * inputs.area * inputs.eb
    productivity_ex_ga_ma = ce_acid * q_ex_ml_min
    frec = (value(m.U[1]) - value(m.UD)) * inputs.area * inputs.eb

    per_comp_recovery = {}
    for i in acid_idx:
        comp_name = inputs.comps[i]
        cf_i = inputs.dict_CF[i + 1]
        recovery_ex_i = ce[i] * q_ex / (cf_i * q_feed)
        recovery_raff_i = cr[i] * q_raff / (cf_i * q_feed)
        per_comp_recovery[f"recovery_ex_{comp_name}"] = recovery_ex_i
        per_comp_recovery[f"recovery_raff_{comp_name}"] = recovery_raff_i
        per_comp_recovery[f"recovery_balance_{comp_name}"] = recovery_ex_i + recovery_raff_i

    return {
        'purity_ex_meoh_free': purity_ex_meoh_free,
        'purity_ex_overall': purity_ex_overall,
        'recovery_ex': recovery_ex,
        'recovery_raff': recovery_raff,
        'recovery_balance_acid': recovery_ex + recovery_raff,
        'productivity_ex_ga_ma': productivity_ex_ga_ma,
        'Frec': frec,
        **per_comp_recovery,
    }
