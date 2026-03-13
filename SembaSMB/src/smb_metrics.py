from typing import Dict, List

from pyomo.environ import value

from .smb_config import SMBInputs


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

    purity_ex_meoh_free = ce_acid / (sum(ce) - ce[-1])
    purity_ex_overall = ce_acid / sum(ce)

    recovery_ex = ce_acid * value(m.UE) / (cf_acid * value(m.UF))
    recovery_raff = sum(cr[i] for i in acid_idx) * value(m.UR) / (cf_acid * value(m.UF))

    q_ex = value(m.UE) * inputs.area * inputs.eb
    productivity_ex_ga_ma = ce_acid * q_ex
    frec = (value(m.U[1]) - value(m.UD)) * inputs.area * inputs.eb

    per_comp_recovery = {}
    for i in acid_idx:
        comp_name = inputs.comps[i]
        cf_i = inputs.dict_CF[i + 1]
        per_comp_recovery[f"recovery_ex_{comp_name}"] = ce[i] * value(m.UE) / (cf_i * value(m.UF))
        per_comp_recovery[f"recovery_raff_{comp_name}"] = cr[i] * value(m.UR) / (cf_i * value(m.UF))

    return {
        'purity_ex_meoh_free': purity_ex_meoh_free,
        'purity_ex_overall': purity_ex_overall,
        'recovery_ex': recovery_ex,
        'recovery_raff': recovery_raff,
        'productivity_ex_ga_ma': productivity_ex_ga_ma,
        'Frec': frec,
        **per_comp_recovery,
    }
