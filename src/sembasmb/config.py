from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .isotherm import get_isotherm_params


@dataclass(frozen=True)
class FlowRates:
    F1: float
    Fdes: float
    Fex: float
    Ffeed: float
    Fraf: Optional[float] = None
    tstep: Optional[float] = None
    u_f: Optional[float] = None
    u_d: Optional[float] = None
    u_e: Optional[float] = None
    u_r: Optional[float] = None
    run_name: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        data = {
            'F1': self.F1,
            'Fdes': self.Fdes,
            'Fex': self.Fex,
            'Ffeed': self.Ffeed,
        }
        if self.Fraf is not None:
            data['Fraf'] = self.Fraf
        if self.tstep is not None:
            data['tstep'] = self.tstep
        if self.u_f is not None:
            data['u_f'] = self.u_f
        if self.u_d is not None:
            data['u_d'] = self.u_d
        if self.u_e is not None:
            data['u_e'] = self.u_e
        if self.u_r is not None:
            data['u_r'] = self.u_r
        return data


@dataclass(frozen=True)
class SMBConfig:
    nc: Tuple[int, ...] = (2, 2, 2, 2)  # columns per section
    nfex: int = 10  # extraction feed points
    nfet: int = 5  # extract tank points
    ncp: int = 2  # component pairs
    comps: Tuple[str, ...] = ('GA', 'MA', 'Water', 'MeOH')  # component names
    F1_init: float = 2.2  # SMB internal flow
    Fdes_init: float = 1.2  # desorbent flow
    Fex_init: float = 0.9  # extract flow
    Ffeed_init: float = 1.3  # feed flow
    Fraf_init: float = 1.6  # raffinate flow
    tstep_init: float = 9.4  # switching time
    L: float = 20.0  # column length
    d: float = 1.0  # column diameter
    eb: float = 0.44  # bed porosity
    ep: float = 0.66  # particle porosity
    isoth: str = 'MLL'  # isotherm type
    kapp: Tuple[float, ...] = (0.8, 1.22, 1.0, 0.69)  # mass transfer coeffs
    rho: Tuple[float, ...] = (1.5, 1.6, 1.0, 0.79)  # densities
    wt0: Tuple[float, ...] = (0.0247, 0.0011, 0.9742, 0.0)  # feed wt fraction
    Pe: float = 1000.0  # Peclet number
    tscheme: str = 'BACKWARD'  # time scheme
    xscheme: str = 'CENTRAL'  # space scheme


@dataclass(frozen=True)
class SMBInputs:
    nc: Tuple[int, ...]
    nsec: int
    ncols: int
    ncomp: int
    comps: Tuple[str, ...]
    area: float
    eb: float
    tstep: float
    u_f: float
    u_d: float
    u_e: float
    u_r: float
    run_name: Optional[str]
    dict_CF: Dict[int, float]
    dict_CD: Dict[int, float]
    dict_kapp: Dict[int, float]
    dict_qm: Dict[int, float]
    dict_K: Dict[int, float]
    dict_H: Dict[int, float]
    dict_U: Dict[int, float]
    dict_Dax: Dict[int, float]


def _slice_or_error(values: Sequence[float], ncomp: int, name: str) -> np.ndarray:
    if len(values) < ncomp:
        raise ValueError(f"{name} has {len(values)} values, need {ncomp}")
    return np.array(values[:ncomp], dtype=float)


def _resolve_flow_rates(config: SMBConfig, flow: Optional[FlowRates]) -> FlowRates:
    if flow is None:
        return FlowRates(
            F1=config.F1_init,
            Fdes=config.Fdes_init,
            Fex=config.Fex_init,
            Ffeed=config.Ffeed_init,
            Fraf=config.Fraf_init,
            tstep=config.tstep_init,
        )
    return FlowRates(
        F1=flow.F1,
        Fdes=flow.Fdes,
        Fex=flow.Fex,
        Ffeed=flow.Ffeed,
        Fraf=config.Fraf_init if flow.Fraf is None else flow.Fraf,
        tstep=config.tstep_init if flow.tstep is None else flow.tstep,
        u_f=flow.u_f,
        u_d=flow.u_d,
        u_e=flow.u_e,
        u_r=flow.u_r,
        run_name=flow.run_name,
    )


def build_inputs(config: SMBConfig, flow: Optional[FlowRates] = None) -> SMBInputs:
    nc = tuple(config.nc)
    if len(nc) != 4:
        raise ValueError('This build_inputs assumes 4 SMB sections')
    nsec = len(nc)
    ncols = sum(nc)
    ncomp = len(config.comps)

    params = get_isotherm_params(config.isoth)
    qm = _slice_or_error(params.qm, ncomp, 'qm')
    K = _slice_or_error(params.K, ncomp, 'K')
    H = _slice_or_error(params.H, ncomp, 'H')
    kapp = _slice_or_error(config.kapp, ncomp, 'kapp')
    rho = _slice_or_error(config.rho, ncomp, 'rho')
    wt0 = _slice_or_error(config.wt0, ncomp, 'wt0')

    flow = _resolve_flow_rates(config, flow)
    area = np.pi * config.d**2 / 4.0

    q_f = flow.Ffeed if flow.u_f is None else flow.u_f * area * config.eb
    q_d = flow.Fdes if flow.u_d is None else flow.u_d * area * config.eb
    q_e = flow.Fex if flow.u_e is None else flow.u_e * area * config.eb
    q_r = flow.Fraf if flow.u_r is None else flow.u_r * area * config.eb
    tstep = flow.tstep

    u_f = q_f / area / config.eb if flow.u_f is None else flow.u_f
    u_d = q_d / area / config.eb if flow.u_d is None else flow.u_d
    u_e = q_e / area / config.eb if flow.u_e is None else flow.u_e
    u_r = q_r / area / config.eb if flow.u_r is None else flow.u_r

    F1 = flow.F1
    F2 = F1 - q_e
    F3 = F2 + q_f
    F4 = F1 - q_d
    F = [F1, F2, F3, F4]

    F_guess = []
    for i, sec_len in enumerate(nc):
        F_guess.extend([F[i]] * sec_len)

    dict_U = {}
    dict_Dax = {}
    for i, f in enumerate(F_guess, start=1):
        dict_U[i] = f / area / config.eb
        dict_Dax[i] = dict_U[i] * config.L / config.Pe

    CF_gL = wt0 / np.sum(wt0 / rho)
    CF = CF_gL
    CD = np.zeros(ncomp, dtype=float)
    CD[-1] = rho[-1]

    dict_CF = {i + 1: CF[i] for i in range(ncomp)}
    dict_CD = {i + 1: CD[i] for i in range(ncomp)}
    dict_kapp = {i + 1: kapp[i] for i in range(ncomp)}
    dict_qm = {i + 1: qm[i] for i in range(ncomp)}
    dict_K = {i + 1: K[i] for i in range(ncomp)}
    dict_H = {i + 1: H[i] for i in range(ncomp)}

    return SMBInputs(
        nc=nc,
        nsec=nsec,
        ncols=ncols,
        ncomp=ncomp,
        comps=tuple(config.comps),
        area=area,
        eb=config.eb,
        tstep=tstep,
        u_f=u_f,
        u_d=u_d,
        u_e=u_e,
        u_r=u_r,
        run_name=flow.run_name,
        dict_CF=dict_CF,
        dict_CD=dict_CD,
        dict_kapp=dict_kapp,
        dict_qm=dict_qm,
        dict_K=dict_K,
        dict_H=dict_H,
        dict_U=dict_U,
        dict_Dax=dict_Dax,
    )
