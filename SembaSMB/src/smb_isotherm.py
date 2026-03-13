from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class IsothermParams:
    qm: Tuple[float, ...]
    K: Tuple[float, ...]
    H: Tuple[float, ...]


def get_isotherm_params(isoth: str) -> IsothermParams:
    # GA, MA, Water, MeOH
    isoth = isoth.upper()
    if isoth == 'L':
        qm = (0.12, 0.28, 0.15)
        K = (50.0, 5.0, 75.0)
        H = (0.0, 0.0, 0.0)
    elif isoth == 'MLL':
        H = (0.61, 0.52, 1e-3, 0.06)
        qm = (0.084, 0.117, 0.02, 0.05)
        K = (254.0, 1208.0, 1e-3, 79.0)

    elif isoth == 'MLLE':
        qm = (1.757, 0.129, 0.02, 0.054)
        K = (9.4, 611.3, 1e-3, 52.0)
        H = (0.0025, 0.00085, 1e-3, 0.00027)
    else:
        raise ValueError(f"Unknown isotherm '{isoth}'")
    return IsothermParams(qm=qm, K=K, H=H)
