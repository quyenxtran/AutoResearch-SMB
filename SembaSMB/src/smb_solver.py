from typing import Dict, Optional
import shutil

from pyomo.opt import SolverFactory


def default_ipopt_options() -> Dict[str, float]:
    return {
        'linear_solver': 'ma57',
        'mu_init': 1e-3,
        'max_iter': 5000,
        'tol': 1e-6,
        'acceptable_tol': 1e-5,
        'halt_on_ampl_error': 'yes',
    }


def check_solver_available(solver_name: str = 'ipopt_sens') -> bool:
    try:
        if shutil.which(solver_name) is None:
            return False
        opt = SolverFactory(solver_name)
        if opt is None:
            return False
        return bool(opt.available())
    except Exception:
        return False


def solve_model(m, solver_name: str = 'ipopt_sens', options: Optional[Dict[str, float]] = None, tee: bool = True):
    opt = SolverFactory(solver_name)
    if options:
        for key, val in options.items():
            opt.options[key] = val
    results = opt.solve(m, tee=tee)


    return results
