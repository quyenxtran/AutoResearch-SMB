from typing import Dict, Optional
import shutil

from pyomo.opt import SolverFactory


def default_ipopt_options() -> Dict[str, float]:
    return {
        'linear_solver': 'ma57',
        'mu_init': 1e-1,           # Conservative: start well inside interior
        'mu_strategy': 'adaptive', # Let IPOPT adapt barrier parameter
        'max_iter': 5000,
        'tol': 1e-6,
        'acceptable_tol': 1e-5,
        'halt_on_ampl_error': 'yes',
        'bound_push': 1e-6,        # Push initial point away from bounds
        'bound_frac': 1e-6,
        'print_level': 5,          # Full iteration table (iter, obj, inf_pr, inf_du, ...)
    }


def warm_start_options() -> Dict[str, object]:
    """Options to enable warm-starting from a prior feasible solution."""
    return {
        'warm_start_init_point': 'yes',
        'warm_start_bound_push': 1e-8,
        'warm_start_bound_frac': 1e-8,
        'warm_start_mult_bound_push': 1e-8,
        'mu_init': 1e-4,           # Tighter start when warm-starting
    }


def feasibility_restoration_options() -> Dict[str, object]:
    """Options for Phase 1 (feasibility) solve — minimize constraint violation."""
    return {
        'mu_init': 1e-1,
        'mu_strategy': 'adaptive',
        'max_iter': 2000,
        'tol': 1e-4,              # Looser tolerance for feasibility phase
        'acceptable_tol': 1e-3,
        'bound_push': 1e-6,
        'bound_frac': 1e-6,
    }


def check_solver_available(solver_name: str = 'ipopt') -> bool:
    try:
        if shutil.which(solver_name) is None:
            return False
        opt = SolverFactory(solver_name)
        if opt is None:
            return False
        return bool(opt.available())
    except Exception:
        return False


def solve_model(
    m,
    solver_name: str = 'ipopt',
    options: Optional[Dict[str, float]] = None,
    tee: bool = True,
    logfile: Optional[str] = None,
):
    """Solve a Pyomo model with IPOPT.

    Parameters
    ----------
    tee : bool
        Stream IPOPT iteration output to stdout (visible in notebook cells
        and SLURM .out files).
    logfile : str, optional
        Path for IPOPT to write its own log file (``output_file`` option).
        Useful on PACE for ``tail -f`` monitoring.  If provided, also sets
        ``file_print_level=5`` so the file gets the full iteration table.
    """
    opt = SolverFactory(solver_name)
    if options:
        for key, val in options.items():
            opt.options[key] = val
    if logfile is not None:
        opt.options['output_file'] = logfile
        opt.options.setdefault('file_print_level', 5)
    results = opt.solve(m, tee=tee)

    return results
