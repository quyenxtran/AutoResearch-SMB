# SMB Simulation (Pyomo DAE)

This workspace contains a Pyomo DAE model for SMB/TMB chromatography and a flowrate sweep notebook.

## Structure
- src/ contains the modularized model code (config, model, discretization, solver, metrics, plotting, optimization)
- SMB_flowrate_sweep.ipynb runs a small sweep of flowrates
- Acids_SMB_25 05 20_Run11_updated.ipynb is the original notebook

## Setup (Windows)
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirement.txt

## Running
.\.venv\Scripts\jupyter nbconvert --execute --to notebook --inplace SMB_flowrate_sweep.ipynb
.\.venv\Scripts\jupyter nbconvert --execute --to notebook --inplace "Acids_SMB_25 05 20_Run11_updated.ipynb"

## Solver
The notebooks use the ipopt_sens solver. Ensure it is available on PATH or update the solver name in src/smb_solver.py or the notebooks.
