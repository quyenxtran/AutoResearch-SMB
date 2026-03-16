cd ~/AutoResearch-SMB
TAG=cmp_$(date +%Y%m%d_%H%M%S)
# Common knobs
export SMB_NC_LIBRARY=all
export SMB_AGENT_MAX_SEARCH_EVALS=120
export SMB_MIN_PROBE_REFERENCE_RUNS=35
export SMB_PROBE_LOW_FIDELITY_ENABLED=1
export SMB_PROBE_NFEX=5
export SMB_PROBE_NFET=2
export SMB_PROBE_NCP=1
export SMB_IPOPT_MAX_ITER=1000
export SMB_IPOPT_TOL=1e-5
export SMB_IPOPT_ACCEPTABLE_TOL=1e-4
export SMB_FFEED_BOUNDS="0.5,4.0"
export SMB_FRAF_BOUNDS="0.5,4.0"
export SMB_MAX_PUMP_ML_MIN=4.0
# 1) MINLP (CPU)
sbatch --export=ALL,SMB_SUITE_TAG=$TAG pace_smb_minlp_cpu_24h.slurm

# 2) Single scientist (GPU)
sbatch --export=ALL,SMB_SUITE_TAG=$TAG pace_smb_single_scientist_24h.slurm

# 3) Two scientists (GPU)
sbatch --export=ALL,SMB_SUITE_TAG=$TAG pace_smb_two_scientists_24h.slurm


