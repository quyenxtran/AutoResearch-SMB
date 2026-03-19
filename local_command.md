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






JOBTAG=two_$(date +%Y%m%d_%H%M%S)
DB=/storage/home/hcoda1/4/qtran47/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
SMB_SINGLE_SCIENTIST_MODE=0,\
SMB_FALLBACK_LLM_ENABLED=0,\
SMB_LOCAL_LLM_MODEL=qwen35-9b-q4-32k:latest,\
OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models,\
OLLAMA_HOST=127.0.0.1:11555,\
SMB_LLM_TIMEOUT_SECONDS=90,\
SMB_LLM_MAX_RETRIES=1,\
SMB_LLM_RETRY_BACKOFF_SECONDS=0,\
SMB_AGENT_TEE=1,\
AGENT_ENTRYPOINT="/storage/scratch1/4/qtran47/AutoResearch-SMB/.venv/bin/python -m benchmarks.agent_runner --single-scientist-mode 0 --tee --run-name two_scientists_${JOBTAG} --research-md /storage/home/hcoda1/4/qtran47/AutoResearch-SMB/artifacts/agent_runs/research_two_${JOBTAG}.md --sqlite-db ${DB} --reset-research-section" \
pace_smb_two_scientists_qwen.slurm



tail -f logs/smb-two-scientists-5056263.out 
tail -f logs/smb-two-scientists-5055879.err

JOB=5056263
srun --jobid=$JOB --pty bash -lc 'while true; do clear; echo "=== GPU ==="; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader; echo; echo "=== CPU ==="; top -b -n 1 | head -20; sleep 1; done'


JOB=5056263
srun --jobid=$JOB --pty bash -lc 
FILE=$(ls -t artifacts/agent_runs/agent-runner.'"$JOB"'.*.conversations.jsonl 2>/dev/null | head -1)
echo "FILE=$FILE"