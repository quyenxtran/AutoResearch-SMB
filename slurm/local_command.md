## Ollama 27B model setup (one-time)

```bash
export OLLAMA_HOST=127.0.0.1:11555
export OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models

cat > /storage/scratch1/4/qtran47/models/gguf/qwen35-27b/Modelfile.chat64k <<'EOF'
FROM qwen35-27b-unsloth-q4-chat:latest
PARAMETER num_ctx 64000
PARAMETER temperature 0
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</think>"
SYSTEM "Return strict JSON only. No commentary."
EOF

~/.local/ollama/bin/ollama create qwen35-27b-unsloth-q4-chat64k \
  -f /storage/scratch1/4/qtran47/models/gguf/qwen35-27b/Modelfile.chat64k

# Verify model exists
find "$OLLAMA_MODELS/manifests" -type f | grep -i qwen35-27b-unsloth || echo "MISSING"
```

---

## Quick 3-job comparative batch (from `~/AutoResearch-SMB`)

```bash
cd ~/AutoResearch-SMB
TAG=cmp_$(date +%Y%m%d_%H%M%S)
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

sbatch --export=ALL,SMB_SUITE_TAG=$TAG slurm/pace_smb_minlp_cpu_24h.slurm
sbatch --export=ALL,SMB_SUITE_TAG=$TAG slurm/pace_smb_single_scientist_24h.slurm
sbatch --export=ALL,SMB_SUITE_TAG=$TAG slurm/pace_smb_two_scientists_24h.slurm
```

---

## Single-scientist mode (Qwen 9B)

```bash
JOBTAG=single_$(date +%Y%m%d_%H%M%S)
DB=~/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
OLLAMA_LOAD_TIMEOUT=20m,\
SMB_LLM_TIMEOUT_SECONDS=600,\
SMB_NC_LIBRARY=all,\
SMB_AGENT_MAX_SEARCH_EVALS=120,\
SMB_MIN_PROBE_REFERENCE_RUNS=35,\
SMB_SINGLE_SCIENTIST_MODE=1,\
SMB_PROBE_LOW_FIDELITY_ENABLED=1,\
SMB_PROBE_NFEX=5,\
SMB_PROBE_NFET=2,\
SMB_PROBE_NCP=1,\
SMB_AGENT_TEE=1,\
AGENT_ENTRYPOINT="/storage/scratch1/4/qtran47/AutoResearch-SMB/.venv/bin/python -m benchmarks.agent_runner --single-scientist-mode 1 --tee --run-name single_scientist_all35_${JOBTAG} --research-md ~/AutoResearch-SMB/artifacts/agent_runs/research_single_${JOBTAG}.md --sqlite-db ${DB} --reset-research-section" \
slurm/pace_smb_two_scientists_qwen.slurm
```

## Two-scientists mode (Qwen 9B)

```bash
JOBTAG=two_$(date +%Y%m%d_%H%M%S)
DB=~/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
OLLAMA_LOAD_TIMEOUT=20m,\
SMB_LLM_TIMEOUT_SECONDS=600,\
SMB_NC_LIBRARY=all,\
SMB_AGENT_MAX_SEARCH_EVALS=120,\
SMB_MIN_PROBE_REFERENCE_RUNS=35,\
SMB_SINGLE_SCIENTIST_MODE=0,\
SMB_PROBE_LOW_FIDELITY_ENABLED=1,\
SMB_PROBE_NFEX=5,\
SMB_PROBE_NFET=2,\
SMB_PROBE_NCP=1,\
SMB_AGENT_TEE=1,\
AGENT_ENTRYPOINT="/storage/scratch1/4/qtran47/AutoResearch-SMB/.venv/bin/python -m benchmarks.agent_runner --single-scientist-mode 0 --tee --run-name two_scientists_all35_${JOBTAG} --research-md ~/AutoResearch-SMB/artifacts/agent_runs/research_two_${JOBTAG}.md --sqlite-db ${DB} --reset-research-section" \
slurm/pace_smb_two_scientists_qwen.slurm
```

## Two-scientists mode (Qwen 27B Unsloth GGUF)

```bash
JOBTAG=two_$(date +%Y%m%d_%H%M%S)
DB=/storage/home/hcoda1/4/qtran47/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
SMB_FALLBACK_LLM_ENABLED=0,\
SMB_LOCAL_LLM_MODEL=qwen35-27b-unsloth-q4-chat64k:latest,\
OLLAMA_MODEL=qwen35-27b-unsloth-q4-chat64k:latest,\
OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models,\
OLLAMA_HOST=127.0.0.1:11555,\
OLLAMA_LOAD_TIMEOUT=40m,\
SMB_LLM_TIMEOUT_SECONDS=1200,\
OLLAMA_NUM_PARALLEL=1,\
SMB_NC_LIBRARY=all,\
SMB_AGENT_MAX_SEARCH_EVALS=120,\
SMB_MIN_PROBE_REFERENCE_RUNS=35,\
SMB_SINGLE_SCIENTIST_MODE=0,\
SMB_PROBE_LOW_FIDELITY_ENABLED=1,\
SMB_PROBE_NFEX=5,\
SMB_PROBE_NFET=2,\
SMB_PROBE_NCP=1,\
SMB_AGENT_TEE=1,\
AGENT_ENTRYPOINT="/storage/scratch1/4/qtran47/AutoResearch-SMB/.venv/bin/python -m benchmarks.agent_runner --single-scientist-mode 0 --tee --run-name two_scientists_all35_${JOBTAG} --research-md /storage/home/hcoda1/4/qtran47/AutoResearch-SMB/artifacts/agent_runs/research_two_${JOBTAG}.md --sqlite-db ${DB} --reset-research-section" \
slurm/pace_smb_two_scientists_qwen.slurm
```

---



## Monitoring a running job

```bash
# Live output/error
tail -f ~/AutoResearch-SMB/logs/smb-two-scientists-<JOBID>.out
tail -f ~/AutoResearch-SMB/logs/smb-two-scientists-<JOBID>.err

# CPU monitor
srun --jobid=<JOBID> --overlap bash -lc 'while true; do top -b -n 1 | head -n 25; sleep 2; done'

# GPU monitor
srun --jobid=<JOBID> --overlap bash -lc '
while true; do
  clear
  echo "=== $(date) ==="
  top -b -n 1 | head -n 20
  echo
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
  sleep 2
done'
```

## Live conversation stream (while job is running)

```bash
JOB=<JOBID>
BASE=$(ls -t ~/AutoResearch-SMB/artifacts/agent_runs/agent-runner.${JOB}.*.conversations.jsonl | head -1 | sed 's/\.conversations\.jsonl$//')

# Compact role/decision stream
tail -f "${BASE}.conversations.jsonl" \
| jq -rc '{id:.call_id,role:.role,backend:.final_backend,ok:(.assistant_response!=null)}'

# Scientist_B decision quality
jq -r 'select(.role=="scientist_b_review")
| (.assistant_response|fromjson? // {})
| [.decision // "NA", .reason // "NA"] | @tsv' \
"${BASE}.conversations.jsonl" | column -t -s $'\t'
```

## Query current results from SQLite

```bash
RUN=<your_run_name>
DB=~/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_<JOBTAG>.sqlite

sqlite3 "$DB" "
SELECT candidate_run_name, nc, seed_name, status, feasible,
       round(purity,4) purity, round(recovery_ga,4) rga, round(recovery_ma,4) rma,
       round(productivity,6) prod, round(coalesce(normalized_total_violation,-1),6) viol
FROM simulation_results
WHERE agent_run_name='$RUN'
ORDER BY feasible DESC, prod DESC
LIMIT 20;"
```

## Deploy new code to PACE

```bash
# Local: commit and push
git add .
git commit -m "your message"
git push origin main

# On PACE
cd ~/AutoResearch-SMB
git pull
```
