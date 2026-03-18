## Deploy new code to PACE

```bash
# Local: commit and push
git add .
git commit -m "further compact initial context"
git push origin main

# On PACE
cd ~/AutoResearch-SMB
git pull
```


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
#9B with 32k context
JOBTAG=two_$(date +%Y%m%d_%H%M%S)
DB=/storage/home/hcoda1/4/qtran47/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite
export SMB_TSTEP_BOUNDS="8.0,12.0"

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
SMB_FALLBACK_LLM_ENABLED=0,\
SMB_LOCAL_LLM_MODEL=qwen35-9b-q4-32k:latest,\
OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models,\
OLLAMA_HOST=127.0.0.1:11555,\
OLLAMA_LOAD_TIMEOUT=40m,\
SMB_LLM_TIMEOUT_SECONDS=1800,\
OLLAMA_NUM_PARALLEL=1,\
OLLAMA_MAX_LOADED_MODELS=1,\
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


#27B
JOBTAG=$(date +%Y%m%d_%H%M%S)
DB=/storage/home/hcoda1/4/qtran47/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite
export SMB_TSTEP_BOUNDS="8.0,12.0"

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
SMB_FALLBACK_LLM_ENABLED=0,\
SMB_LOCAL_LLM_MODEL=qwen3.5:9b,\
OLLAMA_MODEL=qwen35-27b-unsloth-q4-chat64k:latest,\
OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models,\
OLLAMA_HOST=127.0.0.1:11555,\
OLLAMA_LOAD_TIMEOUT=40m,\
SMB_LLM_TIMEOUT_SECONDS=1800,\
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

## Two-scientists mode (Qwen 27B Unsloth GGUF)

```bash

JOBTAG=$(date +%Y%m%d_%H%M%S)
DB=/storage/home/hcoda1/4/qtran47/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite
export SMB_TSTEP_BOUNDS="8.0,12.0"

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
SMB_FALLBACK_LLM_ENABLED=0,\
SMB_LOCAL_LLM_MODEL=qwen35-27b-unsloth-q4-chat64k:latest,\
OLLAMA_MODEL=qwen35-27b-unsloth-q4-chat64k:latest,\
OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models,\
OLLAMA_HOST=127.0.0.1:11555,\
OLLAMA_LOAD_TIMEOUT=40m,\
SMB_LLM_TIMEOUT_SECONDS=1800,\
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



JOBTAG=$(date +%Y%m%d_%H%M%S)
DB=/storage/home/hcoda1/4/qtran47/AutoResearch-SMB/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite
export SMB_TSTEP_BOUNDS="8.0,12.0"

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
SMB_FALLBACK_LLM_ENABLED=0,\
SMB_LOCAL_LLM_MODEL=qwen35-27b-unsloth-q4-chat64k:latest,\
OLLAMA_MODEL=qwen35-27b-unsloth-q4-chat64k:latest,\
OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models,\
OLLAMA_HOST=127.0.0.1:11555,\
OLLAMA_LOAD_TIMEOUT=40m,\
SMB_LLM_TIMEOUT_SECONDS=1800,\
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
slurm/pace_smb_two_scientists_qwen_a100.slurm




```

## Monitoring a running job

```bash
# Live output/error
tail -f ~/AutoResearch-SMB/logs/smb-two-scientists-5052358.out
tail -f ~/AutoResearch-SMB/logs/smb-two-scientists-5030528.err

# CPU monitor
srun --jobid=5030528 --overlap bash -lc 'while true; do top -b -n 1 | head -n 25; sleep 2; done'

# CPU/GPU monitor
srun --jobid=5052358 --overlap bash -lc '
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
Live compact stream (scientist A/B/C only):

JOB=5052358
FILE=$(find ~/AutoResearch-SMB/artifacts/agent_runs -maxdepth 1 -type f -name "agent-runner.${JOB}*.conversations.jsonl" | sort | tail -1)

if [ -z "$FILE" ]; then
  echo "No conversations file found for JOB=$JOB"
  echo "Recent files:"
  ls -lt ~/AutoResearch-SMB/artifacts/agent_runs/agent-runner.*.conversations.jsonl | head
  exit 1
fi

echo "Using: $FILE"

tail -f "$FILE" | jq -r '
def ar:
  (.assistant_response // {}) as $x
  | if ($x|type)=="string" then ($x|fromjson? // {"_raw":$x}) else $x end;

select(.role|test("^scientist_[abc]_")) |
[
  .call_id,
  .role,
  (.final_backend // "NA"),
  (ar.decision // ("candidate=" + ((ar.candidate_index // "NA")|tostring))),
  ((ar.reason // "NA") | gsub("[\r\n\t]+";" "))
] | @tsv
' | column -t -s $'\t'




##
JOB=5050561
FILE=$(ls -t ~/AutoResearch-SMB/artifacts/agent_runs/agent-runner.${JOB}.*.conversations.jsonl | head -1)
echo "Using: $FILE"
Full scientist messages (pretty, A/B/C only):
tail -f "$FILE" | jq -r '
select(.role|test("^scientist_[abc]_")) |
"----- call_id=\(.call_id) role=\(.role) backend=\(.final_backend) -----",
(.assistant_response // "null"),
""
'




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



PORT=11555
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models

cat > /storage/scratch1/4/qtran47/models/gguf/qwen35-27b/Modelfile.chat32k <<'EOF'
FROM qwen35-27b-unsloth-q4-chat:latest
PARAMETER num_ctx 32000
PARAMETER temperature 0
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</think>"
SYSTEM "Return strict JSON only. No commentary."
EOF

~/.local/ollama/bin/ollama create qwen35-27b-unsloth-q4-chat32k \
  -f /storage/scratch1/4/qtran47/models/gguf/qwen35-27b/Modelfile.chat32k


PORT=11555
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models

cat > /storage/scratch1/4/qtran47/models/gguf/qwen35-27b/Modelfile.chat150k <<'EOF'
FROM qwen35-27b-unsloth-q4-chat:latest
PARAMETER num_ctx 150000
PARAMETER temperature 0
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</think>"
SYSTEM "Return strict JSON only. No commentary."
EOF

~/.local/ollama/bin/ollama create qwen35-27b-unsloth-q4-chat150k \
  -f /storage/scratch1/4/qtran47/models/gguf/qwen35-27b/Modelfile.chat150k





export OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models
mkdir -p /storage/scratch1/4/qtran47/models/gguf/qwen35-9b

# if using built-in qwen3.5:9b as base
cat > /storage/scratch1/4/qtran47/models/gguf/qwen35-9b/Modelfile.qwen9b.48k <<'EOF'
FROM qwen3.5:9b
PARAMETER num_ctx 48000
PARAMETER temperature 0.2
PARAMETER top_p 0.95
EOF

~/.local/ollama/bin/ollama create qwen35-9b-48k \
  -f /storage/scratch1/4/qtran47/models/gguf/qwen35-9b/Modelfile.qwen9b.48k


export OLLAMA_HOST=127.0.0.1:11556

# quick liveness
curl -m 600 -sS http://127.0.0.1:11556/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen35-9b-48k:latest","messages":[{"role":"user","content":"Reply with exactly: pong"}],"stream":false}' | jq -r '.message.content'


curl -m 600 -sS http://127.0.0.1:11556/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen35-9b-48k:latest","messages":[{"role":"user","content":"Reply with exactly: pong"}],"stream":false}' | jq .




salloc -A gts-sn73 -p gpu-rtx6000 --qos=inferno --gres=gpu:rtx_6000:1 --cpus-per-task=6 --mem=32G --time=02:00:00
srun --pty -A gts-sn73 -p gpu-rtx6000 --qos=inferno bash -l


# 1) job still alive?
squeue -j 5050561.18i %.10T %.10M %.10l %R"

# 2) is Ollama serving and model loaded?
curl -sS http://127.0.0.1:11555/api/ps | jq .

# 3) any recent Ollama errors/timeouts?
tail -n 120 ~/AutoResearch-SMB/logs/ollama-11555.log | egrep -i "chat|completions|timeout|500|error|loading model"
