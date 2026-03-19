## Deploy new code to PACE

```bash
# Local: commit and push
git add .
git commit -m "Updated prompt"
git push origin main

# On PACE
cd ~/AutoResearch-SMB
git pull
```


ROOT=/storage/home/hcoda1/4/qtran47/Agent-Driven-NLP-Optimizer
JOBTAG=v2_$(date +%Y%m%d_%H%M%S)
DB=$ROOT/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite
LIVE=$ROOT/artifacts/agent_runs/live_results_${JOBTAG}.jsonl
export SMB_TSTEP_BOUNDS="8.0,12.0"

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
LOCAL_LLM_USE_GPU=1,\
OLLAMA_GPU_ID=0,\
CUDA_VISIBLE_DEVICES=0,\
OLLAMA_LLM_LIBRARY=cuda_v12,\
SMB_FALLBACK_LLM_ENABLED=0,\
SMB_LOCAL_LLM_MODEL=qwen35-9b-q4-32k:latest,\
SMB_EXECUTIVE_LLM_MODEL=deepseek-r1:7b,\
OLLAMA_HOST=127.0.0.1:11555,\
OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models,\
OLLAMA_NUM_PARALLEL=1,\
OLLAMA_MAX_LOADED_MODELS=1,\
SMB_LLM_TIMEOUT_SECONDS=1200,\
SMB_OLLAMA_PREWARM_ENABLED=1,\
SMB_OLLAMA_PREWARM_MAX_SECONDS=180,\
SMB_METHOD=agent_v2,\
SMB_EXECUTIVE_ARBITRATION_ENABLED=1,\
SMB_EXECUTIVE_MAX_REVISIONS=1,\
SMB_SYSTEMATIC_INFEASIBILITY_K=5,\
SMB_RANDOM_SEARCH_MODE=0,\
SMB_CONVERSATION_LOG_MODE=full,\
SMB_LIVE_RESULTS_LOG=${LIVE},\
AGENT_ENTRYPOINT="${ROOT}/.venv/bin/python -m benchmarks.agent_runner --method agent_v2 --run-name agent_v2_${JOBTAG} --tee --research-md ${ROOT}/artifacts/agent_runs/research_agent_v2_${JOBTAG}.md --sqlite-db ${DB} --reset-research-section" \
slurm/pace_smb_two_scientists_qwen.slurm




## 3) Random baseline run
JOBTAG=random_$(date +%Y%m%d_%H%M%S)
ROOT=/storage/scratch1/4/qtran47/Agent-Driven-NLP-Optimizer
DB=/storage/home/hcoda1/4/qtran47/Agent-Driven-NLP-Optimizer/artifacts/agent_runs/smb_agent_context_${JOBTAG}.sqlite
export SMB_TSTEP_BOUNDS="8.0,12.0"

sbatch --export=ALL,\
START_LOCAL_LLM=1,\
SMB_METHOD=random,\
SMB_RANDOM_SEARCH_MODE=1,\
SMB_AGENT_LLM_ENABLED=0,\
SMB_FALLBACK_LLM_ENABLED=0,\
SMB_SKIP_INITIAL_PLAN_LLM=1,\
OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models,\
OLLAMA_HOST=127.0.0.1:11555,\
OLLAMA_NUM_PARALLEL=1,\
OLLAMA_MAX_LOADED_MODELS=1,\
SMB_NC_LIBRARY=all,\
SMB_AGENT_MAX_SEARCH_EVALS=120,\
SMB_MIN_PROBE_REFERENCE_RUNS=35,\
SMB_PROBE_LOW_FIDELITY_ENABLED=1,\
SMB_PROBE_NFEX=5,\
SMB_PROBE_NFET=2,\
SMB_PROBE_NCP=1,\
SMB_AGENT_TEE=1,\

AGENT_ENTRYPOINT="${ROOT}/.venv/bin/python -m benchmarks.agent_runner --method random --random-search-mode 1 --run-name random_${JOBTAG} --tee --research-md /storage/home/hcoda1/4/qtran47/Agent-Driven-NLP-Optimizer/artifacts/agent_runs/research_random_${JOBTAG}.md --sqlite-db ${DB} --reset-research-section" \
slurm/pace_smb_two_scientists_qwen.slurm



## Monitoring a running job

```bash
# Live output/error
tail -n 30 -f logs/smb-two-scientists-5077708.out 
tail -n 30 -f logs/smb-two-scientists-5077708.err
tail -n 30 -f logs/ollama-smb-5077708.log 

# CPU/GPU monitor
srun --jobid=5077708 --overlap bash -lc '
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
JOB=5077708

tail -F "$FILE" | jq -r '[.call_id,.role,(.metadata.iteration//""),(.assistant_response//.assistant_response_preview//"")] | @tsv'

## Use this for live A + B + C full text:

JOB=5077708
FILE=$(ls -t artifacts/agent_runs/*.conversations.jsonl 2>/dev/null | head -1); echo "$FILE"
tail -F "$FILE" | jq -r 'select(.role|test("^scientist_(a_pick|b_review|c_arbitrate)$")) | "\n--- call=\(.call_id) role=\(.role) iter=\(.metadata.iteration // "") backend=\(.final_backend) ---\n\(.assistant_response // .assistant_response_preview // "")\n"'


## Use this for a live structured quality view (decision/reason/comparison counts/physics field):

tail -F "$FILE" | jq -r '
def txt: (.assistant_response // .assistant_response_preview // "");
def j: (txt|fromjson? // {});
select(.role|test("^scientist_(a_pick|b_review|c_arbitrate)$")) |
[
  (.call_id|tostring),
  .role,
  ((.metadata.iteration//"")|tostring),
  (.final_backend//""),
  (j.decision // j.acquisition_type // ""),
  (j.reason // ""),
  (((j.comparison_to_previous // j.comparison_assessment // [])|length)|tostring),
  (((j.last_two_run_comparison // j.last_two_run_audit // [])|length)|tostring),
  (((j.physics_rationale // j.physics_audit // "")|tostring|.[0:120]))
] | @tsv'



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

# Test LLM:

srun --pty -A gts-sn73 --qos=inferno -p gpu-rtx6000 \
  --gres=gpu:rtx_6000:1 --cpus-per-task=6 --mem=32G --time=01:00:00 bash -l


export OLLAMA_HOST=127.0.0.1:11555
export OLLAMA_MODELS=/storage/scratch1/4/qtran47/.ollama/models
pkill -u "$USER" -f "ollama (serve|runner)" || true
nohup ~/.local/ollama/bin/ollama serve > ~/AutoResearch-SMB/logs/ollama-salloc.log 2>&1 &
sleep 5
curl -fsS http://127.0.0.1:11555/api/tags | jq '.models[].name'


# Short vs long prompt latency test
python - <<'PY'
import requests,time
host="http://127.0.0.1:11555"; model="qwen35-9b-q4-32k:latest"
for n in [ 32000]:
    p=("SMB mass balance and zone coupling. "*n) + "\nReply with exactly: pong"
    t=time.time()
    r=requests.post(f"{host}/api/chat",json={
        "model":model,
        "messages":[{"role":"user","content":p}],
        "stream":False,
        "options":{"num_ctx":32768,"num_predict":24,"temperature":0}
    },timeout=300)
    dt=time.time()-t
    txt=(r.json().get("message",{}).get("content","") or "").replace("\n"," ")[:80]
    print({"repeat":n,"status":r.status_code,"wall_s":round(dt,2),"head":txt})
PY

Results:

{'repeat': 5000, 'status': 200, 'wall_s': 38.04, 'head': '  pong<|endoftext|><|im_start|> <|im_start|> <|im_start|> <|im_start|> '}  
{'repeat': 8000, 'status': 200, 'wall_s': 38.24, 'head': '  pong<|endoftext|><|im_start|> <|im_start|> <|im_start|> <|im_start|> '}  
{'repeat': 12000, 'status': 200, 'wall_s': 38.7, 'head': '  pong<|endoftext|><|im_start|> <|im_start|> <|im_start|> <|im_start|> '} 