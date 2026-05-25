#!/bin/bash
# BGE-Large embedding server — GPU 1, port 8000
# vLLM auto-detects embedding model type; no --task flag needed.
set -a; source /workspace/.env; set +a
export HF_HOME=/workspace/models
export CUDA_VISIBLE_DEVICES=1

LOG=/workspace/logs/embedding.log
mkdir -p /workspace/logs

echo "[$(date)] Starting embedding server on :8010 → external :8000 (GPU 1)" | tee -a "$LOG"
exec vllm serve BAAI/bge-large-en-v1.5 \
    --api-key "$API_KEY" \
    --revision d4aa6901d3a41ba39fb536a557fa166f842b0e09 \
    --host 0.0.0.0 \
    --port 8010 \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.15 \
    --trust-remote-code \
    2>&1 | tee -a "$LOG"
