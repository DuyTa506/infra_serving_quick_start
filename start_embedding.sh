#!/bin/bash
# BGE-Large embedding server — GPU 1, port 8000
# vLLM auto-detects embedding model type; no --task flag needed.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$SCRIPT_DIR/.env"; set +a
export HF_HOME="$SCRIPT_DIR/models"
export CUDA_VISIBLE_DEVICES=1

LOG="$SCRIPT_DIR/logs/embedding.log"
mkdir -p "$SCRIPT_DIR/logs"

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
