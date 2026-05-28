#!/bin/bash
# BGE-Reranker-v2-m3 reranking server — GPU 0, port 8002
# vLLM auto-detects reranker/scoring model type; no --task flag needed.
# Endpoints: /rerank  /v1/rerank  /v2/rerank  /score
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$SCRIPT_DIR/.env"; set +a
export HF_HOME="$SCRIPT_DIR/models"
export CUDA_VISIBLE_DEVICES=0

LOG="$SCRIPT_DIR/logs/reranker.log"
mkdir -p "$SCRIPT_DIR/logs"

echo "[$(date)] Starting reranker server on :8012 → external :8002 (GPU 0)" | tee -a "$LOG"
exec vllm serve BAAI/bge-reranker-v2-m3 \
    --api-key "$API_KEY" \
    --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
    --host 0.0.0.0 \
    --port 8012 \
    --dtype float16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.15 \
    --trust-remote-code \
    2>&1 | tee -a "$LOG"
