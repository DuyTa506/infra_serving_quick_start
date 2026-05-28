#!/bin/bash
# BGE-Large embedding server — GPU 1, port 8000
# Uses chunked processing (CLS pooling) to support long documents
# without exceeding the model's native 512-token context window.
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
    --gpu-memory-utilization 0.07 \
    --trust-remote-code \
    --served-model-name bge-large-en-v1.5 \
    --pooler-config '{"pooling_type":"CLS","use_activation":true,"enable_chunked_processing":true,"max_embed_len":2048}' \
    2>&1 | tee -a "$LOG"
