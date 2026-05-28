#!/bin/bash
# Qwen3.6-35B-A3B-FP8 LLM server — GPU 0+1 (TP=2), port 8011
# nginx proxies external :8001 → internal :8011
# A100 SXM config — NVLink P2P enabled, no custom-all-reduce workaround needed
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$SCRIPT_DIR/.env"; set +a
export HF_HOME="$SCRIPT_DIR/models"
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_TORCH_COMPILE_CACHE="$SCRIPT_DIR/models/compile_cache"

LOG="$SCRIPT_DIR/logs/llm.log"
mkdir -p "$SCRIPT_DIR/logs"

echo "[$(date)] Starting LLM server on :8011 → external :8001 (GPU 0+1, TP=2)" | tee -a "$LOG"
exec vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --api-key "$API_KEY" \
    --revision 95a723d08a9490559dae23d0cff1d9466213d989 \
    --host 0.0.0.0 \
    --port 8011 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --safetensors-load-strategy prefetch \
    --language-model-only \
    --trust-remote-code \
    2>&1 | tee -a "$LOG"
