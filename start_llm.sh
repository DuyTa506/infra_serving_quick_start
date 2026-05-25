#!/bin/bash
# Qwen3.6-35B-A3B-FP8 LLM server — GPU 0+1 (TP=2), port 8011
# nginx proxies external :8001 → internal :8011
# 0.90 GPU utilization for maximum KV cache / throughput
set -a; source /workspace/.env; set +a
export HF_HOME=/workspace/models
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LOG=/workspace/logs/llm.log
mkdir -p /workspace/logs

echo "[$(date)] Starting LLM server on :8011 → external :8001 (GPU 0+1, TP=2)" | tee -a "$LOG"
exec vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --api-key "$API_KEY" \
    --revision 95a723d08a9490559dae23d0cff1d9466213d989 \
    --host 0.0.0.0 \
    --port 8011 \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --language-model-only \
    --trust-remote-code \
    2>&1 | tee -a "$LOG"
