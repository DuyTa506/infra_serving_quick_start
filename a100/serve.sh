#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  vLLM Stack — No-Docker serve script for 2× A100 80GB (Ampere)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Spins up 6 vLLM processes (LLM + embedding + reranker on each GPU) plus an
#  optional nginx load balancer. Everything runs as background processes; logs
#  go to ./logs/<name>.log. Restart-safe: kills previous instances before start.
#
#  Usage:
#    bash serve.sh              # start everything
#    bash serve.sh llm          # start only the 2 LLM replicas
#    bash serve.sh nginx        # start only nginx (after backends are up)
#
#  Ports (direct — bypass nginx):
#    GPU 0:  embed=18000  llm=18001  rerank=18002
#    GPU 1:  embed=18010  llm=18011  rerank=18012
#
#  Ports (via nginx LB — start separately):
#    8000 embedding   |   8001 llm   |   8002 reranker
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config (override via .env) ──────────────────────────────────────────────
[ -f .env ] && source .env

API_KEY="${API_KEY:-sk-secai2026}"
HF_TOKEN="${HF_TOKEN:-}"

LLM_MODEL="${LLM_MODEL:-palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-32768}"

HF_HOME="${HF_HOME:-$SCRIPT_DIR/../models}"
export HF_HOME
[ -n "$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR" "$HF_HOME"

# ── Helpers ─────────────────────────────────────────────────────────────────
_pid_file() { echo "/tmp/vllm-a100-${1}.pid"; }
_is_running() { [ -f "$(_pid_file "$1")" ] && kill -0 "$(cat "$(_pid_file "$1")")" 2>/dev/null; }

_launch() {
    local name="$1" gpu="$2" port="$3"
    shift 3
    local pidfile="$(_pid_file "$name")"
    local logfile="$LOGDIR/${name}.log"

    if _is_running "$name"; then
        echo "  [skip] $name already running (pid $(cat "$pidfile"))"
        return 0
    fi

    echo "  [start] $name → 127.0.0.1:$port (GPU $gpu)"
    # TP=1 per replica → no P2P needed. Disable to avoid cross-GPU memory errors
    # on A100s with NVLink where another process may have enabled P2P.
    CUDA_VISIBLE_DEVICES="$gpu" \
    NCCL_P2P_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nohup vllm serve "$@" \
            --host 127.0.0.1 --port "$port" \
            --api-key "$API_KEY" \
            >> "$logfile" 2>&1 &
    echo $! > "$pidfile"

    # Wait for this service to finish loading before launching next on same GPU.
    # Prevents OOM races when multiple services compete for VRAM.
    echo -n "    waiting for $name to load..."
    for i in $(seq 1 300); do
        if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$port/health" 2>/dev/null; then
            echo " ready (${i}s)"
            break
        fi
        if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo " FAILED — check $logfile"
            return 1
        fi
        sleep 2
        [ $((i % 15)) -eq 0 ] && echo -n " ${i}s..."
    done
}

# ── Main ────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

echo "=== vLLM Stack — 2× A100 (no Docker) ==="
echo "Logs: $LOGDIR/"
echo ""

# ── GPU 0 ───────────────────────────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "embed" ] || [ "$MODE" = "embed-0" ]; then
    _launch embed-0 0 18000 \
        --model "$EMBED_MODEL" \
        --runner pooling \
        --dtype float16 \
        --max-model-len 512 \
        --gpu-memory-utilization 0.05 \
        --trust-remote-code
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "llm" ] || [ "$MODE" = "llm-0" ]; then
    _launch llm-0 0 18001 \
        --model "$LLM_MODEL" \
        --quantization gptq \
        --tensor-parallel-size 1 \
        --max-model-len "$LLM_MAX_MODEL_LEN" \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.80 \
        --max-num-seqs 128 \
        --enable-prefix-caching \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "rerank" ] || [ "$MODE" = "rerank-0" ]; then
    _launch rerank-0 0 18002 \
        --model "$RERANK_MODEL" \
        --runner pooling \
        --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
        --chat-template "$SCRIPT_DIR/../templates/qwen3_reranker.jinja" \
        --dtype float16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.06 \
        --trust-remote-code
fi

# ── GPU 1 ───────────────────────────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "embed" ] || [ "$MODE" = "embed-1" ]; then
    _launch embed-1 1 18010 \
        --model "$EMBED_MODEL" \
        --runner pooling \
        --dtype float16 \
        --max-model-len 512 \
        --gpu-memory-utilization 0.05 \
        --trust-remote-code
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "llm" ] || [ "$MODE" = "llm-1" ]; then
    _launch llm-1 1 18011 \
        --model "$LLM_MODEL" \
        --quantization gptq \
        --tensor-parallel-size 1 \
        --max-model-len "$LLM_MAX_MODEL_LEN" \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.80 \
        --max-num-seqs 128 \
        --enable-prefix-caching \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "rerank" ] || [ "$MODE" = "rerank-1" ]; then
    _launch rerank-1 1 18012 \
        --model "$RERANK_MODEL" \
        --runner pooling \
        --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
        --chat-template "$SCRIPT_DIR/../templates/qwen3_reranker.jinja" \
        --dtype float16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.06 \
        --trust-remote-code
fi

# ── nginx (optional — load balancer) ─────────────────────────────────────────
NGINX_PIDFILE="/tmp/vllm-a100-nginx.pid"
if [ "$MODE" = "all" ] || [ "$MODE" = "nginx" ]; then
    if command -v nginx &>/dev/null; then
        if [ -f "$NGINX_PIDFILE" ] && kill -0 "$(cat "$NGINX_PIDFILE")" 2>/dev/null; then
            echo "  [skip] nginx already running"
        else
            echo "  [start] nginx → :8000 (embed)  :8001 (llm)  :8002 (rerank)"
            cp -f "$SCRIPT_DIR/nginx.conf" /etc/nginx/sites-available/vllm-stack 2>/dev/null || true
            mkdir -p /etc/nginx/sites-enabled
            ln -sf /etc/nginx/sites-available/vllm-stack /etc/nginx/sites-enabled/vllm-stack 2>/dev/null || true
            rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
            nginx -t && nginx || true
            # Store nginx master PID for stop.sh
            pgrep -f "nginx: master" | head -1 > "$NGINX_PIDFILE" 2>/dev/null || true
        fi
    else
        echo "  [warn] nginx not installed — skipping LB. Install: apt install -y nginx"
    fi
fi

echo ""
echo "=== Done ==="
echo "  Direct (backend):"
echo "    GPU 0:  embed → :18000   llm → :18001   rerank → :18002"
echo "    GPU 1:  embed → :18010   llm → :18011   rerank → :18012"
if [ -f "$NGINX_PIDFILE" ] && kill -0 "$(cat "$NGINX_PIDFILE")" 2>/dev/null; then
    echo "  nginx LB (frontend):"
    echo "    :8000 → embedding   :8001 → llm   :8002 → reranker"
fi
echo ""
echo "  bash status.sh       # health check"
echo "  bash stop.sh         # kill all"
