#!/bin/bash
# Start all three vLLM services in the background.
# Logs  → /workspace/logs/{llm,embedding,reranker}.log
# PIDs  → /workspace/logs/pids/

set -e
mkdir -p /workspace/logs/pids
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

start_service() {
    local name=$1
    local script=$2
    local pidfile=/workspace/logs/pids/${name}.pid

    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "[SKIP] $name already running (PID $(cat "$pidfile"))"
        return
    fi

    nohup bash "$script" > /dev/null 2>&1 &
    echo $! > "$pidfile"
    echo "[START] $name  PID $!"
}

# Small models first so GPU 0/1 memory is claimed before LLM tries to share them
start_service embedding "$SCRIPTS_DIR/start_embedding.sh"
start_service reranker  "$SCRIPTS_DIR/start_reranker.sh"
sleep 2
start_service llm       "$SCRIPTS_DIR/start_llm.sh"

echo ""
echo "Logs:   tail -f /workspace/logs/{embedding,reranker,llm}.log"
echo "Status: bash $SCRIPTS_DIR/status.sh"
echo "Stop:   bash $SCRIPTS_DIR/stop_all.sh"
