#!/bin/bash
# Start all three vLLM services in the background.
# Logs  → PROJECT/logs/{llm,embedding,reranker}.log
# PIDs  → PROJECT/logs/pids/

set -e
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPTS_DIR/logs/pids"

start_service() {
    local name=$1
    local script=$2
    local pidfile="$SCRIPTS_DIR/logs/pids/${name}.pid"

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
echo "Logs:   tail -f $SCRIPTS_DIR/logs/{embedding,reranker,llm}.log"
echo "Status: bash $SCRIPTS_DIR/status.sh"
echo "Stop:   bash $SCRIPTS_DIR/stop_all.sh"
