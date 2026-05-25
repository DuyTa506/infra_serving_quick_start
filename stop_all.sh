#!/bin/bash
# Stop all vLLM services — kills by PID file AND by process name to catch orphans

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDDIR="$SCRIPT_DIR/logs/pids"

# Kill tracked PIDs
for name in embedding reranker llm; do
    pidfile="$PIDDIR/${name}.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "[STOP] $name (PID $pid)"
        else
            echo "[SKIP] $name not running"
        fi
        rm -f "$pidfile"
    else
        echo "[SKIP] $name — no pidfile"
    fi
done

# Kill any orphaned vllm/VLLM processes not tracked by pidfiles
ORPHANS=$(ps aux | grep -E "vllm serve|VLLM::Worker|VLLM::EngineCore" | grep -v grep | awk '{print $2}')
if [ -n "$ORPHANS" ]; then
    echo "[KILL] orphaned processes: $ORPHANS"
    echo "$ORPHANS" | xargs kill -9 2>/dev/null
fi

# Wait for GPU memory to clear
sleep 3
echo ""
echo "GPU memory after stop:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' '{printf "  GPU %s: %s / %s MiB\n",$1,$2,$3}'
