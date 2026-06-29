#!/bin/bash
# Stop all vLLM processes started by serve.sh
set -e

echo "=== Stopping vLLM stack ==="
for pidfile in /tmp/vllm-a100-*.pid; do
    [ -f "$pidfile" ] || continue
    name=$(basename "$pidfile" .pid | sed 's/vllm-a100-//')
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        echo "  [stop] $name (pid $pid)"
        kill "$pid" 2>/dev/null || true
        # Wait for graceful shutdown
        for i in {1..10}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        # Force kill if still alive
        kill -9 "$pid" 2>/dev/null || true
    else
        echo "  [stale] $name (pid $pid)"
    fi
    rm -f "$pidfile"
done

# ── nginx ────────────────────────────────────────────────────────────────────
NGINX_PIDFILE="/tmp/vllm-a100-nginx.pid"
if [ -f "$NGINX_PIDFILE" ]; then
    pid=$(cat "$NGINX_PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "  [stop] nginx (pid $pid)"
        nginx -s quit 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
    rm -f "$NGINX_PIDFILE"
fi

echo "Done."
