#!/bin/bash
# Check health of all three vLLM services

BASE_URL="http://localhost"
declare -A SERVICES=(
    [embedding]="$BASE_URL:8000"
    [llm]="$BASE_URL:8001"
    [reranker]="$BASE_URL:8002"
)

PIDDIR=/workspace/logs/pids

echo "=== Process status ==="
for name in embedding llm reranker; do
    pidfile="$PIDDIR/${name}.pid"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "  $name  RUNNING (PID $(cat "$pidfile"))"
    else
        echo "  $name  STOPPED"
    fi
done

echo ""
echo "=== HTTP health ==="
for name in embedding llm reranker; do
    port=${SERVICES[$name]##*:}
    url="${SERVICES[$name]}/health"
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null)
    if [ "$status" = "200" ]; then
        echo "  :$port ($name)  OK"
    else
        echo "  :$port ($name)  $status (not ready yet or down)"
    fi
done

echo ""
echo "=== GPU usage ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s (%s)  %s/%s MiB  %s%% util\n",$1,$2,$3,$4,$5}'
