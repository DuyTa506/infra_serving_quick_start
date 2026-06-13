#!/bin/bash
# Health snapshot for the Dockerized vLLM stack: containers + HTTP + GPU.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Containers ==="
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null \
    || docker compose ps

echo ""
echo "=== HTTP health (via nginx LB) ==="
declare -A PORTS=( [embedding]=8000 [llm]=8001 [reranker]=8002 )
for name in embedding llm reranker; do
    port=${PORTS[$name]}
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$port/health" 2>/dev/null)
    [ "$status" = "200" ] && echo "  :$port ($name)  OK" || echo "  :$port ($name)  $status (warming up or down)"
done

echo ""
echo "=== Observability (single UI) ==="
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:3000/api/health" 2>/dev/null)
echo "  Grafana  http://localhost:3000  ($code)"
echo "  (Prometheus / DCGM / cAdvisor / Loki are internal — viewed inside Grafana)"

echo ""
echo "=== GPU usage ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "  GPU %s (%s)  %s/%s MiB  %s%% util\n",$1,$2,$3,$4,$5}' \
    || echo "  nvidia-smi not available on host"

echo ""
echo "Logs:  docker compose logs -f llm-0    |    Grafana: http://localhost:3000"
