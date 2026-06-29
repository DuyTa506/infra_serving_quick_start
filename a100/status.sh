#!/bin/bash
# Health snapshot for the no-Docker vLLM stack on 2× A100.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Process status ──────────────────────────────────────────────────────────
echo "=== vLLM Processes ==="
for pidfile in /tmp/vllm-a100-*.pid; do
    [ -f "$pidfile" ] || continue
    name=$(basename "$pidfile" .pid | sed 's/vllm-a100-//')
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        printf "  %-12s pid %-8s  RUNNING\n" "$name" "$pid"
    else
        printf "  %-12s pid %-8s  DEAD\n" "$name" "$pid"
    fi
done
[ -z "$(ls /tmp/vllm-a100-*.pid 2>/dev/null)" ] && echo "  (no processes found)"

# ── HTTP health (direct — bypass nginx) ─────────────────────────────────────
echo ""
echo "=== HTTP Health (direct) ==="
declare -A ENDPOINTS=(
    [embed-0]="localhost:18000/health"
    [llm-0]="localhost:18001/health"
    [rerank-0]="localhost:18002/health"
    [embed-1]="localhost:18010/health"
    [llm-1]="localhost:18011/health"
    [rerank-1]="localhost:18012/health"
)
for name in embed-0 llm-0 rerank-0 embed-1 llm-1 rerank-1; do
    url="${ENDPOINTS[$name]}"
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://$url" 2>/dev/null || echo "000")
    [ "$status" = "200" ] && echo "  $name   OK ($url)" || echo "  $name   $status ($url)"
done

# ── nginx LB health ─────────────────────────────────────────────────────────
echo ""
echo "=== nginx LB ==="
NGINX_PIDFILE="/tmp/vllm-a100-nginx.pid"
if [ -f "$NGINX_PIDFILE" ] && kill -0 "$(cat "$NGINX_PIDFILE")" 2>/dev/null; then
    for label in "embed:8000" "llm:8001" "rerank:8002"; do
        name="${label%%:*}"
        port="${label##*:}"
        status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$port/health" 2>/dev/null || echo "000")
        [ "$status" = "200" ] && echo "  nginx → :$port ($name)  OK" || echo "  nginx → :$port ($name)  $status"
    done
else
    echo "  (nginx not running — direct backend ports only)"
fi

# ── API smoke test ──────────────────────────────────────────────────────────
echo ""
echo "=== API Smoke Test ==="
# LLM
if curl -s --max-time 5 http://localhost:18001/health | grep -q .; then
    resp=$(curl -s --max-time 15 http://localhost:18001/v1/chat/completions \
        -H "Authorization: Bearer sk-secai2026" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${LLM_MODEL:-palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4}"'","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' 2>/dev/null)
    if echo "$resp" | grep -q '"choices"'; then
        token=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:50])" 2>/dev/null)
        echo "  LLM (18001)  OK  →  $token"
    else
        echo "  LLM (18001)  ERR → $(echo "$resp" | head -c 200)"
    fi
else
    echo "  LLM (18001)  not ready"
fi

# Embedding
if curl -s --max-time 3 http://localhost:18000/health | grep -q .; then
    resp=$(curl -s --max-time 10 http://localhost:18000/v1/embeddings \
        -H "Authorization: Bearer sk-secai2026" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${EMBED_MODEL:-BAAI/bge-m3}"'","input":["Hello world"]}' 2>/dev/null)
    if echo "$resp" | grep -q '"embedding"'; then
        dim=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['data'][0]['embedding']))" 2>/dev/null)
        echo "  Embed (18000) OK  →  ${dim}-dim vector"
    else
        echo "  Embed (18000) ERR → $(echo "$resp" | head -c 200)"
    fi
else
    echo "  Embed (18000) not ready"
fi

# Reranker
if curl -s --max-time 3 http://localhost:18002/health | grep -q .; then
    resp=$(curl -s --max-time 10 http://localhost:18002/v1/rerank \
        -H "Authorization: Bearer sk-secai2026" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"'","query":"capital","documents":["Paris is the capital of France."]}' 2>/dev/null)
    if echo "$resp" | grep -q '"score"'; then
        echo "  Rerank (18002) OK"
    else
        echo "  Rerank (18002) ERR → $(echo "$resp" | head -c 200)"
    fi
else
    echo "  Rerank (18002) not ready"
fi

# ── GPU usage ───────────────────────────────────────────────────────────────
echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "  GPU %s (%s)  %s/%s MiB  %s%% util\n",$1,$2,$3,$4,$5}' \
    || echo "  nvidia-smi not available"

echo ""
echo "Logs:  tail -f $SCRIPT_DIR/logs/*.log"
