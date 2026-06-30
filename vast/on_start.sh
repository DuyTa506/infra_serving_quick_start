#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  vast.ai ON-START SCRIPT — LLM + Embedding + Reranker + nginx LB on 2× A100 80GB
# ─────────────────────────────────────────────────────────────────────────────
#
#  Runs INSIDE a single vast.ai instance whose image is `vllm/vllm-openai:latest`
#  (vast.ai's ready-made vLLM template). vast.ai instances ARE Docker containers,
#  so we DON'T run docker-compose / docker-in-docker here. Instead we "flatten"
#  the repo's a100/docker-compose.yml into plain processes inside this one
#  container — same architecture, no DinD:
#
#    GPU 0 ── llm-0 ── embedding-0 ── reranker-0   (CUDA_VISIBLE_DEVICES=0)  ┐
#                                                                            ├─► nginx LB → public
#    GPU 1 ── llm-1 ── embedding-1 ── reranker-1   (CUDA_VISIBLE_DEVICES=1)  ┘
#
#  Each GPU runs ONE full replica (TP=1, zero cross-GPU NCCL — same design rule as
#  the repo). nginx least_conn-balances the replicas → ~2× throughput + HA.
#  LLM is AWQ int4 via awq_marlin (A100 = Ampere/SM80 has no NVFP4 / no FP8 W8A8) —
#  serve-args are copied verbatim from a100/docker-compose.yml.
#
#  Auth: we KEEP vLLM's --api-key (clients send `Authorization: Bearer $API_KEY`).
#  nginx does NOT inject auth — vast.ai ports are exposed on a SHARED public IP with
#  no TLS, so an open keyless endpoint would be reachable by anyone.
#
#  Use as the vast.ai template "On-start Script" (Launch mode: SSH). Required env:
#    API_KEY   (shared bearer token)   HF_TOKEN (HuggingFace, for faster/gated pulls)
#  Everything else has sane defaults below; override via the template's env vars.
#
#  Logs:    /var/log/vast_onstart.log   (this script)
#           $LOG_DIR/{llm,embedding,reranker}-<gpu>.log   (per replica)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LOG_MAIN="/var/log/vast_onstart.log"
mkdir -p "$(dirname "$LOG_MAIN")"
exec > >(tee -a "$LOG_MAIN") 2>&1
echo "═══ vast on_start @ $(date -u '+%Y-%m-%dT%H:%M:%SZ') ═══"

# ── Config (env-driven; defaults mirror a100/.env.example) ───────────────────
API_KEY="${API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"

LLM_MODEL="${LLM_MODEL:-cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"

LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-32768}"
LLM_GPU_UTIL="${LLM_GPU_UTIL:-0.80}"
EMBED_GPU_UTIL="${EMBED_GPU_UTIL:-0.05}"
RERANK_GPU_UTIL="${RERANK_GPU_UTIL:-0.06}"

# Public ports exposed via the vast.ai template's `-p` mappings (LB endpoints).
LLM_PUB_PORT="${LLM_PUB_PORT:-8001}"
EMBED_PUB_PORT="${EMBED_PUB_PORT:-8000}"
RERANK_PUB_PORT="${RERANK_PUB_PORT:-8002}"

# Internal per-replica ports (localhost only; never published). Replica index g
# (0-based GPU id) is appended via the port bases below: e.g. llm gpu0=18101, gpu1=18102.
LLM_PORT_BASE="${LLM_PORT_BASE:-18101}"
EMBED_PORT_BASE="${EMBED_PORT_BASE:-18001}"
RERANK_PORT_BASE="${RERANK_PORT_BASE:-18201}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"   # seconds to wait per replica /health (cold pull is slow)

# Repo (for the reranker jinja template + this dir). Override REPO_URL if you forked.
REPO_URL="${REPO_URL:-https://github.com/DuyTa506/infra_serving_quick_start.git}"
REPO_REF="${REPO_REF:-master}"
REPO_DIR="${REPO_DIR:-/opt/infra_serving_quick_start}"

# Persisted HuggingFace cache on the instance disk (survives container restart).
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
LOG_DIR="${LOG_DIR:-/var/log/vllm}"
mkdir -p "$HF_HOME" "$LOG_DIR"

if [ -z "$API_KEY" ]; then
  echo "FATAL: API_KEY is empty. Set it in the vast.ai template env vars." >&2
  exit 1
fi
[ -n "$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" HF_TOKEN="$HF_TOKEN"

# ── 1) System deps (nginx for LB, envsubst, git, curl) ───────────────────────
echo "── installing nginx + git + gettext-base + curl ──"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nginx git gettext-base curl >/dev/null
# vLLM image already provides python3 + vllm; don't reinstall.

# ── 2) Fetch repo (reranker jinja template lives here) ───────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "── cloning $REPO_URL ($REPO_REF) → $REPO_DIR ──"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
else
  echo "── repo present at $REPO_DIR; pulling ──"
  git -C "$REPO_DIR" pull --ff-only || true
fi
RERANK_TEMPLATE="$REPO_DIR/templates/qwen3_reranker.jinja"
[ -f "$RERANK_TEMPLATE" ] || { echo "FATAL: missing $RERANK_TEMPLATE" >&2; exit 1; }

# ── 3) GPU count ─────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$NUM_GPUS" -ge 1 ] 2>/dev/null || { echo "FATAL: no NVIDIA GPU detected." >&2; exit 1; }
echo "── detected ${NUM_GPUS} GPU(s); launching one full replica per GPU ──"

PYBIN="$(command -v python3 || command -v python)"
RERANK_HF_OVERRIDES='{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'

# ── helpers ──────────────────────────────────────────────────────────────────
start_proc() {  # name port gpu  -- remaining args are the vLLM serve-args
  local name="$1" port="$2" gpu="$3"; shift 3
  local logf="$LOG_DIR/${name}.log"
  echo "▶ ${name}  GPU=${gpu}  :${port}  → ${logf}"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$HF_HOME" \
    nohup "$PYBIN" -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 --port "$port" --api-key "$API_KEY" --trust-remote-code "$@" \
      >"$logf" 2>&1 < /dev/null &
  echo "$!" >> "$LOG_DIR/pids.txt"   # actual vLLM PID (nohup keeps it alive past logout)
}

wait_healthy() {  # port name
  local port="$1" name="$2" deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "  ✓ ${name} healthy on :${port}"; return 0
    fi
    sleep 3
  done
  echo "FATAL: ${name} (:${port}) not healthy after ${HEALTH_TIMEOUT}s — see $LOG_DIR/${name}.log" >&2
  return 1
}

# LLM serve-args == &llm-cmd in a100/docker-compose.yml (AWQ int4 / awq_marlin, Ampere fast path)
llm_args() { echo \
  --model "$LLM_MODEL" --quantization awq_marlin --tensor-parallel-size 1 \
  --max-model-len "$LLM_MAX_MODEL_LEN" --max-num-batched-tokens 16384 \
  --gpu-memory-utilization "$LLM_GPU_UTIL" --max-num-seqs 128 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes ; }

# Embedding serve-args == &embed-cmd (bge-m3, pooling, 1024-d). Reranker == &rerank-cmd
# (CAUSAL-LM reranker: needs --runner pooling + --hf-overrides + jinja, not just the id).

# ── 4) Launch replicas, GPU by GPU, sequential within a GPU (avoid OOM race) ──
: > "$LOG_DIR/pids.txt"
for g in $(seq 0 $((NUM_GPUS - 1))); do
  echo "──────── GPU ${g} ────────"
  start_proc "llm-${g}"       "$((LLM_PORT_BASE + g))"    "$g" $(llm_args)
  wait_healthy "$((LLM_PORT_BASE + g))" "llm-${g}"

  start_proc "embedding-${g}" "$((EMBED_PORT_BASE + g))"  "$g" \
    --model "$EMBED_MODEL" --runner pooling --dtype float16 \
    --max-model-len 512 --gpu-memory-utilization "$EMBED_GPU_UTIL"
  wait_healthy "$((EMBED_PORT_BASE + g))" "embedding-${g}"

  start_proc "reranker-${g}"  "$((RERANK_PORT_BASE + g))" "$g" \
    --model "$RERANK_MODEL" --runner pooling \
    --hf-overrides "$RERANK_HF_OVERRIDES" --chat-template "$RERANK_TEMPLATE" \
    --dtype float16 --max-model-len 4096 --gpu-memory-utilization "$RERANK_GPU_UTIL"
  wait_healthy "$((RERANK_PORT_BASE + g))" "reranker-${g}"
done

# ── 5) Generate + start nginx LB (least_conn over all replicas; NO auth injection) ──
echo "── generating nginx LB config (/etc/nginx/conf.d/vllm.conf) ──"
gen_upstream() {  # name base_port
  local name="$1" base="$2"
  echo "upstream ${name} {"
  echo "    least_conn;"
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    echo "    server 127.0.0.1:$((base + g)) max_fails=3 fail_timeout=30s;"
  done
  echo "    keepalive 32;"
  echo "}"
}
gen_server() {  # listen_port upstream_name read_timeout
  cat <<EOF
server {
    listen ${1};
    client_max_body_size 64M;
    location / {
        proxy_pass http://${2};
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        # NOTE: we deliberately do NOT inject Authorization here — vLLM validates
        # the client's own Bearer token (--api-key), safe on a public IP.
        proxy_buffering off;
        chunked_transfer_encoding on;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout ${3};
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 2;
    }
}
EOF
}
{
  echo "# Auto-generated by vast/on_start.sh — do not edit by hand."
  echo "# Canonical 2-GPU shape is documented in vast/nginx.conf.template."
  gen_upstream llm_backend       "$LLM_PORT_BASE"
  gen_upstream embedding_backend "$EMBED_PORT_BASE"
  gen_upstream reranker_backend  "$RERANK_PORT_BASE"
  gen_server "$LLM_PUB_PORT"    llm_backend       600s   # LLM streaming
  gen_server "$EMBED_PUB_PORT"  embedding_backend 60s
  gen_server "$RERANK_PUB_PORT" reranker_backend  60s
} > /etc/nginx/conf.d/vllm.conf

# Drop the stock default vhost (it would also listen on :80, harmless, but keep clean).
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
nginx -t
nginx -s reload 2>/dev/null || nginx   # reload if already running, else start

echo ""
echo "✅ stack up — ${NUM_GPUS} replica(s) per service, nginx LB:"
echo "   LLM       :${LLM_PUB_PORT}   (${LLM_MODEL})"
echo "   Embedding :${EMBED_PUB_PORT}   (${EMBED_MODEL})"
echo "   Reranker  :${RERANK_PUB_PORT}   (${RERANK_MODEL})"
echo "   Map these ports in the vast.ai template (-p) and read PUBLIC_IP:PORT from 'IP Port Info'."
echo "═══ on_start done @ $(date -u '+%Y-%m-%dT%H:%M:%SZ') ═══"
