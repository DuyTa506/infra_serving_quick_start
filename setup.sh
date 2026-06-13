#!/bin/bash
# One-shot host prep for the vLLM serving stack on a standard Docker host
# (bare-metal or cloud VM) with 2x RTX 6000 Pro Blackwell.
#
# Installs: Docker Engine + Compose plugin, NVIDIA Container Toolkit, and wires
# the nvidia runtime into dockerd. After this, just `docker compose up -d`.
#
# Run as root on Ubuntu 22.04 / 24.04 with the NVIDIA driver already installed.
set -e

echo "=========================================="
echo " vLLM Stack — host setup (Docker + NVIDIA)"
echo "=========================================="

# ── 1. Docker Engine + Compose plugin ─────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[1/4] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
else
    echo "[1/4] Docker present ($(docker --version))"
fi

# ── 2. NVIDIA Container Toolkit ───────────────────────────────────────────────
if ! command -v nvidia-ctk &>/dev/null; then
    echo "[2/4] Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y nvidia-container-toolkit
else
    echo "[2/4] NVIDIA Container Toolkit present"
fi

# ── 3. Wire nvidia runtime into dockerd ───────────────────────────────────────
echo "[3/4] Configuring nvidia runtime for Docker..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker 2>/dev/null || service docker restart 2>/dev/null || true

# ── 4. Workspace ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[4/4] Preparing workspace..."
mkdir -p "$SCRIPT_DIR/models"
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

echo ""
echo " Verifying GPU visibility inside a container..."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi \
    --query-gpu=index,name,memory.total --format=csv,noheader,nounits \
    | awk -F', ' '{printf "   GPU %s: %s (%s MiB)\n",$1,$2,$3}' \
    || echo "   (could not run GPU container — check driver + toolkit)"

cat <<'EOF'

==========================================
 Setup complete.

 Next:
   cp .env.example .env      # set API_KEY + HF_TOKEN
   docker compose up -d      # build/pull + start everything
   bash status.sh            # health + GPU + container check

 Public endpoints (nginx load-balanced over both GPUs):
   8000  embedding   POST /v1/embeddings
   8001  llm         POST /v1/chat/completions   (OpenAI-compatible)
   8002  reranker    POST /v1/rerank  |  /v1/score

 Observability (one UI):
   3000  Grafana     (admin / $GRAFANA_PASSWORD)  GPU dashboards + all logs
         Prometheus / DCGM / cAdvisor / Loki run internally behind Grafana.
==========================================
EOF
