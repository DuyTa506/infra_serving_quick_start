#!/bin/bash
# One-shot setup script for a fresh RunPod pod (Ubuntu 22.04/24.04, CUDA pre-installed).
# Run as root. After this completes, use start_all.sh to launch services.
#
# What this does:
#   1. Install Docker (with nvidia runtime configured)
#   2. Install NVIDIA Container Toolkit
#   3. Install vLLM (latest) for native serving
#   4. Create /workspace/models for persistent model cache
#   5. Make all scripts executable

set -e

echo "=========================================="
echo " vLLM Stack Setup"
echo "=========================================="

# ── 1. Docker ──────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[1/5] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
else
    echo "[1/5] Docker already installed ($(docker --version 2>/dev/null || echo 'daemon not running'))"
fi

# ── 2. NVIDIA Container Toolkit ───────────────────────────────────────────────
if ! command -v nvidia-ctk &>/dev/null; then
    echo "[2/5] Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y nvidia-container-toolkit
else
    echo "[2/5] NVIDIA Container Toolkit already installed"
fi

# ── 3. Configure Docker daemon (nvidia runtime + no iptables for RunPod DinD) ─
echo "[3/5] Configuring Docker daemon..."
cat > /etc/docker/daemon.json << 'EOF'
{
    "iptables": false,
    "bridge": "none",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia"
}
EOF

# Start dockerd if not running
if ! docker info &>/dev/null 2>&1; then
    echo "     Starting dockerd..."
    nohup dockerd 2>/tmp/dockerd.log &
    sleep 5
fi

# Persist dockerd start across reboots (no systemd on RunPod)
if [ ! -f /etc/rc.local ]; then
    echo '#!/bin/bash' > /etc/rc.local
    chmod +x /etc/rc.local
fi
grep -q "dockerd" /etc/rc.local || echo "nohup dockerd 2>/tmp/dockerd.log &" >> /etc/rc.local

# ── 4. Install vLLM ───────────────────────────────────────────────────────────
if python3 -c "import vllm" &>/dev/null 2>&1; then
    echo "[4/5] vLLM already installed ($(python3 -c 'import vllm; print(vllm.__version__)'))"
else
    echo "[4/5] Installing vLLM (cu129 wheel — compatible with CUDA 12.8+)..."
    VLLM_VER=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'].lstrip('v'))")
    ARCH=$(uname -m)
    pip install \
        "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VER}/vllm-${VLLM_VER}+cu129-cp38-abi3-manylinux_2_34_${ARCH}.whl" \
        --extra-index-url https://download.pytorch.org/whl/cu129
fi

# ── 5. Workspace setup ────────────────────────────────────────────────────────
echo "[5/5] Setting up /workspace..."
mkdir -p /workspace/models /workspace/logs/pids

# Make all scripts executable
chmod +x /workspace/*.sh

echo ""
echo "=========================================="
echo " Setup complete!"
echo ""
echo " GPU summary:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits \
    | awk -F', ' '{printf "   GPU %s: %s (%s MiB)\n",$1,$2,$3}'
echo ""
echo " Next steps:"
echo "   1. Create .env from template and set your API key + HF token:"
echo "      cp /workspace/.env.example /workspace/.env"
echo "      nano /workspace/.env   # set API_KEY and HF_TOKEN"
echo "   2. Start all services:"
echo "      bash /workspace/start_all.sh"
echo "   3. Check status:"
echo "      bash /workspace/status.sh"
echo ""
echo " Ports:"
echo "   8000 → BGE-Large embedding  (POST /v1/embeddings)"
echo "   8001 → Qwen3 LLM            (POST /v1/chat/completions)"
echo "   8002 → BGE-Reranker         (POST /v1/score  or  POST /v1/rerank)"
echo "=========================================="
