# vLLM Serving Stack

OpenAI-compatible **LLM + Embedding + Reranker**, fully Dockerized. Ba cách triển khai
tuỳ theo tài nguyên sẵn có — cùng port, cùng API, client không cần đổi code.

| Port | Service | Endpoint |
|------|---------|----------|
| **8001** | LLM | `POST /v1/chat/completions` |
| **8000** | Embedding | `POST /v1/embeddings` |
| **8002** | Reranker | `POST /v1/rerank` |
| **3000** | Grafana (monitoring) | — |

---

## Chọn môi trường

```
Có 2× RTX 6000 Pro (96 GB, Blackwell)?  →  Môi trường 1: GPU (production)
Chỉ có Colab / cloud GPU đơn?           →  Môi trường 2: Colab
Máy thường, không GPU?                  →  Môi trường 3: No-GPU + DeepSeek API
```

---

## Môi trường 1 — GPU (2× RTX 6000 Pro, production)

Chạy toàn bộ 3 model local trên 2 GPU Blackwell. nginx load-balance 2 replica → ~2× throughput + HA.

```
GPU 0 ── llm-0 (NVFP4) ── embedding-0 ── reranker-0  ┐
                                                      ├─► nginx LB ─► clients
GPU 1 ── llm-1 (NVFP4) ── embedding-1 ── reranker-1  ┘
```

| Model | Spec |
|-------|------|
| LLM | `nvidia/Qwen3.6-35B-A3B-NVFP4` — NVFP4 native Blackwell (SM120), FP8 KV cache |
| Embedding | `BAAI/bge-m3` — multilingual, 1024-dim, fp16 |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` — causal-LM reranker, pooling runner |

**VRAM mỗi card (96 GB):**
```
LLM      (NVFP4 ~20 GB weights + FP8 KV)  util 0.80  ≈ 77 GB
Embedding (bge-m3, fp16)                  util 0.05  ≈  5 GB
Reranker  (Qwen3-Reranker, fp16)          util 0.06  ≈  6 GB
                                          ─────────────────
                                          ≈ 88 GB / 96 GB
```

**Khởi động:**
```bash
# 1. Host prep — Docker + NVIDIA Container Toolkit (chạy 1 lần, root)
bash setup.sh

# 2. Cấu hình
cp .env.example .env
nano .env          # set API_KEY, HF_TOKEN

# 3. Lên toàn bộ stack (6 vLLM replica + nginx + monitoring)
docker compose up -d

# 4. Kiểm tra
bash status.sh
docker compose logs -f llm-0    # lần đầu tải model ~20 GB
```

**Thao tác thường dùng:**
```bash
docker compose ps
docker compose logs -f llm-0
docker compose restart llm-1
docker compose down
bash status.sh
```

**Files chính:**
```
docker-compose.yml          ← toàn bộ stack (YAML anchor: *vllm-common, *gpu0/gpu1)
.env.example                ← biến cấu hình
setup.sh / status.sh
nginx/default.conf.template ← LB + auth (envsubst khi boot)
monitoring/                 ← Prometheus · DCGM · cAdvisor · Loki · Grafana
```

---

## Môi trường 2 — Colab (single GPU qua Cloudflare tunnel)

Dùng khi chỉ có GPU đơn từ Google Colab, Kaggle, hoặc cloud free-tier.
Notebook **tự dò GPU → tự chọn model phù hợp → serve 3 dịch vụ → mở Cloudflare tunnel → in public URL**.

| GPU | Model LLM tự chọn |
|-----|-------------------|
| H100 (80 GB) | Qwen3-30B-A3B FP8 |
| A100-80 GB | Qwen3-30B-A3B int4 |
| A100-40 GB / 48 GB | Qwen3-30B-A3B int4 |
| 24 GB (3090/4090) | Qwen3-14B int4 |
| 16 GB / T4 | Qwen3-7B int4 |

Embedding và Reranker luôn là `BAAI/bge-m3` + `Qwen/Qwen3-Reranker-0.6B`, serve tuần tự (không tràn VRAM).

**Khởi động:**
1. Mở notebook `colab/00_auto_serve_anygpu.ipynb` trên Colab
2. Chạy một cell duy nhất — tự động hoàn toàn
3. Notebook in ra URL dạng `https://xxxx.trycloudflare.com`
4. Dùng URL đó như `LLM_URL` / `EMB_URL` / `RERANK_URL`

**Files:**
```
colab/00_auto_serve_anygpu.ipynb   ← notebook tự dò GPU, dùng cho mọi loại
colab/serve_a100.ipynb             ← tuned cho A100
colab/serve_h100.ipynb             ← tuned cho H100
```

---

## Môi trường 3 — No-GPU (CPU + DeepSeek API)

Dùng khi không có GPU. LLM gọi qua **DeepSeek API** (cloud), Embedding + Reranker chạy **local trên CPU**.
Cùng port, cùng API — client không đổi gì.

```
CPU server (không cần GPU)

  embedding  (BAAI/bge-m3, CPU)          → :8000
  litellm    (proxy → DeepSeek API)      → :8001   → api.deepseek.com
  reranker   (Qwen3-Reranker-0.6B, CPU) → :8002
                    ↕
              nginx :8000/8001/8002 → clients
```

| Service | Image | Ghi chú |
|---------|-------|---------|
| Embedding | `secai-embedding:bge-m3` (build local) | SentenceTransformer, dim=1024 |
| LLM proxy | `secai-litellm:local` (build local) | LiteLLM → DeepSeek V4 Pro |
| Reranker | `secai-reranker:qwen3` (build local) | CausalLM yes/no logit scoring |

**Model LLM hỗ trợ (qua LiteLLM — `litellm/config.yml`):**
| Gọi với `model` | Thực tế |
|---|---|
| `deepseek-v4-pro` | DeepSeek V4 Pro — reasoning, coding |
| `deepseek-v4-flash` | DeepSeek V4 Flash — nhanh, rẻ hơn |
| `deepseek-chat` | alias → V4 Pro (deprecated 2026-07-24) |
| `deepseek-reasoner` | alias → V4 Pro (deprecated 2026-07-24) |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | alias → V4 Pro (backward compat) |

**Khởi động:**
```bash
# 1. Setup host — chỉ Docker, không cần NVIDIA toolkit
bash setup-cpu.sh

# 2. Cấu hình — điền DeepSeek API key
cp .env.cpu.example .env.cpu
nano .env.cpu     # set DEEPSEEK_API_KEY=sk-...

# 3. Build image + start (lần đầu build ~5 phút, download model ~3.6 GB)
docker compose -f docker-compose-cpu.yml --env-file .env.cpu up -d --build

# 4. Kiểm tra
bash status-cpu.sh
```

**Nếu kết nối với backend trên cùng máy** (ví dụ secai-allinone):
```bash
# Join 3 container vào network của backend rồi cập nhật URL
docker network connect <backend-network> embedding
docker network connect <backend-network> litellm
docker network connect <backend-network> reranker

# Backend gọi bằng container name:
# LLM   → http://litellm:8001/v1
# Embed → http://embedding:8000
# Rerank→ http://reranker:8002
```

**Files chính:**
```
docker-compose-cpu.yml          ← stack CPU (embedding + litellm + reranker + nginx + monitoring)
.env.cpu.example                ← template — thêm DEEPSEEK_API_KEY
setup-cpu.sh / status-cpu.sh
nginx/cpu.conf.template         ← single upstream (không LB)
litellm/config.yml              ← model mapping → DeepSeek API
reranker-cpu/
  Dockerfile                    ← python:3.11-slim + transformers (CPU torch)
  server.py                     ← CausalLM yes/no logit scoring cho Qwen3-Reranker
monitoring/
  prometheus/prometheus-cpu.yml ← scrape cAdvisor (không có DCGM)
  grafana/dashboards-cpu/       ← CPU/memory dashboard
```

---

## API usage (giống nhau cho cả 3 môi trường)

```bash
LLM_URL="http://<host>:8001"
EMB_URL="http://<host>:8000"
RERANK_URL="http://<host>:8002"
```

### LLM — Chat Completions

```bash
curl "$LLM_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1024
  }'
```

```python
from openai import OpenAI
client = OpenAI(base_url=f"{LLM_URL}/v1", api_key="sk-...")
resp = client.chat.completions.create(
    model="nvidia/Qwen3.6-35B-A3B-NVFP4",   # hoặc "deepseek-v4-pro" (môi trường 3)
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

### Embedding

```bash
curl "$EMB_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-m3", "input": ["text 1", "text 2"]}'
```

```python
client = OpenAI(base_url=f"{EMB_URL}/v1", api_key="sk-...")
r = client.embeddings.create(model="BAAI/bge-m3", input="Hello world")
print(len(r.data[0].embedding))   # 1024
```

### Reranker

```bash
curl "$RERANK_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "capital of France",
    "documents": ["Paris is the capital.", "Berlin is the capital of Germany."],
    "top_n": 2
  }'
```

---

## Security

nginx **injects** `Authorization: Bearer ${API_KEY}` upstream — clients không cần gửi key.
An toàn sau firewall/VPN. Nếu expose public IP: xoá dòng `proxy_set_header Authorization`
trong `nginx/default.conf.template` (hoặc `nginx/cpu.conf.template`) để vLLM/LiteLLM tự validate.
