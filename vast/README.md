# Môi trường 1d — vast.ai (2× A100 80GB, template vLLM dựng sẵn)

Triển khai **LLM + Embedding + Reranker + nginx LB** lên **một instance vast.ai 2× A100 80GB**,
dùng **image vLLM dựng sẵn** (`vllm/vllm-openai:latest`) — **không build, không docker-compose,
không docker-in-docker**.

## Vì sao không dùng `docker compose` trên vast.ai?

vast.ai instance **bản thân đã là một Docker container** ([docs](https://docs.vast.ai/documentation/instances/templates/docker-environment)).
Chạy `docker compose` bên trong cần **Docker-in-Docker** (`--privileged`) — vast.ai **không cấp** cho
Docker instance thường; chỉ sản phẩm **VM** mới chạy được compose (ít máy hơn, boot chậm hơn). Nên ở
đây ta **"flatten"** `a100/docker-compose.yml` (6 replica + nginx) thành **tiến trình** trong 1 container,
giữ nguyên kiến trúc:

```
GPU 0 ── llm-0 (AWQ int4) ── embedding-0 ── reranker-0   (CUDA_VISIBLE_DEVICES=0)  ┐
                                                                                   ├─► nginx LB → public
GPU 1 ── llm-1 (AWQ int4) ── embedding-1 ── reranker-1   (CUDA_VISIBLE_DEVICES=1)  ┘
```

Mỗi card 1 replica đầy đủ (TP=1, zero NCCL). nginx `least_conn` 2 replica → ~2× throughput + HA.

## Model trên A100 (đã đổi quant — giống `a100/`)

A100 = Ampere (SM80): **không NVFP4, không FP8 W8A8**. Nên LLM dùng **AWQ int4 qua `awq_marlin`**:

| Service | Model | Ghi chú |
|---|---|---|
| LLM | `cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit` | `--quantization awq_marlin`, `--tool-call-parser hermes`, len 32768 |
| Embedding | `BAAI/bge-m3` | pooling, 1024-d, fp16 — **giữ nguyên** |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` | causal-LM reranker: `--runner pooling` + `--hf-overrides` + jinja — **giữ nguyên** |

VRAM mỗi card: `LLM 0.80 (~64) + embed 0.05 (~4) + rerank 0.06 (~5) ≈ 73 / 80 GB`.

## Các bước triển khai

### 1. Thuê instance
- Console vast.ai → tạo **Custom template** với image `vllm/vllm-openai:latest`
  (đừng dùng template "vLLM" one-click — nó auto-serve sẵn 1 model, chiếm GPU + port 8000).
- **Launch mode: SSH** ⟵ quan trọng: SSH mode *thay entrypoint gốc* → image **không auto-serve**.
- **Filter máy:** `num_gpus = 2`, GPU = **A100 SXM4/PCIE 80GB**, **Disk ≥ 80 GB** (image vLLM ~10GB + model ~21GB).
  CUDA ≥ 12.8 (A100 nào cũng đủ cho int4 Marlin).

### 2. Mở port (ô `-p` trong template)
```
-p 8001:8001 -p 8000:8000 -p 8002:8002
```
vast.ai map mỗi port nội bộ → **external port ngẫu nhiên trên IP công cộng DÙNG CHUNG**.
Sau khi máy lên, bấm **"IP Port Info"** để xem `PUBLIC_IP:EXTERNAL_PORT → 8001/...`.

### 3. Env vars (ô Environment Variables)
- **`API_KEY`** — bắt buộc (bearer token, client phải gửi).
- **`HF_TOKEN`** — *tuỳ chọn*: 3 model mặc định đều public nên không cần; chỉ set nếu đổi sang model gated hoặc muốn tải nhanh hơn.

### 4. Deploy = `git clone` → chạy `on_start.sh`

**Cách A — SSH vào rồi chạy tay (khuyến nghị: xem được log, debug dễ):**
```bash
ssh -p <PORT> root@<HOST>        # đúng lệnh ở nút "Connect" của instance
git clone --depth 1 -b justtuananh \
  https://github.com/DuyTa506/infra_serving_quick_start.git /opt/iss
API_KEY=sk-secai2026 bash /opt/iss/vast/on_start.sh
```
> `on_start.sh` tự suy ra repo dir từ vị trí của nó (clone đâu cũng chạy, khỏi set `REPO_DIR`).
> Muốn chạy nền + đóng SSH vẫn tiếp tục: `... setsid bash /opt/iss/vast/on_start.sh >/var/log/onstart_run.log 2>&1 </dev/null &`

**Cách B — On-start Script tự động (deploy khỏi cần SSH):** đặt `API_KEY`(+`HF_TOKEN`) ở ô env, rồi dán vào ô **On-start Script**:
```bash
set -e
git clone --depth 1 -b justtuananh \
  https://github.com/DuyTa506/infra_serving_quick_start.git /opt/iss || true
bash /opt/iss/vast/on_start.sh        # API_KEY/HF_TOKEN lấy từ env template
```

`on_start.sh` làm gì: cài nginx → launch 6 tiến trình vLLM (pin GPU 0/1, serve-args = `a100/`,
**quantization auto-detect** từ config model — compressed-tensors/AWQ đều chạy Marlin int4 trên Ampere)
→ **health-gate tuần tự trong mỗi GPU** (llm → embed → rerank, tránh OOM) → sinh nginx LB config động
→ start nginx. Lần đầu tải ~21 GB model có thể mất vài phút.

## Kiểm tra (sau khi instance lên)

```bash
# 1) Log on-start
tail -f /var/log/vast_onstart.log
tail -f /var/log/vllm/llm-0.log        # log từng replica: {llm,embedding,reranker}-{0,1}.log

# 2) Trong instance — qua nginx LB (client tự gửi Bearer)
curl -s -H "Authorization: Bearer $API_KEY" localhost:8001/v1/models | jq .   # → AWQ-4bit model id
curl -s -H "Authorization: Bearer $API_KEY" localhost:8000/v1/models          # embedding
curl -s -H "Authorization: Bearer $API_KEY" localhost:8002/v1/models          # reranker

# 3) GPU: 2 card đều có llm+embed+rerank, mỗi card ~73 GB
nvidia-smi

# 4) Từ ngoài — lấy PUBLIC_IP:PORT ở "IP Port Info"
LLM_URL=http://PUBLIC_IP:EXTERNAL_PORT python bench.py     # bench.py tự gửi Bearer ${API_KEY}
```

Reranker:
```bash
curl -s localhost:8002/v1/rerank -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"Qwen/Qwen3-Reranker-0.6B","query":"capital of France",
       "documents":["Paris is the capital.","Berlin is the capital of Germany."],"top_n":2}'
```

## ⚠️ Bảo mật

- Port lộ trên **IP công cộng dùng chung của vast.ai**, **không có TLS tự động**. Stack này **giữ
  `--api-key`** (nginx **không** inject auth) — ai gọi cũng phải có `Authorization: Bearer $API_KEY`,
  vLLM tự validate. Đặt `API_KEY` đủ mạnh.
- Muốn HTTPS thật / ẩn IP: thêm **Cloudflare tunnel** hoặc **Caddy** trỏ vào `localhost:8001` (tham
  khảo `colab/` đã dùng `cloudflared`). Ngoài phạm vi script này.

## Khác biệt so với các môi trường khác

- **Không có Grafana/Prometheus/Loki** (monitoring cũ là compose-based). Quan sát tạm bằng vLLM
  `/metrics` từng replica + `nvidia-smi`. Cần dashboard → dựng thêm sau (out of scope).
- Serve-args LLM/embed/rerank **đồng bộ** với `a100/docker-compose.yml`. Đổi model → sửa env
  `LLM_MODEL` / `EMBED_MODEL` / `RERANK_MODEL` (reranker đổi id là **chưa đủ** cho họ Qwen3-Reranker —
  serve-args đặc thù đã nằm sẵn trong `on_start.sh`).

## Files

```
vast/
  on_start.sh          ← on-start script: launch 6 vLLM proc + health-gate + nginx LB (sinh config động)
  nginx.conf.template  ← REFERENCE cấu hình 2-GPU (on_start.sh sinh config live, adapt theo số GPU)
  .env.example         ← mọi env var + default
  README.md            ← file này
```
