# vLLM Serving Stack — 2× NVIDIA A100 80GB (Ampere)

Version Docker cho **2× A100 80GB**, phục vụ đúng 3 service như stack chính
(**LLM + embedding + reranker**) sau một nginx load balancer, kèm monitoring tập
trung qua Grafana. **`docker compose up -d` là chạy được hết.**

> Đây là một **version mới hoàn toàn**, song song với:
> - **repo root** — 2× RTX 6000 Pro Blackwell (NVFP4)
> - **`colab/`** — single-GPU notebook (Cloudflare tunnel)
> - **`a100/`** (thư mục này) — 2× A100 80GB Docker (AWQ int4)

## Vì sao A100 cần bản riêng (không xài thẳng compose của root)

A100 là **Ampere (SM80)**, khác hẳn Blackwell:

- **NVFP4 không chạy được.** `--quantization modelopt` (NVFP4) yêu cầu compute
  capability **≥ 89** (Ada/Hopper/Blackwell). A100 là SM80 → model
  `nvidia/Qwen3.6-35B-A3B-NVFP4` của root **không load** trên A100.
- **FP8 W8A8 là Hopper/Ada-only.** Trên Ampere FP8 chỉ là weight-only (W8A16),
  không tăng tốc compute; `--kv-cache-dtype fp8` chỉ là emulation → bỏ.
- **AWQ/GPTQ chạy nhanh trên Ampere qua Marlin.** Nên LLM ở đây là bản **AWQ int4**
  của `Qwen3-30B-A3B-Instruct-2507` phục vụ với `--quantization awq_marlin`.

Khác biệt **duy nhất** so với root nằm ở khối `&llm-cmd`:

| | root (RTX 6000 Blackwell) | a100 (A100 Ampere) |
|---|---|---|
| LLM | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit` |
| quant | `modelopt` (NVFP4) | `awq_marlin` (int4) |
| KV cache | `fp8` | (mặc định fp16/bf16) |
| speculative | MTP | — |
| reasoning parser | `qwen3` (thinking) | — (2507-Instruct non-thinking) |
| tool-call parser | `qwen3_coder` | `hermes` |

Embedding (`BAAI/bge-m3`) và reranker (`Qwen/Qwen3-Reranker-0.6B`) **giống hệt** root.

## Kiến trúc (giữ nguyên triết lý repo)

Mỗi GPU chạy **một replica đầy đủ** (`--tensor-parallel-size 1`), nginx load-balance
2 replica → ~2× throughput + HA. Model int4 (~18 GB) thừa sức nằm gọn 1 card nên
**không cần tensor-parallel** kể cả khi A100 có NVLink → zero NCCL cross-GPU.

```
GPU 0 ── llm-0 (AWQ int4) ── embedding-0 ── reranker-0  ┐
                                                        ├─► nginx LB ─► clients
GPU 1 ── llm-1 (AWQ int4) ── embedding-1 ── reranker-1  ┘
```

| Public port | Service | Model | Endpoint |
|------|---------|-------|----------|
| 8001 | LLM | Qwen3-30B-A3B-Instruct-2507 (AWQ int4) | `POST /v1/chat/completions` |
| 8000 | Embedding | BAAI/bge-m3 (multilingual, 1024-d) | `POST /v1/embeddings` |
| 8002 | Reranker | Qwen/Qwen3-Reranker-0.6B | `POST /v1/rerank` · `/v1/score` |
| 3000 | **Grafana** | — | GPU dashboards (DCGM) + logs (Loki) |

Prometheus / DCGM / cAdvisor / Loki chạy **internal-only**, Grafana query qua mạng
`serving`. Cấu hình nginx + monitoring + reranker template + HF cache **tái dùng**
từ repo root qua bind-mount tương đối (`../`), nên không bị drift.

## Quick start

```bash
cd a100

# 1. Host prep (Docker + NVIDIA Container Toolkit). Chạy 1 lần, quyền root.
bash setup.sh

# 2. Config
cp .env.example .env
nano .env            # set API_KEY và HF_TOKEN

# 3. Launch (6 vLLM replica + nginx + monitoring)
docker compose up -d

# 4. Theo dõi
bash status.sh
docker compose logs -f llm-0      # lần đầu tải weights (~18 GB int4)
```

## GPU memory layout (mỗi card 80 GB)

```
LLM  (Qwen3-30B-A3B AWQ int4, ~18 GB weights + KV)   util 0.80  ≈ 64 GB
Embedding (bge-m3, fp16)                              util 0.05  ≈  4 GB
Reranker  (Qwen3-Reranker-0.6B, fp16)                util 0.06  ≈  5 GB
                                                     ─────────────────
                                                     ≈ 73 GB / 80 GB
```

int4 chừa rất nhiều VRAM cho KV → có thể nâng `LLM_MAX_MODEL_LEN` (tới 262144) hoặc
`--gpu-memory-utilization` của LLM nếu cần context/throughput cao hơn.

## API usage

Giống stack root, chỉ khác `model` của LLM. nginx inject API key nên client gọi
thẳng port mà không cần gửi key.

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
       "messages":[{"role":"user","content":"Hello!"}]}'

curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":["text 1","text 2"]}'        # → 1024-d

curl http://localhost:8002/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Reranker-0.6B","query":"capital of France",
       "documents":["Paris is the capital of France.","Berlin is the capital of Germany."]}'
```

## Operations

```bash
docker compose up -d            # start
docker compose ps               # what's running
docker compose logs -f llm-0    # follow one replica
docker compose restart llm-1    # restart a single replica
docker compose down             # stop (giữ models + volumes)
bash status.sh                  # health + GPU + endpoint
```

## Lưu ý quan trọng

- **Dùng chung port + tên container với stack root** (8000/8001/8002/3000, `llm-0`…).
  Chỉ chạy **một trong hai** trên cùng một host — điều này luôn đúng vì 1 host chỉ
  có 1 loại card (A100 *hoặc* RTX 6000).
- **HF cache dùng chung `../models`** → `bge-m3` và reranker không tải lại nếu stack
  root đã pull. Chỉ bản LLM (AWQ int4) là tải mới.
- **Auth:** nginx inject key, port mở cho ai reach được → để sau firewall. Muốn bắt
  client gửi key: xóa các dòng `proxy_set_header Authorization` trong
  `../nginx/default.conf.template` (chung với root).
- **A100 40GB:** không khuyến nghị cho bản này — phải hạ `LLM_MAX_MODEL_LEN` (~16384)
  và xác nhận int4 + KV + 2 model nhỏ vừa 40 GB.
