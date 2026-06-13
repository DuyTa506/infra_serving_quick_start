# vLLM Serving Stack — 2× RTX 6000 Pro (Blackwell)

OpenAI-compatible LLM + embedding + reranker, fully Dockerized, with centralized
GPU + log monitoring. **`docker compose up -d` brings up everything.**

## Why this layout

The RTX 6000 Pro Blackwell (96 GB GDDR7) is **PCIe-only — no NVLink P2P**.
Tensor-parallel across the two cards would bottleneck on the PCIe bus, so instead
each GPU runs a **full independent replica** and nginx load-balances across them:

```
GPU 0 ── llm-0 (NVFP4) ── embedding-0 ── reranker-0  ┐
                                                      ├─► nginx LB ─► clients
GPU 1 ── llm-1 (NVFP4) ── embedding-1 ── reranker-1  ┘
```

No cross-GPU NCCL traffic happens at all (TP=1 per replica), so the missing P2P
link is irrelevant — and you get ~2× LLM throughput plus HA: if one card's
replica dies, nginx routes to the other.

The LLM is **`nvidia/Qwen3.6-35B-A3B-NVFP4`** — NVFP4 is the native fast path on
Blackwell (SM120). NVFP4 weights are ~20 GB, and `--kv-cache-dtype fp8` halves
KV memory, so a single card has huge headroom for context + the two small models.

| Public port | Service | Model | Endpoint |
|------|---------|-------|----------|
| 8001 | LLM | nvidia/Qwen3.6-35B-A3B-NVFP4 (NVFP4, FP8 KV) | `POST /v1/chat/completions` |
| 8000 | Embedding | BAAI/bge-large-en-v1.5 | `POST /v1/embeddings` |
| 8002 | Reranker | BAAI/bge-reranker-v2-m3 | `POST /v1/rerank` · `/v1/score` |

**Observability is unified behind one port** — open **`:3000` (Grafana)** and
that's it. Prometheus, the DCGM GPU exporter, cAdvisor, and Loki run as
**internal-only** services on the Docker network; Grafana queries them by name,
so you never visit their ports.

| Observability port | Service |
|------|---------|
| 3000 | **Grafana** — the single pane: GPU dashboards + searchable logs of all containers |
| _internal_ | Prometheus (metrics) · DCGM (GPU) · cAdvisor (containers) · Loki (logs) |

---

## Quick start

```bash
# 1. Host prep (Docker + NVIDIA Container Toolkit). Run once, as root.
bash setup.sh

# 2. Configure
cp .env.example .env
nano .env            # set API_KEY and HF_TOKEN

# 3. Launch the whole stack (6 vLLM replicas + nginx + monitoring)
docker compose up -d

# 4. Watch it come up
bash status.sh
docker compose logs -f llm-0      # first boot downloads the model (~20 GB)
```

First boot pulls the NVFP4 weights into `./models/` (persisted, so restarts are
fast). Services report `:8001 (llm) OK` in `status.sh` once ready.

---

## Centralized monitoring (single pane of glass)

Open **http://localhost:3000** (Grafana, default `admin` / `admin` — change in
`.env`). The pre-provisioned **"vLLM Serving — GPU & Logs"** dashboard shows:

- **GPU panels** (NVIDIA DCGM): utilization, VRAM used, temperature, power, clocks — per card.
- **Logs panel** (Loki): live, searchable logs of every container. Use the
  **Container** dropdown to filter to `llm-0`, `embedding-1`, `nginx`, etc.

How the logs get there: **Promtail** tails every container via the Docker socket
and ships them to **Loki**; **Prometheus** scrapes **DCGM** (GPUs) and **cAdvisor**
(containers). Nothing to wire up — it's all in `docker compose up`.

Prefer the terminal? `docker compose logs -f` (all) or `docker compose logs -f llm-0`.

---

## API usage

**Auth:** nginx injects the shared key, so clients hit the public ports **without**
a token. (Keep the ports behind a firewall, or add your own auth — see Security.)

```bash
API_KEY="sk-..."                  # only needed if you add client-side auth
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
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url=f"{LLM_URL}/v1", api_key=API_KEY)

completion = client.chat.completions.create(
    model="nvidia/Qwen3.6-35B-A3B-NVFP4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(completion.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="nvidia/Qwen3.6-35B-A3B-NVFP4",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Embedding

```bash
curl "$EMB_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-large-en-v1.5", "input": ["text 1", "text 2"]}'
```

```python
client = OpenAI(base_url=f"{EMB_URL}/v1", api_key=API_KEY)
r = client.embeddings.create(model="BAAI/bge-large-en-v1.5", input="Hello world")
print(len(r.data[0].embedding))   # 1024
```

### Reranker

```bash
# Cohere-style list rerank
curl "$RERANK_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "capital of France",
    "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
    "top_n": 2
  }'

# Pairwise score
curl "$RERANK_URL/v1/score" \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-reranker-v2-m3", "text_1": "What is AI?", "text_2": "AI is artificial intelligence."}'
```

---

## Operations

```bash
docker compose up -d            # start everything
docker compose ps               # what's running
docker compose logs -f llm-0    # follow one replica
docker compose restart llm-1    # restart a single replica
docker compose down             # stop everything (keeps models + volumes)
docker compose pull             # update images, then `up -d` again
bash status.sh                  # health + GPU + endpoint check
```

### Scale / tune

- **Context length:** raise `LLM_MAX_MODEL_LEN` in `.env` (NVFP4 + FP8 KV leaves
  room toward the model's full 262 144 window).
- **VRAM split per card:** in `docker-compose.yml`, `--gpu-memory-utilization`
  is `0.80` (LLM) / `0.05` (embed) / `0.06` (rerank) → ~87 GB of 96 GB used.
- **Optional Blackwell tuning:** the model card also suggests
  `--moe-backend marlin` and `--attention-backend flashinfer`; add them to the
  `llm-cmd` block if your vLLM build supports them.

---

## GPU memory layout (per 96 GB card)

```
LLM  (NVFP4 ~20 GB weights + FP8 KV cache)   util 0.80  ≈ 77 GB
Embedding (bge-large, fp16)                  util 0.05  ≈  5 GB
Reranker  (bge-reranker-v2-m3, fp16)         util 0.06  ≈  6 GB
                                              ─────────────────
                                              ≈ 88 GB / 96 GB
```

---

## Security note

nginx **injects** the API key, so anything that can reach ports 8000/8001/8002
gets free access. That's convenient behind a firewall/VPN but unsafe on a public
IP. To require client auth instead, edit `nginx/default.conf.template`: remove the
`proxy_set_header Authorization ...` lines so the client's own
`Authorization: Bearer` header is passed through to vLLM (which validates it).

---

## Files

```
├── docker-compose.yml              ← the whole stack (serving + monitoring)
├── .env.example                    ← config template
├── setup.sh                        ← host prep: Docker + NVIDIA toolkit
├── status.sh                       ← health / GPU / endpoint check
├── nginx/
│   └── default.conf.template       ← load-balancer + auth (envsubst'd at boot)
├── monitoring/
│   ├── prometheus/prometheus.yml   ← scrape DCGM + cAdvisor
│   ├── loki/loki-config.yml        ← log store
│   ├── promtail/promtail-config.yml← ship container logs → Loki
│   └── grafana/
│       ├── provisioning/           ← datasources + dashboard providers
│       └── dashboards/vllm-stack.json  ← GPU + logs dashboard
├── bench.py / bench_ccu.py / bench_stress.py   ← throughput benchmarks
└── models/                         ← HF model cache (persisted)
```
