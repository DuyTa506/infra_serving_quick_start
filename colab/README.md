# Colab / cloud serving variant — 3 services behind one public URL

A notebook version of this repo's serving stack for a **single GPU** (Colab, RunPod, Lambda, …).
It runs all three services — **LLM + embedding + reranker** — on one card and exposes them as
**one public HTTPS URL** (`https://*.trycloudflare.com`) that a backend can call.

It is **additive**: the Docker stack in the repo root is unchanged. This folder only adds
notebooks + helper scripts, and **reuses** `../templates/qwen3_reranker.jinja`.

## 👉 Start here: `00_auto_serve_anygpu.ipynb` (recommended)

A **single, fully-automatic cell**. Turn on a GPU, set `API_KEY`, press Run. It:

1. **Detects the GPU** (name / compute-capability / VRAM).
2. **Picks an LLM that fits** (see table) — or **STOPS** with a clear reason if none does.
3. Installs deps and **auto-repairs CUDA mismatches** (e.g. vLLM cu13 vs Colab torch cu128).
4. Downloads + serves the 3 models (sequential, health-gated → no OOM).
5. Opens a **Cloudflare tunnel**, self-checks, and prints the **public URL + key**.

| Detected GPU | LLM picked | Quant / KV |
|---|---|---|
| H100 / H200 / Blackwell (≥70 GB, FP8) | `Qwen3-30B-A3B-Instruct-2507-FP8` | FP8 + FP8 KV |
| A100 80 GB (Ampere) | `Qwen3-30B-A3B-Instruct-2507-AWQ-4bit` | int4 (awq_marlin) |
| 40–48 GB (A100-40 / L40) | `Qwen3-30B-A3B-Instruct-2507-AWQ-4bit` (ctx 16k) | int4 |
| 24 GB (L4 / A10 / 3090 / 4090) | `Qwen2.5-14B-Instruct-AWQ` | int4 |
| 16 GB (T4 / "G4") | `Qwen2.5-7B-Instruct-AWQ` | int4 |
| no GPU / compute < 7.5 / < 14 GB | — | **STOP** with message |

Embedding/reranker `--gpu-memory-utilization` is computed from absolute VRAM need (not a fixed
fraction), so the small models still fit on a 16 GB T4, and the LLM takes the rest of a 0.92
budget. **Verified live on an RTX PRO 6000 Blackwell (96 GB):** all 3 services healthy and
correct (embed 1024-d, reranker Paris>Berlin, chat) through the Cloudflare URL, with auth
returning 401 when the key is missing.

> The auto-repair handles the case where recent vLLM ships for CUDA 13 while the Colab image has
> torch cu128 (`import vllm._C` → `libcudart.so.13` error). It realigns torch to cu130, drops the
> mismatched torchvision, and on Blackwell falls back to the SDPA attention + torch sampler
> (FlashInfer's JIT is unreliable on sm120). It only acts when it actually detects the mismatch.

## Why this isn't just the Docker stack with one GPU

| | Docker stack (repo root) | This notebook variant |
|---|---|---|
| GPU | 2× RTX 6000 Pro (Blackwell) | any single GPU (auto-selected) |
| LLM | `nvidia/Qwen3.6-35B-A3B-NVFP4` (NVFP4 + FP8 KV + MTP) | FP8 or int4 build that fits the card |
| Runtime | `docker compose up` | native vLLM processes in a notebook |
| Front door | nginx load balancer | FastAPI gateway (single port) |
| Exposure | host ports behind a firewall | cloudflared tunnel → public URL |
| Monitoring | Grafana/Prometheus/Loki | dropped (each vLLM keeps `/metrics`) |

NVFP4 + `--quantization modelopt` + MTP are Blackwell-native and don't load on Ampere, so the
auto picker swaps to a fitting FP8 (Hopper/Blackwell) or int4 AWQ (Ampere) build of the same
Qwen3-30B-A3B MoE family. Embedding/reranker serve-args match the Docker `&embed-cmd` /
`&rerank-cmd` anchors.

## Routes on the public URL

| Route | → service |
|---|---|
| `POST /v1/chat/completions`, `/v1/completions`, `GET /v1/models` | LLM (:8001) |
| `POST /v1/embeddings` | embedding (:8000) |
| `POST /v1/rerank`, `/v1/score` | reranker (:8002) |
| `GET /health` | aggregated (all three) |

## Backend usage

```python
from openai import OpenAI
client = OpenAI(base_url="https://xxxx.trycloudflare.com/v1", api_key="sk-secai2026")
client.chat.completions.create(model="<the LLM id printed by the notebook>",
                               messages=[{"role": "user", "content": "Hello!"}])
client.embeddings.create(model="BAAI/bge-m3", input="hello world")  # 1024-d
```

```bash
# rerank / score aren't in the OpenAI SDK — call the route directly
curl https://xxxx.trycloudflare.com/v1/rerank \
  -H "Authorization: Bearer sk-secai2026" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Reranker-0.6B","query":"capital of France",
       "documents":["Paris is the capital of France.","Berlin is the capital of Germany."]}'
```

## Auth (differs from Docker on purpose)

The Docker stack injects the key and leaves the firewalled ports open. The cloudflared URL is
**public**, so the gateway **requires** the backend to send `Authorization: Bearer $API_KEY`
(then forwards it to vLLM). Set `GATEWAY_REQUIRE_KEY=0` for the nginx-style inject-only behavior.

## Files

```
colab/
├── 00_auto_serve_anygpu.ipynb  ← 👉 recommended: ONE cell, auto-detect any GPU → serve
├── 00_auto_serve_anygpu.py     ← same content as a standalone script (python 00_auto_serve_anygpu.py)
├── serve_h100.ipynb            ← H100 (FP8) — ONE self-contained cell, fixed target
├── serve_h100.py               ← same content as a standalone script
├── serve_a100.ipynb            ← A100 (int4 AWQ) — multi-cell (clones repo, uses the helpers below)
├── launch_vllm.py              ← sequential, health-gated launcher (used by serve_a100)
├── gateway.py                  ← single-port FastAPI reverse proxy (replaces nginx) + auth
├── .env.colab.example          ← every config knob, with defaults
└── README.md                   ← this file
```

## Explicit-target variants

- **`serve_h100.ipynb`** — for an H100 80 GB (RunPod/Lambda; Colab rarely offers H100). One
  self-contained cell, fixed FP8 build (`Qwen3-30B-A3B-Instruct-2507-FP8` + FP8 KV).
- **`serve_a100.ipynb`** — A100 80 GB int4, multi-cell with checkpoints; clones the repo and
  uses `launch_vllm.py` + `gateway.py`. Good if you want to step through each phase.

Both are superseded by `00_auto_serve_anygpu.ipynb` for hands-off use.

## VRAM budget (example: A100 80 GB, int4 30B)

```
LLM  (Qwen3-30B-A3B AWQ int4, ~18 GB weights)  util 0.60  ≈ 48 GB (incl. KV)
Embedding (bge-m3, fp16)                                   ≈  4 GB
Reranker  (Qwen3-Reranker-0.6B, fp16)                      ≈  5 GB
                                               ─────────────────
                                               ≈ 57 GB / 80 GB  (~23 GB free)
```

Stability on a single card comes from **sequential, health-gated startup**: the LLM must report
`/health` 200 before the embedding starts, and so on — each instance reserves its VRAM slice
before the next, so they never contend or OOM.

## Gotchas

- **Ephemeral session.** When the VM stops, the tunnel URL dies and weights re-download.
- **URL changes** every run. For a stable address use a *named* cloudflared tunnel or ngrok.
- **Thinking mode.** `…-Instruct-2507` is non-thinking; the reasoning parser is dropped. For
  thinking, point the LLM id at an AWQ/FP8 build of the thinking `Qwen3-30B-A3B`.
- **40 GB cards.** The auto picker tightens context to fit; very long context still won't fit
  three models in 40 GB.
