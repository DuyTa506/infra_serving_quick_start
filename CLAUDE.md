# CLAUDE.md

Guidance for working in this repo. Read this before changing the stack.

## What this is

A fully Dockerized, OpenAI-compatible serving stack for **2× RTX 6000 Pro
(Blackwell, 96 GB, PCIe — NO NVLink P2P)**. Three model services (LLM, embedding,
reranker) + nginx load balancer + a centralized GPU/log monitoring stack.
Everything starts with a single `docker compose up -d`.

## Core design constraint (don't break this)

The two GPUs have **no NVLink / P2P** — they talk over PCIe. So:

- **Never reintroduce tensor-parallel across the cards.** TP over PCIe bottlenecks.
- Each GPU runs a **full independent replica** (`llm-N`, `embedding-N`, `reranker-N`),
  all with `--tensor-parallel-size 1`. There is zero cross-GPU NCCL traffic by design.
- **nginx load-balances** the two replicas of each service. This gives ~2× LLM
  throughput and HA (one card down → traffic routes to the other).

```
GPU 0 ── llm-0 ── embedding-0 ── reranker-0  ┐
                                             ├─► nginx LB ─► clients
GPU 1 ── llm-1 ── embedding-1 ── reranker-1  ┘
```

## Models

- **LLM:** `nvidia/Qwen3.6-35B-A3B-NVFP4` — NVFP4 is the native fast path on
  Blackwell (SM120). Served with `--quantization modelopt` + `--kv-cache-dtype fp8`.
  NVFP4 weights ~20 GB; FP8 KV halves KV memory.
- **Embedding:** `BAAI/bge-m3` (MIT, multilingual, fp16, dim 1024) — served as a
  pooling model (`--runner pooling`); default dense pooling → 1024-d vectors.
- **Reranker:** `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0) — `/v1/rerank` + `/v1/score`.
  This is a **causal-LM reranker, not a cross-encoder**, so the `&rerank-cmd` block
  must serve it with `--runner pooling` + `--hf-overrides`
  (`architectures:[Qwen3ForSequenceClassification]`, `classifier_from_token:["no","yes"]`,
  `is_original_qwen3_reranker:true`) + `--chat-template /templates/qwen3_reranker.jinja`.
  **Changing `RERANK_MODEL` alone is NOT enough** for a Qwen3-Reranker — a plain
  cross-encoder id (e.g. bge-reranker) needs no extra flags, but without these the
  model loads as a generative LM and `/score` won't work. The jinja template is
  bind-mounted via `./templates:/templates:ro` (added to the `*vllm-common` volumes).

Model ids and key knobs are env vars in `.env` (`LLM_MODEL`, `EMBED_MODEL`,
`RERANK_MODEL`, `LLM_MAX_MODEL_LEN`, `VLLM_IMAGE`). The reranker serve-args above
are NOT env-driven — they live in the `&rerank-cmd` anchor in `docker-compose.yml`.

## Ports

| Published port | What |
|------|------|
| 8001 | LLM (LB) `POST /v1/chat/completions` |
| 8000 | Embedding (LB) `POST /v1/embeddings` |
| 8002 | Reranker (LB) `/v1/rerank` `/v1/score` |
| 3000 | **Grafana — the ONLY monitoring UI** (GPU dashboards + logs) |

**Monitoring is unified to one UI.** Prometheus, dcgm-exporter, cAdvisor, and
Loki are internal-only (`expose:`, not `ports:`) — Grafana queries them by name
over the `serving` network. Don't re-add their `ports:` mappings unless someone
explicitly wants the raw Prometheus/DCGM UI for debugging.

Internal container ports: llm `8001`, embedding `8000`, reranker `8002` (each
container has its own netns on the `serving` bridge, so ports can repeat).

## Layout

```
docker-compose.yml   ← the whole stack. Uses YAML anchors:
                         *vllm-common (image/env/volume), *gpu0 / *gpu1 (device pin),
                         &llm-cmd / &embed-cmd / &rerank-cmd (shared between replicas)
nginx/default.conf.template  ← upstreams + LB + auth. Rendered by the official
                               nginx image's envsubst at boot (NGINX_ENVSUBST_FILTER=API_KEY).
monitoring/
  prometheus/prometheus.yml        ← scrapes dcgm-exporter + cadvisor
  loki/loki-config.yml             ← single-binary log store (filesystem)
  promtail/promtail-config.yml     ← docker_sd → ships ALL container logs to Loki
  grafana/provisioning/            ← datasources (uid: prometheus, loki) + dashboard provider
  grafana/dashboards/vllm-stack.json  ← GPU panels (DCGM) + Loki logs panel
setup.sh    ← host prep: Docker + NVIDIA Container Toolkit (standard host, not RunPod)
status.sh   ← compose ps + HTTP health + GPU snapshot
bench*.py   ← throughput benchmarks, hit localhost:8001
models/     ← HF cache (gitignored, persisted)
```

## Conventions / gotchas

- **Editing the LLM serve args:** change the `&llm-cmd` anchor block in
  `docker-compose.yml` once — `llm-1` references it via `*llm-cmd`, so both
  replicas stay in sync. Same for `&embed-cmd` / `&rerank-cmd`.
- **GPU pinning:** services merge `*gpu0` or `*gpu1` (sets `device_ids`). Replica
  `-0` → GPU 0, replica `-1` → GPU 1. `dcgm-exporter` uses `count: all`.
- **VRAM budget per card (~88/96 GB):** `--gpu-memory-utilization` is 0.80 (LLM) /
  0.05 (embed) / 0.06 (rerank). Bumping LLM context or util can OOM the small models.
- **Auth:** nginx **injects** `Authorization: Bearer ${API_KEY}`, so clients hit
  the public ports without a key. This means the ports are open to anyone who can
  reach them — fine behind a firewall, not on a public IP. To require client auth,
  remove the `proxy_set_header Authorization` lines in `nginx/default.conf.template`.
- **No build step:** services use `vllm/vllm-openai` directly with `command:`
  overrides — there are no per-service Dockerfiles anymore.
- **Promtail** reads the Docker socket + `/var/lib/docker/containers`; it labels
  streams by `container` and compose `service`. Loki query: `{job="docker"}`.
- This targets a **standard Docker host with bridge networking** (not RunPod host
  networking). nginx runs as a container and reaches services by name.

## Common commands

```bash
docker compose up -d              # start everything
docker compose logs -f llm-0      # follow one replica
docker compose restart llm-1      # restart a single replica
docker compose config --quiet     # validate compose + anchor merges
bash status.sh                    # health + GPU + endpoints
```

## Verifying changes

- `docker compose config --quiet` after any compose edit (catches anchor/YAML errors).
- `python3 -m json.tool monitoring/grafana/dashboards/vllm-stack.json` after dashboard edits.
- There are no unit tests; correctness is validated by bringing the stack up on the
  GPU host and checking `bash status.sh` + Grafana at :3000.
