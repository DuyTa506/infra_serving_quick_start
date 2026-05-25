# vLLM Serving Stack

3 services on 2x A100-SXM4 80GB (NVLink):

| Port | Service | Model |
|------|---------|-------|
| 8001 | LLM | Qwen/Qwen3.6-35B-A3B-FP8 (TP=2, MTP, 64K ctx) |
| 8000 | Embedding | BAAI/bge-large-en-v1.5 |
| 8002 | Reranker | BAAI/bge-reranker-v2-m3 |

**Performance:** ~5,300 tok/s peak, 128 CCU stable, 16K-context TTFB <10s at 32 CCU.

---

## API Usage

```bash
# Configure these
API_KEY="sk-..."                          # from .env
LLM_URL="https://<pod>-8001.proxy.runpod.net"
EMB_URL="https://<pod>-8000.proxy.runpod.net"
RERANK_URL="https://<pod>-8002.proxy.runpod.net"
```

**Auth:** All `/v1/*` endpoints require `Authorization: Bearer $API_KEY`.

Swagger docs:
- LLM: `$LLM_URL/docs`
- Embedding: `$EMB_URL/docs`
- Reranker: `$RERANK_URL/docs`

---

### LLM — Chat Completions

```
POST /v1/chat/completions
```

```bash
curl "$LLM_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url=f"{LLM_URL}/v1",
    api_key=API_KEY,
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
print(completion.choices[0].message.content)
```

**Streaming:**

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Python (requests):**

```python
import requests

r = requests.post(
    f"{LLM_URL}/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "messages": [
            {"role": "user", "content": "Hello!"},
        ],
    },
)
print(r.json()["choices"][0]["message"]["content"])
```

**JavaScript:**

```js
const r = await fetch(`${LLM_URL}/v1/chat/completions`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${API_KEY}`,
  },
  body: JSON.stringify({
    model: "Qwen/Qwen3.6-35B-A3B-FP8",
    messages: [{ role: "user", content: "Hello!" }],
  }),
});
const data = await r.json();
console.log(data.choices[0].message.content);
```

---

### Embedding

```
POST /v1/embeddings
```

```bash
curl "$EMB_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model": "BAAI/bge-large-en-v1.5", "input": "Hello world"}'

# batch
curl "$EMB_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model": "BAAI/bge-large-en-v1.5", "input": ["text 1", "text 2", "text 3"]}'
```

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url=f"{EMB_URL}/v1", api_key=API_KEY)

r = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input="Hello world",
)
print(f"Dim: {len(r.data[0].embedding)}")  # 1024
```

**Python (requests):**

```python
import requests

r = requests.post(
    f"{EMB_URL}/v1/embeddings",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": "BAAI/bge-large-en-v1.5", "input": "Hello world"},
)
embedding = r.json()["data"][0]["embedding"]
```

---

### Reranker

```
POST /v1/score       (pairwise)
POST /v1/rerank      (Cohere format — list rerank)
```

#### /v1/score — Pairwise

```bash
curl "$RERANK_URL/v1/score" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "text_1": "What is AI?",
    "text_2": "AI is artificial intelligence."
  }'
```

```python
import requests

r = requests.post(
    f"{RERANK_URL}/v1/score",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "BAAI/bge-reranker-v2-m3",
        "text_1": "What is AI?",
        "text_2": "AI is artificial intelligence.",
    },
)
print(r.json()["data"][0]["score"])  # e.g. 0.9989
```

#### /v1/rerank — List

```bash
curl "$RERANK_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "capital of France",
    "documents": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany.",
      "France is a country in Europe."
    ],
    "top_n": 2
  }'
```

```python
import requests

r = requests.post(
    f"{RERANK_URL}/v1/rerank",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "capital of France",
        "documents": [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "France is a country in Europe.",
        ],
        "top_n": 2,
    },
)
for item in r.json()["results"]:
    print(f"  [{item['index']}] {item['relevance_score']:.4f}  {item['document']['text']}")
```

---

## Server Operations

### First time setup

```bash
bash setup.sh
cp .env.example .env
# edit .env — set API_KEY and HF_TOKEN
```

### Start / Stop / Status

```bash
bash start_all.sh
bash status.sh
bash stop_all.sh
```

### Logs

```bash
tail -f logs/llm.log
tail -f logs/embedding.log
tail -f logs/reranker.log
```

---

## Benchmarking

Three benchmark scripts included:

```bash
# Basic throughput (C=1,2,4,8)
python3 bench.py

# CCU stress test (C=1..64)
python3 bench_ccu.py

# Full stress: 128 CCU + long prompts + 32k output
python3 bench_stress.py
```

All scripts hit `localhost:8001` with the API key from `.env`. Edit `API` and `KEY` at the top of each script to change target.

---

## GPU Layout

```
GPU 0  →  LLM shard 0  (94% ≈ 75 GB)  +  Reranker (15% ≈ 12 GB)
GPU 1  →  LLM shard 1  (94% ≈ 75 GB)  +  Embedding (15% ≈ 12 GB)
```

LLM: TP=2, 64K context, MTP speculative decoding, NVLink P2P, prefix caching.
Max batched tokens: 32,768. Max concurrent seqs: 128.

---

## Files

```
├── .env                ← API key + HF token
├── .env.example        ← config template
├── setup.sh            ← one-shot setup
├── start_all.sh        ← start all services
├── stop_all.sh         ← stop all services
├── status.sh           ← health + GPU check
├── start_llm.sh
├── start_embedding.sh
├── start_reranker.sh
├── docker-compose.yml
├── bench.py            ← basic throughput benchmark
├── bench_ccu.py        ← CCU stress test
├── bench_stress.py     ← full stress (128 CCU, 16K ctx, 32K output)
├── llm/Dockerfile
├── embedding/Dockerfile
├── reranker/Dockerfile
├── models/             ← HF model cache
└── logs/               ← log files + PIDs
```
