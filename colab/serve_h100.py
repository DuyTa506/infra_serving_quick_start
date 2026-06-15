# ============================================================================
#  vLLM serving on H100 80GB — LLM + Embedding + Reranker → 1 PUBLIC URL
#  SINGLE CELL: checks the GPU, installs deps, launches all 3 models
#  (sequential, no-OOM), opens a cloudflared tunnel, self-checks, and prints
#  the public API URL + key. Run this ONE cell, wait, copy the URL.
#
#  H100 has native FP8, so the LLM uses the FP8 build + FP8 KV cache (better
#  quality/throughput than int4). For A100 use the int4 notebook instead.
# ============================================================================
import os, sys, subprocess, time, json, re, stat, urllib.request, threading

# ── 1. Config ────────────────────────────────────────────────────────────────
API_KEY  = "sk-secai2026"        # 🔑 CHANGE ME — backend sends this as Bearer
HF_TOKEN = ""                     # optional: faster / gated downloads

LLM_MODEL    = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"   # FP8 native on H100
EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
LLM_MAX_MODEL_LEN = "40960"
LLM_GPU_UTIL, EMBED_GPU_UTIL, RERANK_GPU_UTIL = "0.70", "0.05", "0.06"
LLM_PORT, EMBED_PORT, RERANK_PORT, GATEWAY_PORT = 8001, 8000, 8002, 8080
REQUIRE_KEY = True                # public URL → require backend's Bearer key
LOG_DIR = "/content/logs"; os.makedirs(LOG_DIR, exist_ok=True)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# ── 2. GPU check ──────────────────────────────────────────────────────────────
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "nvidia-smi not found")
if "H100" not in gpu:
    print("⚠️  Not an H100. This notebook is tuned for H100 80GB (FP8 + FP8 KV).")
    print("    On an A100 use the int4 notebook (serve_a100.ipynb) — A100 has no native FP8.")

# ── 3. Install vLLM + gateway deps + cloudflared ──────────────────────────────
print("\n⏳ installing vllm + gateway deps (a few minutes on first run)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm>=0.10.2",
                "huggingface_hub>=0.24", "fastapi", "uvicorn", "httpx"], check=True)
CF = "/usr/local/bin/cloudflared"
if not os.path.exists(CF):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", CF)
    os.chmod(CF, os.stat(CF).st_mode | stat.S_IEXEC)
print("cloudflared:", subprocess.run([CF, "--version"], capture_output=True, text=True).stdout.strip())

# ── 4. Reranker chat template (Qwen3-Reranker is a causal-LM reranker) ─────────
TEMPLATE_PATH = "/content/qwen3_reranker.jinja"
with open(TEMPLATE_PATH, "w") as _f:
    _f.write(
'''<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ instruction | default(instruct | default(messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query", true), true), true) }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>
''')

# ── 5. Launch the 3 vLLM servers (sequential + health-gated → no OOM) ──────────
def base_cmd(model, port, util):
    c = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
         "--host", "0.0.0.0", "--port", str(port),
         "--gpu-memory-utilization", util, "--trust-remote-code"]
    if API_KEY:
        c += ["--api-key", API_KEY]
    return c

def llm_cmd():
    # FP8 weights auto-detected from the model config; FP8 KV cache is native on H100.
    return base_cmd(LLM_MODEL, LLM_PORT, LLM_GPU_UTIL) + [
        "--kv-cache-dtype", "fp8",
        "--max-model-len", LLM_MAX_MODEL_LEN, "--max-num-seqs", "128",
        "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]

def embed_cmd():
    return base_cmd(EMBED_MODEL, EMBED_PORT, EMBED_GPU_UTIL) + [
        "--runner", "pooling", "--dtype", "float16", "--max-model-len", "512"]

def rerank_cmd():
    hf = json.dumps({"architectures": ["Qwen3ForSequenceClassification"],
                     "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": True})
    return base_cmd(RERANK_MODEL, RERANK_PORT, RERANK_GPU_UTIL) + [
        "--runner", "pooling", "--hf-overrides", hf, "--chat-template", TEMPLATE_PATH,
        "--dtype", "float16", "--max-model-len", "4096"]

def start(name, cmd):
    print(f"▶ {name}: starting")
    return subprocess.Popen(cmd, stdout=open(f"{LOG_DIR}/{name}.log", "ab"), stderr=subprocess.STDOUT)

def wait_healthy(port, name, timeout=1200):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"  ✓ {name} healthy on :{port}"); return
        except Exception:
            pass
        time.sleep(4)
    raise TimeoutError(f"{name} not healthy in {timeout}s — see {LOG_DIR}/{name}.log")

procs = {}
procs["llm"] = start("llm", llm_cmd());            wait_healthy(LLM_PORT, "llm")
procs["embedding"] = start("embedding", embed_cmd()); wait_healthy(EMBED_PORT, "embedding")
procs["reranker"] = start("reranker", rerank_cmd());  wait_healthy(RERANK_PORT, "reranker")
json.dump({k: p.pid for k, p in procs.items()}, open(f"{LOG_DIR}/pids.json", "w"))
print("✅ all 3 vLLM servers up")

# ── 6. Single-port gateway (path routing + auth + SSE) — runs in a thread ──────
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

LLM = f"http://localhost:{LLM_PORT}"; EMB = f"http://localhost:{EMBED_PORT}"; RR = f"http://localhost:{RERANK_PORT}"
ROUTES = {"/v1/chat/completions": LLM, "/v1/completions": LLM, "/v1/models": LLM,
          "/v1/embeddings": EMB, "/v1/rerank": RR, "/rerank": RR, "/v1/score": RR, "/score": RR}
_STRIP = {"host", "content-length", "connection", "authorization"}
app = FastAPI()
_client = None  # created in the gateway's own event loop (startup) to avoid loop-mismatch

@app.on_event("startup")
async def _startup():
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=600.0))

@app.get("/health")
async def _health():
    res = {}; ok = True
    for n, b in (("llm", LLM), ("embedding", EMB), ("reranker", RR)):
        try:
            r = await _client.get(f"{b}/health", timeout=5.0); res[n] = r.status_code; ok = ok and r.status_code == 200
        except Exception as e:
            res[n] = f"down: {e}"; ok = False
    return JSONResponse({"ok": ok, "services": res}, status_code=200 if ok else 503)

@app.api_route("/{p:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def _proxy(p: str, request: Request):
    path = "/" + p
    if REQUIRE_KEY and API_KEY and request.headers.get("authorization", "") != f"Bearer {API_KEY}":
        return JSONResponse({"error": {"message": "Missing or invalid API key"}}, status_code=401)
    up = ROUTES.get(path, LLM)
    url = up + path + (f"?{request.url.query}" if request.url.query else "")
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _STRIP}
    if API_KEY:
        headers["authorization"] = f"Bearer {API_KEY}"
    body = await request.body()
    stream = (path in ("/v1/chat/completions", "/v1/completions") and b'"stream"' in body
              and b'"stream": false' not in body and b'"stream":false' not in body)
    if stream:
        req = _client.build_request(request.method, url, headers=headers, content=body)
        ur = await _client.send(req, stream=True)
        async def _it():
            try:
                async for c in ur.aiter_raw():
                    yield c
            finally:
                await ur.aclose()
        h = {k: v for k, v in ur.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "connection")}
        return StreamingResponse(_it(), status_code=ur.status_code, headers=h,
                                 media_type=ur.headers.get("content-type", "text/event-stream"))
    ur = await _client.request(request.method, url, headers=headers, content=body)
    h = {k: v for k, v in ur.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "connection")}
    return Response(content=ur.content, status_code=ur.status_code, headers=h,
                    media_type=ur.headers.get("content-type"))

_cfg = uvicorn.Config(app, host="0.0.0.0", port=GATEWAY_PORT, log_level="warning")
_server = uvicorn.Server(_cfg)
_server.install_signal_handlers = lambda: None   # required: we run uvicorn off the main thread
threading.Thread(target=_server.run, daemon=True).start()
time.sleep(4)
print(f"gateway up on :{GATEWAY_PORT}")

# ── 7. cloudflared tunnel → public URL ────────────────────────────────────────
cf = subprocess.Popen([CF, "tunnel", "--url", f"http://localhost:{GATEWAY_PORT}", "--no-autoupdate"],
                      stdout=open(f"{LOG_DIR}/cloudflared.log", "ab"), stderr=subprocess.STDOUT)
PUBLIC_URL = None
for _ in range(45):
    time.sleep(2)
    try:
        log = open(f"{LOG_DIR}/cloudflared.log").read()
    except FileNotFoundError:
        continue
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
    if m:
        PUBLIC_URL = m.group(0); break
assert PUBLIC_URL, "tunnel URL not found — see /content/logs/cloudflared.log"

# ── 8. Self-check through the public URL (retries while the tunnel warms up) ───
def _call(path, payload, retries=6):
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(PUBLIC_URL + path, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:
            last = e; time.sleep(5)
    raise last

chat = _call("/v1/chat/completions", {"model": LLM_MODEL,
    "messages": [{"role": "user", "content": "Say hi in one short sentence."}], "max_tokens": 64})
emb = _call("/v1/embeddings", {"model": EMBED_MODEL, "input": "hello world"})
rr = _call("/v1/rerank", {"model": RERANK_MODEL, "query": "capital of France",
    "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."], "top_n": 2})
print("\n  CHAT  :", chat["choices"][0]["message"]["content"].strip())
print("  EMBED :", len(emb["data"][0]["embedding"]), "dims")
print("  RERANK:", [(d["index"], round(d["relevance_score"], 3)) for d in rr["results"]])

print("\n" + "=" * 72)
print("  ✅ PUBLIC API URL  →  " + PUBLIC_URL)
print("  🔑 API key (Bearer) →  " + API_KEY)
print("  Routes: /v1/chat/completions   /v1/embeddings   /v1/rerank   /v1/score")
print("=" * 72)

# Keep alive when run as a plain script; in a notebook the kernel stays alive.
try:
    get_ipython()  # noqa: F821 — defined only inside IPython/Colab
    _IN_NB = True
except NameError:
    _IN_NB = False
if not _IN_NB:
    print("\n(serving... press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        for _p in list(procs.values()) + [cf]:
            try: _p.terminate()
            except Exception: pass
