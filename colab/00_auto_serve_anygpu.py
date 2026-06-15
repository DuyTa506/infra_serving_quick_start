# ============================================================================
#  AUTO vLLM serving — detect GPU → pick a feasible model → serve 3 services
#  (LLM + Embedding + Reranker) behind ONE cloudflare URL. Fully automatic.
#
#  SINGLE CELL: detect GPU (name / compute-cap / VRAM) → choose an LLM that
#  fits → install → download + launch all 3 (sequential, no-OOM) → open a
#  cloudflared tunnel → self-check → print the PUBLIC API URL + key.
#  If there is no usable GPU, it STOPS with a clear message.
# ============================================================================
import os, sys, subprocess, time, json, re, stat, urllib.request, threading

# ── 0. Knobs you may want to change ───────────────────────────────────────────
API_KEY  = "sk-secai2026"        # 🔑 CHANGE ME — backend sends this as Bearer
HF_TOKEN = ""                     # optional: faster / gated downloads
EMBED_MODEL  = "BAAI/bge-m3"      # fixed across all GPUs (multilingual, 1024-d)
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
LLM_PORT, EMBED_PORT, RERANK_PORT, GATEWAY_PORT = 8001, 8000, 8002, 8080
REQUIRE_KEY = True                # public URL → require backend's Bearer key
LOG_DIR = "/content/logs"; os.makedirs(LOG_DIR, exist_ok=True)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# ── 1. Detect the GPU ─────────────────────────────────────────────────────────
def _name_to_cc(name):
    n = name.upper()
    for key, cc in [("H200", 9.0), ("H100", 9.0), ("L40", 8.9), ("L4", 8.9), ("4090", 8.9),
                    ("ADA", 8.9), ("A100", 8.0), ("A10G", 8.6), ("A10", 8.6), ("3090", 8.6),
                    ("A6000", 8.6), ("A40", 8.6), ("T4", 7.5), ("2080", 7.5),
                    ("V100", 7.0), ("P100", 6.0)]:
        if key in n:
            return cc
    return 0.0

def detect_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    lines = (out.stdout or "").strip().splitlines()
    if not lines:
        return None
    parts = [x.strip() for x in lines[0].split(",")]
    name = parts[0]
    try:
        vram_gb = float(parts[1]) / 1024.0
    except (IndexError, ValueError):
        vram_gb = 0.0
    cc = 0.0
    if len(parts) > 2:
        try:
            cc = float(parts[2])
        except ValueError:
            cc = 0.0
    if cc == 0.0:
        cc = _name_to_cc(name)
    n_gpus = len(lines)
    return {"name": name, "vram_gb": vram_gb, "cc": cc, "n_gpus": n_gpus}


# ── 2. Pick a feasible model profile for this GPU ─────────────────────────────
def pick_profile(gpu):
    cc, vram = gpu["cc"], gpu["vram_gb"]
    fp8 = cc >= 8.9          # native FP8: Hopper (9.0) / Ada (8.9)
    marlin = cc >= 8.0       # AWQ marlin kernel needs Ampere+
    awq = "awq_marlin" if marlin else "awq"
    # Each tier: model that fits + quant + whether to use FP8 KV + LLM util + context.
    if vram >= 70 and fp8:                       # H100 / H200
        return dict(model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", quant=None, kv_fp8=True,
                    llm_util=0.70, max_len=40960, tier="H100-class (FP8 30B)")
    if vram >= 70:                               # A100 80GB / A6000-class
        return dict(model="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit", quant=awq, kv_fp8=False,
                    llm_util=0.60, max_len=32768, tier="80GB Ampere (int4 30B)")
    if vram >= 38:                               # A100 40GB / L40 48 / A6000 48
        return dict(model="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit", quant=awq, kv_fp8=False,
                    llm_util=0.80, max_len=16384, tier="40-48GB (int4 30B, tight)")
    if vram >= 20:                               # L4 / A10 / 3090 / 4090 (24GB)
        return dict(model="Qwen/Qwen2.5-14B-Instruct-AWQ", quant=awq, kv_fp8=False,
                    llm_util=0.62, max_len=16384, tier="24GB (int4 14B)")
    if vram >= 14:                               # T4 / V100-16 (16GB)
        return dict(model="Qwen/Qwen2.5-7B-Instruct-AWQ", quant=awq, kv_fp8=False,
                    llm_util=0.60, max_len=8192, tier="16GB (int4 7B)")
    return None


gpu = detect_gpu()
if gpu is None:
    raise SystemExit("❌ No NVIDIA GPU found. In Colab: Runtime → Change runtime type → GPU. "
                     "STOPPING.")
print(f"GPU: {gpu['name']}  |  {gpu['vram_gb']:.0f} GB  |  compute {gpu['cc']}  |  x{gpu['n_gpus']}")
if gpu["cc"] < 7.5:
    raise SystemExit(f"❌ {gpu['name']} (compute {gpu['cc']}) is too old for this vLLM stack "
                     f"(need Turing/7.5+). STOPPING.")
prof = pick_profile(gpu)
if prof is None:
    raise SystemExit(f"❌ {gpu['vram_gb']:.0f} GB is not enough to serve LLM + embedding + reranker "
                     f"(need ≥14 GB). STOPPING.")

LLM_MODEL = prof["model"]
# Size the two small models by absolute need (so util works on small cards too), give the
# LLM the rest of a 0.92 budget. On a 16 GB T4, 0.05 util would be < 1 GB and bge-m3 (~2.3 GB)
# would OOM — that's why these are computed from GB, not fixed fractions.
vram = gpu["vram_gb"]
EMBED_GPU_UTIL = round(max(0.04, 3.0 / vram), 3)     # bge-m3 needs ~2.5 GB
RERANK_GPU_UTIL = round(max(0.04, 2.5 / vram), 3)    # reranker needs ~2 GB
LLM_GPU_UTIL = min(prof["llm_util"], round(0.92 - EMBED_GPU_UTIL - RERANK_GPU_UTIL, 3))
LLM_MAX_MODEL_LEN = str(prof["max_len"])
LLM_QUANT = prof["quant"]
LLM_KV_FP8 = prof["kv_fp8"]

print("\n── Auto-selected profile ─────────────────────────────────────────")
print(f"  tier      : {prof['tier']}")
print(f"  LLM       : {LLM_MODEL}")
print(f"  quant     : {LLM_QUANT or 'auto (FP8)'}   kv_fp8={LLM_KV_FP8}   max_len={LLM_MAX_MODEL_LEN}")
print(f"  util      : llm={LLM_GPU_UTIL}  embed={EMBED_GPU_UTIL}  rerank={RERANK_GPU_UTIL}")
print(f"  embedding : {EMBED_MODEL}")
print(f"  reranker  : {RERANK_MODEL}")
print("──────────────────────────────────────────────────────────────────")
if LLM_GPU_UTIL < 0.45:
    print("⚠️  Little VRAM left for the LLM after the small models — it may be slow or OOM.")


# Runtime env passed to every vLLM subprocess.
IS_BLACKWELL = gpu["cc"] >= 10
SERVE_ENV = os.environ.copy()
# Tell FlashInfer's JIT the target arch on ANY card — without it, sampler/attention
# compilation can abort with a spurious "requires sm75" on images that don't export it.
SERVE_ENV.setdefault("TORCH_CUDA_ARCH_LIST", f"{int(gpu['cc'])}.0+PTX")
if IS_BLACKWELL:
    # Blackwell (sm100/sm120): the FlashInfer sampler JIT is unreliable → use the torch
    # sampler + SDPA attention (always available). Verified on RTX PRO 6000 Blackwell.
    SERVE_ENV["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    SERVE_ENV["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"


# ── 3. Install vLLM + gateway deps + cloudflared ──────────────────────────────
print("\n⏳ installing vllm + gateway deps (first run takes a few minutes)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm>=0.10.2",
                "huggingface_hub>=0.24", "fastapi", "uvicorn", "httpx"], check=True)

# vLLM's compiled extension must match the installed torch's CUDA. Recent vLLM is built for
# CUDA 13, but some Colab images ship torch cu128 → `import vllm._C` dies with
# "libcudart.so.13: cannot open shared object file". This is an ENVIRONMENT mismatch (not
# GPU-specific — it hits A100/H100/Blackwell alike), so detect it dynamically and only then
# realign torch to cu130, drop the cu128 torchvision (transformers imports it and the
# CUDA-major mismatch crashes), and match flashinfer to cu13. Versions track vLLM's current
# torch pin — bump if a future vLLM moves off torch 2.11 / CUDA 13.
def _vllm_c_error():
    r = subprocess.run([sys.executable, "-c", "import vllm._C"], capture_output=True, text=True)
    return "" if r.returncode == 0 else (r.stderr or "")

if "libcudart.so.13" in _vllm_c_error():
    print("CUDA mismatch (vLLM=cu13 vs torch=cu128) → realigning torch to cu130...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--index-url",
                    "https://download.pytorch.org/whl/cu130", "--force-reinstall",
                    "--no-cache-dir", "torch==2.11.0+cu130"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--extra-index-url",
                    "https://flashinfer.ai/whl/cu130/torch2.11/", "flashinfer-python"], check=False)
    leftover = _vllm_c_error()
    if leftover:
        print("⚠️  vllm._C still not importable after realign:\n", leftover[-600:])

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
         "--gpu-memory-utilization", str(util), "--trust-remote-code"]
    if API_KEY:
        c += ["--api-key", API_KEY]
    return c

def llm_cmd():
    c = base_cmd(LLM_MODEL, LLM_PORT, LLM_GPU_UTIL)
    if LLM_QUANT:
        c += ["--quantization", LLM_QUANT]
    if LLM_KV_FP8:
        c += ["--kv-cache-dtype", "fp8"]
    c += ["--max-model-len", LLM_MAX_MODEL_LEN, "--max-num-seqs", "64",
          "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    return c

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
    print(f"▶ {name}: starting (downloads model on first run)")
    return subprocess.Popen(cmd, stdout=open(f"{LOG_DIR}/{name}.log", "ab"),
                            stderr=subprocess.STDOUT, env=SERVE_ENV)

def wait_healthy(port, name, proc, timeout=2400):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:                      # process died → surface the tail
            tail = subprocess.run(["tail", "-n", "25", f"{LOG_DIR}/{name}.log"],
                                  capture_output=True, text=True).stdout
            raise SystemExit(f"❌ {name} crashed during startup. Last log lines:\n{tail}")
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"  ✓ {name} healthy on :{port}"); return
        except Exception:
            pass
        time.sleep(4)
    raise SystemExit(f"❌ {name} not healthy in {timeout}s — see {LOG_DIR}/{name}.log")

procs = {}
procs["llm"] = start("llm", llm_cmd());            wait_healthy(LLM_PORT, "llm", procs["llm"])
procs["embedding"] = start("embedding", embed_cmd()); wait_healthy(EMBED_PORT, "embedding", procs["embedding"])
procs["reranker"] = start("reranker", rerank_cmd());  wait_healthy(RERANK_PORT, "reranker", procs["reranker"])
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
_client = None

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
_server.install_signal_handlers = lambda: None   # required: uvicorn runs off the main thread
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
if not PUBLIC_URL:
    raise SystemExit("❌ tunnel URL not found — see /content/logs/cloudflared.log")


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
print("  LLM model id        →  " + LLM_MODEL)
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
