#!/usr/bin/env python3
"""
Sequential, health-gated launcher for the 3 vLLM servers on ONE GPU (Colab A100 80GB).

Why sequential: running 3 vLLM instances on a single card only stays stable if each one
reserves its slice of VRAM *before* the next starts. We start the LLM, wait until its
/health is 200, then the embedding, then the reranker. The three --gpu-memory-utilization
values sum to well under 1.0 (default 0.60 + 0.05 + 0.06 = 0.71 → ~57/80 GB), so there is
no contention and no OOM.

This is the Colab counterpart of the docker-compose `&llm-cmd / &embed-cmd / &rerank-cmd`
anchors. The embedding and reranker serve-args are copied verbatim from docker-compose.yml;
only the LLM args change for the A100 (NVFP4/modelopt/fp8-KV/MTP are Blackwell-only and are
removed here).

Usage:
    python launch_vllm.py            # start all 3, block until healthy, then keep running
    python launch_vllm.py --no-wait  # start them but don't block (use the cell's own poll)

Config comes from environment variables (see .env.colab.example). The notebook sets these
before calling this script.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

# ── Repo paths ───────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RERANK_TEMPLATE = os.path.join(REPO_ROOT, "templates", "qwen3_reranker.jinja")
LOG_DIR = os.environ.get("LOG_DIR", "/content/logs")

# ── Config (env-driven, same names as the Compose .env) ──────────────────────
API_KEY = os.environ.get("API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

LLM_MODEL = os.environ.get("LLM_MODEL", "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")

LLM_MAX_MODEL_LEN = os.environ.get("LLM_MAX_MODEL_LEN", "32768")
LLM_GPU_UTIL = os.environ.get("LLM_GPU_UTIL", "0.60")
EMBED_GPU_UTIL = os.environ.get("EMBED_GPU_UTIL", "0.05")
RERANK_GPU_UTIL = os.environ.get("RERANK_GPU_UTIL", "0.06")

# Quantization: empty = let vLLM auto-detect from the model's config. The default model
# (cpatonn/...-AWQ-4bit) is packed as `compressed-tensors`, NOT classic AWQ, so forcing
# "awq_marlin" raises a config-mismatch error (validated on vast.ai A100). Auto-detect
# handles both — on Ampere it still dispatches to Marlin int4 kernels. Override only if
# you KNOW the build's format (e.g. "awq_marlin" for a classic AWQ build, "gptq_marlin" for GPTQ).
LLM_QUANTIZATION = os.environ.get("LLM_QUANTIZATION", "")
# 2507-Instruct is non-thinking → no reasoning parser. Tool calls use the hermes parser.
LLM_TOOL_PARSER = os.environ.get("LLM_TOOL_PARSER", "hermes")

# Internal ports (mirror docker-compose). The gateway + tunnel sit in front of these.
LLM_PORT = int(os.environ.get("LLM_PORT", "8001"))
EMBED_PORT = int(os.environ.get("EMBED_PORT", "8000"))
RERANK_PORT = int(os.environ.get("RERANK_PORT", "8002"))

HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "900"))  # seconds per service


def vllm_base(model: str, port: int, gpu_util: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--gpu-memory-utilization", gpu_util,
        "--trust-remote-code",
    ]
    if API_KEY:
        cmd += ["--api-key", API_KEY]
    return cmd


def llm_cmd() -> list[str]:
    # A100-adapted LLM args. Compare with the &llm-cmd anchor in docker-compose.yml:
    #   REMOVED  --quantization modelopt   (NVFP4 / Blackwell only)
    #   REMOVED  --kv-cache-dtype fp8      (native FP8 is Hopper+; emulated on Ampere)
    #   REMOVED  --speculative-config mtp  (model-specific to the NVFP4 build)
    #   REMOVED  --reasoning-parser qwen3  (2507-Instruct is non-thinking)
    #   CHANGED  --quantization awq_marlin (Ampere AWQ fast path)
    cmd = vllm_base(LLM_MODEL, LLM_PORT, LLM_GPU_UTIL)
    if LLM_QUANTIZATION:
        cmd += ["--quantization", LLM_QUANTIZATION]
    cmd += [
        "--max-model-len", LLM_MAX_MODEL_LEN,
        "--max-num-seqs", "64",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser", LLM_TOOL_PARSER,
    ]
    return cmd


def embed_cmd() -> list[str]:
    # Identical to &embed-cmd in docker-compose.yml.
    cmd = vllm_base(EMBED_MODEL, EMBED_PORT, EMBED_GPU_UTIL)
    cmd += [
        "--runner", "pooling",
        "--dtype", "float16",
        "--max-model-len", "512",
    ]
    return cmd


def rerank_cmd() -> list[str]:
    # Identical to &rerank-cmd in docker-compose.yml. Qwen3-Reranker is a CAUSAL-LM reranker
    # (not a cross-encoder), so it needs --runner pooling + the hf-overrides routing it to
    # Qwen3ForSequenceClassification + the jinja chat template. Changing RERANK_MODEL alone
    # is NOT enough for this model family.
    hf_overrides = json.dumps({
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    })
    cmd = vllm_base(RERANK_MODEL, RERANK_PORT, RERANK_GPU_UTIL)
    cmd += [
        "--runner", "pooling",
        "--hf-overrides", hf_overrides,
        "--chat-template", RERANK_TEMPLATE,
        "--dtype", "float16",
        "--max-model-len", "4096",
    ]
    return cmd


def wait_healthy(port: int, name: str, timeout: int) -> None:
    """Poll http://localhost:<port>/health until 200 or raise on timeout."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"  ✓ {name} healthy on :{port}", flush=True)
                    return
        except Exception as e:  # noqa: BLE001 — server not up yet is expected
            last_err = str(e)
        time.sleep(3)
    raise TimeoutError(
        f"{name} (:{port}) not healthy after {timeout}s. Last error: {last_err}. "
        f"Check {LOG_DIR}/{name}.log"
    )


def start(name: str, cmd: list[str]) -> subprocess.Popen:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    env = os.environ.copy()
    if HF_TOKEN:
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        env["HF_TOKEN"] = HF_TOKEN
    print(f"▶ starting {name} → {log_path}", flush=True)
    print("    " + " ".join(cmd), flush=True)
    logf = open(log_path, "ab")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    wait = "--no-wait" not in sys.argv
    procs: dict[str, subprocess.Popen] = {}

    # 1) LLM first (largest, most sensitive to free VRAM) → wait healthy
    procs["llm"] = start("llm", llm_cmd())
    if wait:
        wait_healthy(LLM_PORT, "llm", HEALTH_TIMEOUT)

    # 2) Embedding → wait healthy
    procs["embedding"] = start("embedding", embed_cmd())
    if wait:
        wait_healthy(EMBED_PORT, "embedding", HEALTH_TIMEOUT)

    # 3) Reranker → wait healthy
    procs["reranker"] = start("reranker", rerank_cmd())
    if wait:
        wait_healthy(RERANK_PORT, "reranker", HEALTH_TIMEOUT)

    print("\n✅ all three vLLM servers are up:", flush=True)
    print(f"   llm        :{LLM_PORT}  ({LLM_MODEL})", flush=True)
    print(f"   embedding  :{EMBED_PORT}  ({EMBED_MODEL})", flush=True)
    print(f"   reranker   :{RERANK_PORT}  ({RERANK_MODEL})", flush=True)

    # Write the PIDs so the notebook/teardown cell can stop them.
    with open(os.path.join(LOG_DIR, "pids.json"), "w") as f:
        json.dump({k: p.pid for k, p in procs.items()}, f)

    if "--keep-foreground" in sys.argv:
        # Block, forwarding SIGTERM/SIGINT to children. Used if run as a standalone process.
        def _term(_signum, _frame):
            for p in procs.values():
                p.terminate()
            sys.exit(0)
        signal.signal(signal.SIGTERM, _term)
        signal.signal(signal.SIGINT, _term)
        for p in procs.values():
            p.wait()


if __name__ == "__main__":
    main()
