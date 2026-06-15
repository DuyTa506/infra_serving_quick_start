#!/usr/bin/env python3
"""
Single-port FastAPI gateway — the Colab replacement for nginx.

The 3 vLLM servers listen on separate localhost ports (8001/8000/8002). cloudflared's quick
tunnel exposes exactly ONE port, so this gateway fronts all three on :8080 and routes by URL
path, exactly like nginx/default.conf.template does for the Docker stack:

    /v1/chat/completions, /v1/completions, /v1/models   → llm        :8001
    /v1/embeddings                                       → embedding  :8000
    /v1/rerank, /v1/score                                → reranker   :8002
    /health                                              → all three (aggregated)

Auth model (differs from Docker on purpose):
  - Docker ports sit behind a firewall, so nginx just *injects* the key and clients send none.
  - The cloudflared URL is PUBLIC. So by default this gateway REQUIRES the backend to send
    `Authorization: Bearer $API_KEY`, then forwards that to vLLM. Set GATEWAY_REQUIRE_KEY=0
    to fall back to the nginx-style "inject, don't require" behavior.

Run:
    API_KEY=... uvicorn gateway:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

API_KEY = os.environ.get("API_KEY", "")
REQUIRE_KEY = os.environ.get("GATEWAY_REQUIRE_KEY", "1") == "1" and bool(API_KEY)

LLM_PORT = int(os.environ.get("LLM_PORT", "8001"))
EMBED_PORT = int(os.environ.get("EMBED_PORT", "8000"))
RERANK_PORT = int(os.environ.get("RERANK_PORT", "8002"))

LLM = f"http://localhost:{LLM_PORT}"
EMBED = f"http://localhost:{EMBED_PORT}"
RERANK = f"http://localhost:{RERANK_PORT}"

# Exact-path → upstream. Anything not listed falls through to the LLM (so /v1/models,
# /tokenize, etc. keep working against the chat server).
ROUTES = {
    "/v1/chat/completions": LLM,
    "/v1/completions": LLM,
    "/v1/models": LLM,
    "/v1/embeddings": EMBED,
    "/v1/rerank": RERANK,
    "/rerank": RERANK,
    "/v1/score": RERANK,
    "/score": RERANK,
}

# Hop-by-hop headers we must not forward verbatim.
_STRIP = {"host", "content-length", "connection", "authorization"}

app = FastAPI(title="Colab vLLM gateway")
# One shared client; generous read timeout for long generations, no write/pool limit issues.
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=600.0))


def _pick_upstream(path: str) -> str:
    return ROUTES.get(path, LLM)


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {"error": {"message": "Missing or invalid API key", "type": "invalid_request_error"}},
        status_code=401,
    )


@app.get("/health")
async def health() -> Response:
    """Aggregate the three backends. 200 only if all are healthy."""
    results = {}
    ok = True
    for name, base in (("llm", LLM), ("embedding", EMBED), ("reranker", RERANK)):
        try:
            r = await _client.get(f"{base}/health", timeout=5.0)
            results[name] = r.status_code
            ok = ok and r.status_code == 200
        except Exception as e:  # noqa: BLE001
            results[name] = f"down: {e}"
            ok = False
    return JSONResponse({"ok": ok, "services": results}, status_code=200 if ok else 503)


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(full_path: str, request: Request) -> Response:
    path = "/" + full_path

    # Public-URL auth: require the backend's Bearer token to match API_KEY.
    if REQUIRE_KEY:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {API_KEY}":
            return _unauthorized()

    upstream = _pick_upstream(path)
    url = f"{upstream}{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    # Rebuild headers: drop hop-by-hop + inbound auth, then inject our key toward vLLM.
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _STRIP}
    if API_KEY:
        headers["authorization"] = f"Bearer {API_KEY}"

    body = await request.body()

    # Detect a streaming chat/completions request → stream the response through (SSE).
    is_stream = False
    if path in ("/v1/chat/completions", "/v1/completions") and body:
        # cheap check — avoids JSON-parsing every request
        is_stream = b'"stream"' in body and b'"stream": false' not in body and b'"stream":false' not in body

    if is_stream:
        req = _client.build_request(request.method, url, headers=headers, content=body)
        upstream_resp = await _client.send(req, stream=True)

        async def _aiter():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    yield chunk
            finally:
                await upstream_resp.aclose()

        passthrough = {
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "connection")
        }
        return StreamingResponse(
            _aiter(),
            status_code=upstream_resp.status_code,
            headers=passthrough,
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
        )

    # Non-streaming: buffer and return.
    upstream_resp = await _client.request(request.method, url, headers=headers, content=body)
    passthrough = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in ("content-length", "transfer-encoding", "connection")
    }
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=passthrough,
        media_type=upstream_resp.headers.get("content-type"),
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    await _client.aclose()
