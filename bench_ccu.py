import asyncio, aiohttp, time, sys

API = "http://localhost:8001/v1/chat/completions"
KEY = "sk-secai2026"
MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"

PROMPT = "Explain the Transformer attention mechanism in detail, covering Q, K, V projections, scaled dot-product, multi-head attention, and how they interact."

async def single(session, max_tokens=256):
    t0 = time.time()
    try:
        async with session.post(API, json={
            "model": MODEL, "messages": [{"role":"user","content":PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.0
        }, headers={"Authorization": f"Bearer {KEY}"}, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
        ttfb = time.time() - t0
        usage = data.get("usage", {})
        tok = usage.get("completion_tokens", 0)
        return {"ok": True, "ttfb": ttfb, "tokens": tok, "status": resp.status}
    except Exception as e:
        return {"ok": False, "ttfb": time.time()-t0, "tokens": 0, "error": str(e)[:80]}

async def bench(concurrency, n_requests, label=""):
    async with aiohttp.ClientSession() as session:
        # warmup
        await single(session, max_tokens=16)
        await asyncio.sleep(0.5)

        tasks = [single(session) for _ in range(n_requests)]
        t0 = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    total_tok = sum(r["tokens"] for r in ok)
    ttfb_ok = sorted(r["ttfb"] for r in ok)
    p50 = ttfb_ok[len(ttfb_ok)//2] if ttfb_ok else 0
    p95 = ttfb_ok[int(len(ttfb_ok)*0.95)] if len(ttfb_ok) > 1 else (ttfb_ok[0] if ttfb_ok else 0)

    return {
        "label": label,
        "c": concurrency,
        "n": n_requests,
        "elapsed": round(elapsed, 1),
        "req_s": round(n_requests/elapsed, 1),
        "tok_s": round(total_tok/elapsed, 1),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "ok": len(ok),
        "fail": len(fail),
        "err_samples": [f["error"] for f in fail[:3]],
    }

async def main():
    print(f"CCU stress test — {MODEL}\n")
    print(f"{'CCU':>5s} | {'req/s':>7s} | {'tok/s':>8s} | {'p50':>6s} | {'p95':>6s} | {'ok':>4s} | {'fail':>4s} | note")
    print("-" * 85)

    for ccu, n in [(1,4), (2,8), (4,16), (8,32), (16,48), (32,64), (48,64), (64,64)]:
        r = await bench(ccu, n, str(ccu))
        note = ""
        if r["fail"] > 0:
            note = f"ERRORS: {r['err_samples'][:2]}"
        elif r["p95"] > 30:
            note = "p95 > 30s — queueing hard"
        elif r["p95"] > 10:
            note = "p95 > 10s — degrading"
        print(f"  {r['c']:2d} | {r['req_s']:6.1f} | {r['tok_s']:7.1f} | {r['p50']:5.1f}s | {r['p95']:5.1f}s | {r['ok']:3d} | {r['fail']:3d} | {note}")

    print()

asyncio.run(main())
