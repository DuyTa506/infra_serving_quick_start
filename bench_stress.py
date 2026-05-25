import asyncio, aiohttp, time

API = "http://localhost:8001/v1/chat/completions"
KEY = "sk-secai2026"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

LONG_PROMPT = """You are a technical expert. Please provide a comprehensive analysis of distributed systems, covering:
1. Consistency models (strong, eventual, causal) and their tradeoffs
2. Consensus algorithms (Paxos, Raft) and when to use each
3. Partitioning strategies (range, hash, directory-based)
4. Replication topologies and their failure modes
5. Real-world case studies of systems that got these wrong

Be thorough and detailed in your response. Include specific examples."""

SHORT_PROMPT = "Explain the Transformer attention mechanism in one paragraph."

async def single(session, prompt, max_tokens=256):
    t0 = time.time()
    try:
        async with session.post(API, json={
            "model": MODEL, "messages": [{"role":"user","content":prompt}],
            "max_tokens": max_tokens, "temperature": 0.0
        }, headers={"Authorization": f"Bearer {KEY}"}, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            data = await resp.json()
        ttfb = time.time() - t0
        usage = data.get("usage", {})
        tok = usage.get("completion_tokens", 0)
        prompt_tok = usage.get("prompt_tokens", 0)
        return {"ok": True, "ttfb": ttfb, "tokens": tok, "prompt_tok": prompt_tok, "status": resp.status}
    except Exception as e:
        return {"ok": False, "ttfb": time.time()-t0, "tokens": 0, "prompt_tok": 0, "error": str(e)[:100]}

async def bench(ccu, n, prompt, max_tok, label):
    async with aiohttp.ClientSession() as session:
        await single(session, "Hello", max_tokens=8)
        await asyncio.sleep(0.3)

        tasks = [single(session, prompt, max_tok) for _ in range(n)]
        t0 = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    total_tok = sum(r["tokens"] for r in ok)
    ttfb_ok = sorted(r["ttfb"] for r in ok)
    p50 = ttfb_ok[len(ttfb_ok)//2] if ttfb_ok else 0
    p95_idx = min(int(len(ttfb_ok)*0.95), len(ttfb_ok)-1) if len(ttfb_ok) > 1 else 0
    p95 = ttfb_ok[p95_idx] if ttfb_ok else 0
    avg_prompt_tok = sum(r["prompt_tok"] for r in ok) // len(ok) if ok else 0

    return {
        "label": label, "ccu": ccu, "req_s": round(n/elapsed, 1), "tok_s": round(total_tok/elapsed, 1),
        "p50": round(p50, 2), "p95": round(p95, 2),
        "ok": len(ok), "fail": len(fail), "elapsed": round(elapsed, 1),
        "prompt_tok": avg_prompt_tok,
    }

async def main():
    print("=" * 95)
    print("  STRESS TEST — CCU up to 128, long context prompts")
    print("=" * 95)

    # Phase 1: High CCU, short prompt, 256 output
    print("\n--- Phase 1: Short prompt, max_tokens=256, high CCU ---")
    print(f"{'CCU':>5s} | {'req/s':>7s} | {'tok/s':>8s} | {'p50':>6s} | {'p95':>6s} | {'ok':>4s} | {'fail':>4s} | {'elapsed':>7s}")
    print("-" * 85)
    for ccu, n in [(64,128), (96,128), (128,128)]:
        r = await bench(ccu, n, SHORT_PROMPT, 256, str(ccu))
        note = "!! ERRORS" if r["fail"] > 0 else ""
        print(f"  {r['ccu']:3d} | {r['req_s']:6.1f} | {r['tok_s']:7.1f} | {r['p50']:5.1f}s | {r['p95']:5.1f}s | {r['ok']:3d} | {r['fail']:3d} | {r['elapsed']:6.1f}s {note}")

    # Phase 2: Long prompt, moderate CCU, longer output
    print("\n--- Phase 2: Long prompt (~500 tok), max_tokens=512, varying CCU ---")
    print(f"{'CCU':>5s} | {'req/s':>7s} | {'tok/s':>8s} | {'p50':>6s} | {'p95':>6s} | {'ok':>4s} | {'fail':>4s} | {'elapsed':>7s}")
    print("-" * 85)
    for ccu, n in [(2,4), (4,8), (8,16), (16,32), (32,32)]:
        r = await bench(ccu, n, LONG_PROMPT, 512, str(ccu))
        note = "!! ERRORS" if r["fail"] > 0 else ""
        print(f"  {r['ccu']:3d} | {r['req_s']:6.1f} | {r['tok_s']:7.1f} | {r['p50']:5.1f}s | {r['p95']:5.1f}s | {r['ok']:3d} | {r['fail']:3d} | {r['elapsed']:6.1f}s {note}")

    # Phase 3: 32k output stress (single request first, then a few concurrent)
    print("\n--- Phase 3: max_tokens=32768 (32k output) ---")
    print(f"{'CCU':>5s} | {'req/s':>7s} | {'tok/s':>8s} | {'p50':>6s} | {'p95':>6s} | {'ok':>4s} | {'fail':>4s} | {'elapsed':>7s}")
    print("-" * 85)
    for ccu, n in [(1,1), (2,2)]:
        r = await bench(ccu, n, SHORT_PROMPT, 32768, str(ccu))
        note = "!! ERRORS" if r["fail"] > 0 else ""
        print(f"  {r['ccu']:3d} | {r['req_s']:6.1f} | {r['tok_s']:7.1f} | {r['p50']:5.1f}s | {r['p95']:5.1f}s | {r['ok']:3d} | {r['fail']:3d} | {r['elapsed']:6.1f}s {note}")

    print()

asyncio.run(main())
