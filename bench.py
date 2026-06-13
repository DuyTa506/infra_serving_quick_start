import asyncio, aiohttp, time, json, sys

API = "http://localhost:8001/v1/chat/completions"
KEY = "sk-secai2026"
MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"

PROMPTS = [
    "Explain the Transformer attention mechanism in one paragraph.",
    "Write a Python function to compute Fibonacci numbers recursively.",
    "What is the capital of France? Answer in one sentence.",
    "Summarize the key differences between REST and gRPC.",
    "Write a short poem about artificial intelligence.",
    "Explain what Docker is in 2-3 sentences.",
    "Convert this to a list comprehension: result = []; for x in range(10): if x % 2 == 0: result.append(x*2)",
    "What are the three laws of thermodynamics? List them briefly.",
]

async def single(session, prompt, max_tokens=256):
    t0 = time.time()
    async with session.post(API, json={
        "model": MODEL, "messages": [{"role":"user","content":prompt}],
        "max_tokens": max_tokens, "temperature": 0.0
    }, headers={"Authorization": f"Bearer {KEY}"}) as resp:
        data = await resp.json()
    ttfb = time.time() - t0
    usage = data.get("usage", {})
    tok = usage.get("completion_tokens", 0)
    return {"ttfb": ttfb, "tokens": tok, "tok_s": tok/ttfb if ttfb > 0 else 0, "status": resp.status}

async def bench(concurrency, n_requests):
    prompt = PROMPTS[0]
    async with aiohttp.ClientSession() as session:
        # warmup
        await single(session, "Hello", max_tokens=16)
        
        tasks = [single(session, prompt) for _ in range(n_requests)]
        t0 = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0
    
    total_tok = sum(r["tokens"] for r in results)
    ttfb_avg = sum(r["ttfb"] for r in results) / len(results)
    ttfb_p50 = sorted(r["ttfb"] for r in results)[len(results)//2]
    errors = sum(1 for r in results if r["status"] != 200)
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "elapsed_s": round(elapsed, 2),
        "req_s": round(n_requests/elapsed, 2),
        "tok_s": round(total_tok/elapsed, 2),
        "ttfb_avg": round(ttfb_avg, 2),
        "ttfb_p50": round(ttfb_p50, 2),
        "total_tok": total_tok,
        "errors": errors,
    }

async def main():
    print(f"Benchmarking {MODEL} | max_tokens=256\n")
    for cc in [1, 2, 4, 8]:
        result = await bench(cc, cc * 4)
        print(f"  C={result['concurrency']:2d} | {result['req_s']:6.1f} req/s | {result['tok_s']:7.1f} tok/s | avg_ttfb={result['ttfb_avg']:5.1f}s | p50_ttfb={result['ttfb_p50']:5.1f}s | err={result['errors']}")
    print()

asyncio.run(main())
