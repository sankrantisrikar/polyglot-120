"""
End-to-end latency and throughput benchmark against the running API.

Usage:
  python scripts/bench.py --url http://localhost:7860 --n 300 --concurrency 8
  python scripts/bench.py --url https://<your-api> --n 300 --concurrency 8

Reports, for successful requests:
  round trip   p50 / p95 / p99 as seen by the client (network + queueing + inference)
  server side  p50 / p95 / p99 of the API's own `latency_ms` (tokenize + forward + fallback)
plus requests per second, the share routed to the fallback path, and a
breakdown of any non-200 responses or transport errors.
"""
import argparse
import asyncio
import random
import statistics
import time
from collections import Counter

import httpx

SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "El rápido zorro marrón salta sobre el perro perezoso.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "快速的棕色狐狸跳过了懒狗。",
    "素早い茶色の狐が怠け者の犬を飛び越える。",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    "వేగవంతమైన గోధుమ నక్క సోమరి కుక్కపై దూకుతుంది.",
    "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "ok", "merci beaucoup", "lol idk", "Bonjour, how are you today?",
]


async def worker(client, url, n, out):
    for _ in range(n):
        text = random.choice(SAMPLES)
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/identify", json={"text": text, "top_k": 3}, timeout=60)
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                d = r.json()
                out.append(dict(ok=True, rt=dt, server=float(d.get("latency_ms", 0.0)),
                                routed=bool(d.get("routed_to_fallback", False)), text=text))
            else:
                out.append(dict(ok=False, rt=dt, err=f"HTTP {r.status_code}", text=text,
                                body=r.text[:120].replace("\n", " ")))
        except Exception as e:  # timeout, connection reset, ...
            out.append(dict(ok=False, rt=(time.perf_counter() - t0) * 1000,
                            err=type(e).__name__, text=text, body=str(e)[:120]))


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    url = args.url.rstrip("/")

    out = []
    per = max(1, args.n // args.concurrency)
    async with httpx.AsyncClient() as client:
        t_warm = time.perf_counter()
        w = await client.post(f"{url}/identify", json={"text": "warm up"}, timeout=180)
        print(f"warm-up: HTTP {w.status_code} in {(time.perf_counter() - t_warm) * 1000:.0f} ms "
              f"(includes cold start if the service was idle)")
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client, url, per, out) for _ in range(args.concurrency)])
        total = time.perf_counter() - t0

    oks = [o for o in out if o["ok"]]
    bad = [o for o in out if not o["ok"]]
    print(f"requests={len(out)} ok={len(oks)} failed={len(bad)} concurrency={args.concurrency}")
    if bad:
        print("failures:")
        for err, c in Counter(o["err"] for o in bad).most_common():
            ex = next(o for o in bad if o["err"] == err)
            print(f"  {err}: {c}   e.g. text={ex['text']!r} body={ex.get('body', '')!r}")
    if not oks:
        return
    rt = [o["rt"] for o in oks]
    sv = [o["server"] for o in oks]
    print(f"round trip  p50={pct(rt, .5):.1f}ms p95={pct(rt, .95):.1f}ms p99={pct(rt, .99):.1f}ms mean={statistics.mean(rt):.1f}ms")
    print(f"server side p50={pct(sv, .5):.1f}ms p95={pct(sv, .95):.1f}ms p99={pct(sv, .99):.1f}ms mean={statistics.mean(sv):.1f}ms")
    print(f"throughput={len(oks) / total:.1f} req/s")
    print(f"routed_to_fallback={sum(o['routed'] for o in oks) / len(oks):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
