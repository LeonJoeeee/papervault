"""Search-relevance baseline (round 2, issue #4): precision@served for ``search_papers``.

The live return gate (LLM3 ``judge_return``) already thresholds what gets served, so
the served set is by-construction "relevant according to the gate". This script is the
EXTERNAL check: it re-judges the SERVED set against the caller's intent with the strong
slot at a measurement bar (score >= 0.7 counts as relevant) and reports precision@served.
It measures, it never gates — a low number is a finding, not a failure of this script.

Usage (live service up; sequential on purpose — one search at a time, real caller shape):

    .venv/bin/python scripts/search_relevance_baseline.py \
        --intents src/papervault/eval/intents_v1.jsonl --n 12 --tag srb_$(date +%m%d)

Writes ``src/papervault/eval/results/<tag>.jsonl`` (one line per intent: served papers,
judge scores, latency) and prints the summary card. Latency is also independently
visible in the MCPCALL access log; judge-side token spend is OUTSIDE the LLMTOK book
(offline-tool scope, see papervault/llm_usage.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from papervault import config  # noqa: E402

JUDGE_MODEL = "mimo-v2.5-pro"
RELEVANT_BAR = 0.7
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)

_JUDGE_SYSTEM = """You are a strict literature-relevance judge. A researcher states what \
they are working on and what they want to find; you score how relevant each returned paper \
is to THAT stated need, from its title and abstract alone.

Scoring bar (absolute, not relative to the other papers in the list):
1.0 = directly on the stated need (the researcher would certainly open it)
0.7 = clearly useful for the need, even if partial or adjacent in method/system
0.4 = same broad field but does not serve the stated need
0.0 = off-topic for the need

Return ONLY a JSON object: {"items": [{"idx": <int>, "score": <float 0-1>}]}, one item
per paper, no prose."""


def _judge_prompt(intent: str, papers: list[dict]) -> str:
    lines = [f"RESEARCHER'S NEED:\n{intent}\n\nRETURNED PAPERS:"]
    for i, p in enumerate(papers):
        ab = (p.get("abstract") or "(no abstract)")[:1200]
        lines.append(f"[{i}] {p.get('title')} ({p.get('year')})\n{ab}")
    return "\n\n".join(lines)


async def _judge(client: AsyncOpenAI, intent: str, papers: list[dict]) -> list[float]:
    for attempt in (1, 2):
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": _judge_prompt(intent, papers)},
                ],
            )
            raw = _FENCE_RE.sub("", resp.choices[0].message.content or "")
            items = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["items"]
            scores = [0.0] * len(papers)
            for it in items:
                idx = int(it["idx"])
                if 0 <= idx < len(papers):
                    scores[idx] = max(0.0, min(1.0, float(it["score"])))
            return scores
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"    judge retry after: {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(5)
    raise AssertionError("unreachable")


async def _search(intent: str) -> tuple[list[dict], float]:
    t0 = time.monotonic()
    async with streamablehttp_client(
        "http://127.0.0.1:8080/mcp", sse_read_timeout=timedelta(seconds=120)
    ) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(
                "search_papers", {"query": intent},
                read_timeout_seconds=timedelta(seconds=600),
            )
            data = json.loads(res.content[0].text)
    return data.get("results") or [], time.monotonic() - t0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intents", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    intents = [json.loads(ln) for ln in Path(args.intents).read_text().splitlines() if ln.strip()]
    intents = intents[: args.n]
    out_path = Path(__file__).resolve().parent.parent / "src/papervault/eval/results" / f"{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judge_client = AsyncOpenAI(base_url=config.GATEWAY_URL, api_key=config.GATEWAY_KEY or "sk-")
    cards = []
    with out_path.open("w") as fh:
        for row in intents:
            qid, intent = row["qid"], row["intent"]
            print(f"== {qid}", flush=True)
            try:
                papers, dt = await _search(intent)
            except Exception as e:  # noqa: BLE001
                print(f"    SEARCH FAILED: {type(e).__name__}: {e}", flush=True)
                fh.write(json.dumps({"qid": qid, "error": f"search:{type(e).__name__}"}) + "\n")
                fh.flush()
                continue
            if not papers:
                cards.append({"qid": qid, "served": 0, "precision": None, "latency_s": dt})
                fh.write(json.dumps({"qid": qid, "served": 0, "latency_s": dt, "papers": []}) + "\n")
                fh.flush()
                print(f"    served=0 latency={dt:.0f}s", flush=True)
                continue
            scores = await _judge(judge_client, intent, papers)
            rel = sum(1 for s in scores if s >= RELEVANT_BAR)
            prec = rel / len(papers)
            cards.append({"qid": qid, "served": len(papers), "precision": prec, "latency_s": dt})
            fh.write(json.dumps({
                "qid": qid, "served": len(papers), "latency_s": round(dt, 1),
                "precision": round(prec, 4),
                "papers": [
                    {"title": p.get("title"), "year": p.get("year"), "doi": p.get("doi"),
                     "score": s}
                    for p, s in zip(papers, scores)
                ],
            }) + "\n")
            fh.flush()
            print(f"    served={len(papers)} P@served={prec:.2f} latency={dt:.0f}s", flush=True)

    scored = [c for c in cards if c["precision"] is not None]
    if scored:
        precs = sorted(c["precision"] for c in scored)
        lats = sorted(c["latency_s"] for c in scored)
        print("\n=== SEARCH RELEVANCE BASELINE ===")
        print(f"intents scored: {len(scored)}/{len(cards)}")
        print(f"P@served mean={sum(precs)/len(precs):.4f} min={precs[0]:.4f} max={precs[-1]:.4f}")
        print(f"served mean={sum(c['served'] for c in scored)/len(scored):.1f}")
        import math
        p95 = lats[min(len(lats) - 1, math.ceil(0.95 * len(lats)) - 1)]  # nearest-rank
        print(f"latency mean={sum(lats)/len(lats):.0f}s p95={p95:.0f}s max={lats[-1]:.0f}s")
    print(f"results: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
