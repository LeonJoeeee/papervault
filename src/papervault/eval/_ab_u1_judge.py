"""MiMo U1 faithfulness judge (via the LiteLLM gateway) — for the extraction-prompt A/B.

Reads a u1 edges jsonl (from _ab_upstream.py sample-u1): {head, tail, description, source_chunks[]}.
For each edge asks MiMo whether the relationship_description is FAITHFUL — i.e. explicitly stated /
directly supported by the source chunk text, not an inferred or fabricated connection. U1 = fraction
faithful. Same gateway/model as the downstream judge (judge_mimo); needs KS_VIRTUAL_KEY exported.

Usage:  set -a; . ./.env; set +a
        uv run python experiments/eval/_ab_u1_judge.py /tmp/u1_new.jsonl
"""
import asyncio, json, os, re, sys

GATEWAY = os.getenv("KS_GATEWAY_URL", "http://127.0.0.1:4000/v1")
VKEY = os.getenv("KS_VIRTUAL_KEY", "")
MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

SYS = ("You audit one edge of a scientific knowledge graph for FAITHFULNESS. You are given the edge "
       "(head entity, tail entity, and its relationship_description) and the SOURCE TEXT chunks the edge "
       "was extracted from. Decide whether the relationship_description is FAITHFUL: it must report a "
       "connection between head and tail that the source text EXPLICITLY states or directly implies — a "
       "paraphrase of something actually in the text, NOT an inference from mere co-occurrence, NOT "
       "background knowledge, NOT a fabricated mechanism. If the source does not actually state this "
       "relationship, it is unfaithful. Output ONLY JSON: {\"faithful\": true|false, \"reason\": \"<short>\"}.")


def _extract_json(s: str):
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            pass
    return None


async def _judge(client, sem, e):
    user = (f"head: {e['head']}\ntail: {e['tail']}\nrelationship_description: {e.get('description')}\n"
            f"keywords: {e.get('keywords')}\n\nSOURCE TEXT (the only evidence):\n" +
            "\n---\n".join(e.get("source_chunks", [])))
    async with sem:
        for _ in range(3):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                    max_tokens=4000, temperature=0)
                obj = _extract_json(r.choices[0].message.content or "")
                if obj is not None and "faithful" in obj:
                    return bool(obj["faithful"])
            except Exception:
                await asyncio.sleep(2)
    return None  # unparseable → excluded


async def main():
    path = sys.argv[1]
    edges = [json.loads(l) for l in open(path)]
    if not VKEY:
        raise SystemExit("KS_VIRTUAL_KEY not set — `set -a; . ./.env; set +a` first")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=GATEWAY, api_key=VKEY, timeout=300, max_retries=0)
    sem = asyncio.Semaphore(int(os.getenv("AB_JUDGE_CONC", "20")))
    verdicts = await asyncio.gather(*[_judge(client, sem, e) for e in edges])
    scored = [v for v in verdicts if v is not None]
    n_faith = sum(1 for v in scored if v)
    u1 = n_faith / len(scored) if scored else 0.0
    print(json.dumps({"file": path, "edges": len(edges), "judged": len(scored),
                      "unparseable": len(edges) - len(scored), "faithful": n_faith,
                      "U1_faithfulness": round(u1, 4)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
