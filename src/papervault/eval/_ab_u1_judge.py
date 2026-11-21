"""MiMo U1 faithfulness judge (via the configured LLM endpoint) — for the extraction-prompt A/B.

Reads a u1 edges jsonl (from _ab_upstream.py sample-u1): {head, tail, description, source_chunks[]}.
For each edge asks the judge whether the relationship_description is FAITHFUL — i.e. explicitly
stated / directly supported by the source chunk text, not an inferred or fabricated connection.
U1 = fraction faithful. Uses explicit ``KS_GATEWAY_URL`` / ``KS_VIRTUAL_KEY`` if set, otherwise
falls back to ``papervault.config``.

Usage:  python -m papervault.eval._ab_u1_judge /tmp/u1_new.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from papervault.llm_routing import (
    create_openai_chat_completion_with_fallback,
    http_status_code,
    parse_targets,
    route,
)


def _resolve_endpoint() -> tuple[str, str]:
    """Resolve judge endpoint URL and API key.

    If either legacy ``KS_GATEWAY_URL`` or ``KS_VIRTUAL_KEY`` is set, both
    values resolve strictly within the legacy/gateway domain (never mixing
    direct-provider credentials with a gateway URL or vice versa). Otherwise
    fall back to ``papervault.config`` (gateway vs direct transport).
    """
    from papervault import config

    legacy_url = os.getenv("KS_GATEWAY_URL", "").strip()
    legacy_key = os.getenv("KS_VIRTUAL_KEY", "").strip()
    if legacy_url or legacy_key:
        url = legacy_url or "http://127.0.0.1:4000/v1"
        key = legacy_key or (config.GATEWAY_KEY.strip() if config.USE_GATEWAY else "")
        return url, key
    if config.USE_GATEWAY:
        return config.GATEWAY_URL.strip(), config.GATEWAY_KEY.strip()
    return config.LLM_BASE_URL.strip(), config.LLM_API_KEY.strip()


GATEWAY, VKEY = _resolve_endpoint()

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
    targets = parse_targets(route("eval_judge"))
    async with sem:
        for _ in range(3):
            try:
                r = await create_openai_chat_completion_with_fallback(
                    client,
                    targets,
                    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                    temperature=0,
                )
                obj = _extract_json(r.choices[0].message.content or "")
                if obj is not None and "faithful" in obj:
                    return bool(obj["faithful"])
            except Exception as exc:  # noqa: BLE001
                if http_status_code(exc) in {400, 401, 403, 422}:
                    raise
                await asyncio.sleep(2)
    return None  # unparseable → excluded


async def main():
    path = sys.argv[1]
    edges = [json.loads(line) for line in open(path)]
    from papervault import config

    gateway, vkey = _resolve_endpoint()
    if not vkey:
        raise SystemExit(
            "No judge API key configured — set KS_VIRTUAL_KEY or configure "
            "PAPERVAULT_LLM_GATEWAY_KEY / PAPERVAULT_LLM_API_KEY in .env"
        )
    if config.USE_GATEWAY and not gateway:
        raise SystemExit(
            "PAPERVAULT_LLM_GATEWAY_URL is empty while gateway mode is enabled"
        )
    from openai import AsyncOpenAI

    client_kwargs: dict[str, object] = {"api_key": vkey, "timeout": 300, "max_retries": 0}
    if gateway:
        client_kwargs["base_url"] = gateway
    client = AsyncOpenAI(**client_kwargs)
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
