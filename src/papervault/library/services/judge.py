"""Question-driven scored judges for ``search_papers`` (LLM2 ingest + LLM3 return).

Two judges over one shared machinery (batched + parallel + retry + ABSOLUTE scoring):

- ``judge_ingest``  (LLM2 / ingest gate): for each *external* candidate, assign a
  ``domain.md`` tier AND an ``is_paper`` backstop. Off-domain (Tier 3) is rejected.
  ``is_paper`` is decided by the caller as **metadata-first, LLM-backstop**: source
  ``publication_types`` is authoritative when present, but it's frequently empty (CORE
  never sets it; some Crossref/OpenAlex records don't) — so the LLM's ``is_paper`` fills
  the gap rather than letting empty metadata fail-open (datasets in) or fail-closed (CORE out).
- ``judge_return``  (LLM3 / return gate): for each candidate (library + just-ingested),
  an ABSOLUTE 0-1 relevance score vs the user's query intent. Caller thresholds +
  sorts + truncates to the requested count.

Design (principle 8, question-driven): the LLM answers a fixed question set + scores; it does
NOT make the final decision — the caller gates / thresholds / sorts mechanically.

Failure handling (✅ user 2026-05-31 — NEVER fail-open): a batch's LLM call is RETRIED
with exponential backoff; on persistent failure the batch is **dropped** (contributes
no judgments) — ingest-side that means "not ingested" (stop-the-bleed: never let an unjudged paper
into the library), return-side "not returned". We never fabricate ``keep=True``.

See ``PAPER_LIBRARY_SDD.md`` §5.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

from . import concurrency

logger = logging.getLogger(__name__)

BATCH_SIZE = 30
MAX_RETRIES = 2          # → up to 3 attempts per batch
_BACKOFF_BASE = 2.0
DOMAIN_PATH = Path(__file__).parent.parent / "domain.md"

# Tiers that count as in-domain (keep for ingest); "3" = off-domain reject.
INGEST_TIERS = {"1A", "1B", "1C", "2A", "2B", "2C"}


# ───────────────────────── shared machinery ─────────────────────────

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
_ITEM_RE = re.compile(r"\{[^{}]*\}")  # flat (non-nested) item object


def _extract_json_obj(raw: str) -> dict:
    """Parse a gate LLM reply into ``{"items": [...]}``.

    Fast path: strip markdown ``` fences, grab the greedy ``{...}`` envelope, json.loads it
    (unchanged behaviour whenever the whole batch is well-formed).

    SALVAGE path: a *single* malformed item in a 30-batch used to make ``json.loads`` raise,
    which the caller (``_scored_judge._one``) turns into a fail-CLOSED DROP of ALL 30
    judgments — even though the per-item parsers (``_parse_ingest`` / ``_parse_score``)
    already tolerate a bad item. So when the envelope parse fails, regex-extract each FLAT
    ``{...}`` item carrying an ``"i"`` index and json.loads each INDEPENDENTLY, returning only
    the survivors → the per-item loop then drops ONLY the genuinely-broken item, not the batch.
    Flat per-item (NOT brace-balancing): one malformed middle item breaks balance for
    everything after it (brace-walk recovers ~15/30 vs flat ~23/30 on real MiMo output)."""
    text = _FENCE_RE.sub("", raw or "")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in LLM response: {(raw or '')[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        items = [obj for chunk in _ITEM_RE.findall(text) if '"i"' in chunk
                 for obj in (_try_loads(chunk),) if obj is not None]
        if not items:
            raise  # nothing salvageable → genuine failure; caller DROPs (counted)
        return {"items": items}


def _try_loads(chunk: str):
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


def _candidate_to_item(i: int, c: dict) -> dict:
    """Lean projection of a candidate sent to the judge LLM.

    Shared by BOTH gates and ALWAYS emits ``citation_count`` + ``is_review`` (§7): the
    return rubric's soft prefs reference them, so they must be visible. The ingest prompt
    ignores them (slightly larger payload, contamination-safe — judge inputs are
    LLM-payload only, never persisted). Do NOT split per-gate.
    """
    return {
        "i": i,
        "title": c.get("title") or "",
        "venue": c.get("venue") or "",
        "year": c.get("year"),
        "authors": (c.get("authors") or [])[:6],
        "abstract": c.get("abstract") or "",
        "citation_count": int(c.get("citation_count") or 0),
        "is_review": bool(c.get("is_review", False)),
    }


async def _call_with_retry(system_prompt: str, user_prompt: str, llm: Any) -> str:
    """One LLM call, retried with exponential backoff. Raises if all attempts fail."""
    last: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with concurrency.llm_sem:  # per-CALL LLM concurrency cap (§0/§6/§7)
                return await asyncio.to_thread(
                    llm.call,
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
                )
        except Exception as e:  # noqa: BLE001 — transient LLM hiccup; retry
            last = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
    raise RuntimeError(f"judge batch failed after {MAX_RETRIES + 1} attempts: {last}")


async def _scored_judge(
    candidates: list[dict],
    system_prompt: str,
    build_user_prompt: Callable[[list[dict]], str],
    parse_items: Callable[[str, set[int]], dict[int, dict]],
    *,
    batch_size: int,
    llm: Any,
) -> tuple[dict[int, dict], int]:
    """Batched + parallel scored judgment. Returns ``({global_idx: judgment},
    dropped_batches)``. A batch that fails all retries — OR whose response parses
    OK but matched ZERO expected indices (the index-rebasing silent-drop mode,
    §6) — contributes NOTHING (conservative drop, never fail-open) and counts as
    one DROPPED batch. The parse-OK-zero-matched case emits a ``logging.warning``
    (it is otherwise invisible — un-retried, un-logged); ``dropped_batches`` is
    threaded out to the caller and surfaced as ``judge_batches_dropped`` (§8) so a
    silent recall loss is machine-visible."""
    if not candidates:
        return {}, 0
    indexed = list(enumerate(candidates))
    batches = [indexed[k:k + batch_size] for k in range(0, len(indexed), batch_size)]

    async def _one(batch: list[tuple[int, dict]]) -> tuple[dict[int, dict], int]:
        items = [_candidate_to_item(i, c) for i, c in batch]
        expected = {i for i, _ in batch}
        try:
            raw = await _call_with_retry(system_prompt, build_user_prompt(items), llm)
            parsed = parse_items(raw, expected)
        except Exception as e:  # noqa: BLE001 — persistent LLM failure OR unparseable response: DROP
            # fail-CLOSED: a refusal / empty / truncated / malformed-JSON reply must
            # be a conservative DROP (counted), NEVER an exception that crashes search.
            logger.error("judge: dropping batch of %d (no fail-open): %s", len(batch), e)
            return {}, 1
        if not parsed:
            # Parsed cleanly but nothing matched the expected global indices —
            # the index-rebasing later-batch silent-drop mode (§6). Otherwise
            # un-retried + un-logged; surface it as a WARN + a dropped batch.
            logger.warning(
                "judge: batch of %d parsed OK but 0 items matched expected idx "
                "(possible index-rebasing); dropping", len(batch),
            )
            return {}, 1
        return parsed, 0

    merged: dict[int, dict] = {}
    dropped = 0
    for r, d in await asyncio.gather(*(_one(b) for b in batches)):
        merged.update(r)
        dropped += d
    return merged, dropped


def _read_domain_md() -> str:
    try:
        return DOMAIN_PATH.read_text(encoding="utf-8")
    except OSError as e:
        logger.error("judge: cannot read domain.md at %s: %s", DOMAIN_PATH, e)
        return "(domain context unavailable)"


# ───────────────────────── LLM2: judge_ingest (domain tier) ─────────────────────────

_INGEST_SYSTEM = """You screen each candidate for INGEST into a **space physics + AI4Science**
research library. Judge each candidate on its OWN merit (absolute), not relative to the others
in this batch.

For EACH candidate output, in order:
  1. ``reason`` (<= 20 words): briefly note (a) whether it's a genuine research paper vs a
     non-paper record, and (b) which domain tier it fits.
  2. ``is_paper`` (true/false): is it a research paper / journal article / review / conference
     paper / preprint? Answer **false** for non-paper records: datasets, software, whole books,
     errata / corrections, retractions, editorials, pure news items, conference programs.
  3. ``tier`` (one of 1A / 1B / 1C / 2A / 2B / 2C / 3) per the rubric below. Tier 3 = off-domain.

NOTE on ``is_paper``: this is a **backstop**. The caller trusts source metadata when it's present
and only falls back to your ``is_paper`` when metadata is missing — so judge it honestly from the
title / venue / abstract, independent of any metadata.

==== Domain rubric (start) ====
{domain_md}
==== Domain rubric (end) ====

Output ONLY one JSON object:
{{"items": [{{"i": <int>, "reason": "<str>", "is_paper": <bool>, "tier": "<1A|1B|1C|2A|2B|2C|3>"}}, ...]}}
The ``i`` must match the candidate's ``i``.
"""


def _parse_ingest(raw: str, expected: set[int]) -> dict[int, dict]:
    parsed = _extract_json_obj(raw)
    out: dict[int, dict] = {}
    for item in parsed.get("items", []):
        try:
            i = int(item["i"])
            if i not in expected:
                continue
            out[i] = {
                "reason": str(item.get("reason", ""))[:200],
                "tier": str(item.get("tier", "?")).strip()[:4],
                # LLM's is_paper — a BACKSTOP used by the caller only when source
                # metadata can't decide. Default True (lenient) if the LLM omits it;
                # the domain-tier gate is the primary contamination guard.
                "llm_is_paper": bool(item.get("is_paper", True)),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return out


async def judge_ingest(candidates: list[dict], *, llm: Any = None) -> tuple[dict[int, dict], int]:
    """LLM2 — domain tier + is_paper backstop for each candidate. Returns
    ``({global_idx: {reason, tier, ingest_ok, llm_is_paper}}, judge_batches_dropped)``
    (judgments only for successfully judged; the int is the count of parse-OK-zero-matched
    or all-retries-failed DROPPED batches — surfaced as ``judge_batches_dropped`` §8).

    - ``ingest_ok = tier in INGEST_TIERS`` — the **domain** gate (Tier-3 off-domain rejected).
    - ``llm_is_paper`` — the LLM's read of "is this a real paper". This is a **backstop**: the
      caller trusts source metadata (``publication_types``) when present and falls back to this
      only when metadata is empty/missing (CORE never sets it; some Crossref/OpenAlex records
      don't either). The caller combines: ``ingest = ingest_ok AND is_paper`` where
      ``is_paper = metadata-if-conclusive else llm_is_paper``."""
    if not candidates:
        return {}, 0
    if llm is None:
        from ..llm import get_llm
        llm = get_llm()
    system = _INGEST_SYSTEM.format(domain_md=_read_domain_md())

    def build_user(items: list[dict]) -> str:
        return f"Candidates ({len(items)}):\n{json.dumps(items, ensure_ascii=False)}"

    judged, dropped = await _scored_judge(
        candidates, system, build_user, _parse_ingest, batch_size=BATCH_SIZE, llm=llm)
    for j in judged.values():
        j["ingest_ok"] = j.get("tier") in INGEST_TIERS
    logger.info("judge_ingest: %d/%d judged, %d in-domain, %d llm-says-paper, %d batches dropped",
                len(judged), len(candidates),
                sum(1 for j in judged.values() if j["ingest_ok"]),
                sum(1 for j in judged.values() if j.get("llm_is_paper")), dropped)
    return judged, dropped


# ───────────────────────── LLM3: judge_return (relevance) ─────────────────────────

_RETURN_SYSTEM = """You score how well each candidate paper matches the user's QUERY INTENT for a
researcher in space physics + AI4Science. Judge each on its OWN merit (ABSOLUTE
scale), not relative to the batch.

RELEVANCE = UTILITY-TO-INTENT, not topical or field overlap. Score by whether a
researcher DOING THIS INTENT would actually open the paper and get something they can
cite, build on, or borrow a method from. Same field / shared keywords is evidence,
never the verdict. Before scoring a paper high, name what it does NOT give this intent.

The query intent is often MULTI-PART (several sub-topics, a method × an application,
a main thread + a niche side-thread). Decompose it yourself; if a list of intended
sub-topics is given, treat each as a genuine sub-part the user cares about.

SCORE EACH PAPER ON ITS BEST-MATCHING SUB-PART — not on how much of the whole query
it covers. A paper that FULLY answers ONE genuine intended sub-topic is HIGHLY
relevant (0.7-0.9) EVEN IF it ignores the other sub-topics. Do NOT penalize a paper
for being focused. Only a paper answering the WHOLE intent reaches 0.9+. But do NOT
reward mere field membership: same broad area, no sub-part answered → 0.2-0.4.

CROSS-DOMAIN METHOD TRANSFER is real utility: a paper from ANOTHER field whose
method/technique the researcher can lift into this intent (a PINN / operator / inverse
trick from fluids, imaging, etc. for a physics-ML intent) scores on its transferability
(0.7-0.9), NOT cut for the wrong field. But "also ML / also a PDE / shared vocabulary"
with no actually-borrowable mechanism is NOT transfer → 0.2-0.4.

SOFT PREFERENCES (gentle nudges, NEVER a cut): given soft prefs (recent via the year
field+year_window / well-cited via citation_count / reviews via is_review), an on-topic
paper missing one loses a little, never its relevance; prefs only break near-ties. When
a year_window is given, a candidate with an UNKNOWN year (null) is recency-unverifiable
— do not boost it for recency; rank it on topical merit alone (still NEVER cut it).

For EACH candidate, output: reason (<=20 words: name the single sub-part it best
answers + what it gives the intent, or "no sub-part: same field only") then score:
  0.9+    : answers the WHOLE intent (or one sub-part + meaningfully contributes to others)
  0.7-0.9 : fully answers >=1 intended sub-topic, or a borrowable cross-domain method (focused paper nailing one thread belongs HERE)
  0.4-0.7 : GENUINELY partial — gives real but incomplete payload toward a sub-topic the intent names
  0.2-0.4 : same field, addresses none of the sub-parts (incl. adjacent / keyword-overlapping papers with no real payload for THIS intent — the precision floor; use it readily)
  0.0-0.2 : off-topic
Output ONLY: {"items":[{"i":<int>,"reason":"<str>","score":<float>}, ...]}
The "i" MUST be the candidate's provided "i" (do NOT renumber).
"""


def _parse_score(raw: str, expected: set[int]) -> dict[int, dict]:
    parsed = _extract_json_obj(raw)
    out: dict[int, dict] = {}
    for item in parsed.get("items", []):
        try:
            i = int(item["i"])
            if i not in expected:
                continue
            score = max(0.0, min(1.0, float(item.get("score", 0.0))))
            out[i] = {"reason": str(item.get("reason", ""))[:200], "score": score}
        except (KeyError, ValueError, TypeError):
            continue
    return out


async def judge_return(
    candidates: list[dict],
    intent: str,
    *,
    search_terms: list[str] | None = None,
    filters: dict | None = None,
    llm: Any = None,
) -> tuple[dict[int, dict], int]:
    """LLM3 — absolute 0-1 relevance score vs ``intent``. Returns
    ``({global_idx: {reason, score}}, judge_batches_dropped)`` (judgments only for
    successfully judged ones; the int counts parse-OK-zero-matched or all-retries-failed
    DROPPED batches — surfaced as ``judge_batches_dropped`` §8). Caller thresholds + sorts
    + truncates.

    The raw ``query`` is passed DIRECTLY as ``intent`` (no alias — verbatim-ness automatic).
    ``search_terms`` (the parser's distinct sub-topics) are fed to the user prompt as the
    "intended sub-topics" the §7 rubric scores each paper's BEST-matching sub-part against.
    ``filters`` carries the SOFT prefs (citation_pref/review_pref) + the year window — gentle
    nudges only, NEVER a cut (§7). Both are keyword-only with safe defaults so older
    ``judge_return(candidates, intent, *, llm=)`` calls keep working (no ``aspects`` arg)."""
    if not candidates:
        return {}, 0
    search_terms = search_terms or []
    filters = filters or {}
    if llm is None:
        from ..llm import get_llm
        llm = get_llm()

    def build_user(items: list[dict]) -> str:
        ymin, ymax = filters.get("year_min"), filters.get("year_max")
        has_window = ymin is not None or ymax is not None
        window = f"from {ymin or 'any'} to {ymax or 'any'}" if has_window else "none"
        soft = (f"citation_pref={filters.get('citation_pref')}, "
                f"review_pref={filters.get('review_pref')}, "
                f"year_window={window}")
        return (f"User query intent: {intent}\n"
                f"The intended sub-topics include: {', '.join(search_terms)}\n"
                f"Soft preferences: {soft}\n\n"
                f"Candidates ({len(items)}):\n{json.dumps(items, ensure_ascii=False)}")

    judged, dropped = await _scored_judge(
        candidates, _RETURN_SYSTEM, build_user, _parse_score, batch_size=BATCH_SIZE, llm=llm)
    logger.info("judge_return: %d/%d judged, %d batches dropped",
                len(judged), len(candidates), dropped)
    return judged, dropped
