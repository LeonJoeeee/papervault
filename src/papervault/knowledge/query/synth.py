"""S3 synth: prose synthesis over aquery_data structured subgraph (SDD §6.4).

KS-written exit-LLM stage. Takes the structured `data` dict from `aquery_data`
(entities / relationships / chunks — NOT pre-LLM'd) + the NL intent, and emits
prose with inline [paper_key] cites (operator sources cite as [textbook:Key], issue #122).

DECOUPLED from the out-feed's source-of-truth: this module only produces the
PROSE. `cited_papers` is aggregated separately in aquery.py from
`data.references[].file_path` (the chunk-level, untruncated, deduped reference
list — SDD §4.3 F12), NEVER regex-scraped from this prose (the v2
_extract_cited_papers mistake: an LLM that forgets to tag a key would drop it,
and a key-format regex silently misses legit keys).

credibility (SDD §12): derived from the file_path ingest_source prefix
(textbook→established, paper→empirical/interpretive, web→preliminary). web and
preliminary content MUST be surfaced as tentative. A paper chunk from an abstract-only doc
(#144, ingest/abstract_doc.py) is tagged credibility=abstract-only instead, and rule 6b bounds
what it may be cited for.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from openai import APIConnectionError

from papervault.knowledge.ingest.abstract_doc import is_abstract_text, strip_abstract_header
from papervault.knowledge.ingest.operator_docs import strip_section_suffix
from papervault.knowledge.store.llm import StreamTruncated, _error_code, mimo_complete
from papervault.llm_routing import route

logger = logging.getLogger("ks.query.synth")

# Sentinel prefix returned (instead of raising) when the synth MiMo call fails/times out, so
# the out-feed still yields a well-formed {answer, cited_papers, kb_coverage}. Exported so
# callers/tests can distinguish "synth produced real prose" from "synth integrally failed but
# retrieval succeeded" — a non-empty `answer` is NOT proof of a real synthesized answer.
SYNTH_FAILED_PREFIX = "(synthesis LLM failed"
# Full honesty fallback returned when synth INTEGRALLY fails — exception/timeout OR an
# empty/whitespace-200 (S16, SDD §6.4: a 200 carrying no usable text is a SILENT gateway/
# model failure; the caller MUST treat empty/whitespace as failure, not as a blank answer).
# Retrieval still succeeded, so point the caller at cited_papers to pull the sources directly.
_SYNTH_FAILED_MSG = (
    f"{SYNTH_FAILED_PREFIX}; the knowledge base did return relevant sources — "
    "see cited_papers and pull them via get_paper.)"
)

# MiMo is a REASONING model: it spends a large chain-of-thought budget BEFORE the visible
# answer, billed from the SAME max_tokens budget. The synth task (deep multi-paper cited answer)
# is far larger than the sibling decompose JSON-array task that already needed 8000 just to fit
# its tiny output after the CoT burn — so synth needs MORE headroom or a deep answer is cut
# mid-sentence, dropping trailing [key] cites and corrupting both the prose and citation metrics.
# Raising the ceiling is purely protective: a short answer still stops early on its own.
_SYNTH_MAX_TOKENS = 16000

# Synth's OWN business deadline (Phase 0, 2026-06-04 — docs/history/2026-06-04-gateway/2026-06-04-llm-gateway.md). A deep
# 16k-token cited synthesis legitimately takes ~600s under contention (~20-45 tok/s), so the old
# 180s ceiling killed every long answer (the transport-layer 120s did too — both had to be lifted;
# raising only one just moved the death from 120s→180s). This is the per-CALLER business deadline
# that the transport backstop (KS_LLM_CLIENT_TIMEOUT ~15min) deliberately leaves to the caller. env-tunable.
# 840 (not 900) keeps it STRICTLY below the transport backstop (KS_LLM_CLIENT_TIMEOUT default
# 900s) so synth's own business deadline always wins the race — the transport stays a pure
# dead-connection catch. 840s ≫ a realistic 16k-token answer (~533-600s); a maxed-out worst case
# still funnels to the same clean SYNTH_FAILED fallback as before.
_SYNTH_TIMEOUT_S = float(os.getenv("KS_SYNTH_TIMEOUT_S", "840"))


_SYNTH_SYSTEM = """You are a research-knowledge synthesizer for a senior researcher.
Read the retrieved knowledge-graph entities, relationships, and source chunks, and
produce a substantive answer to the user's intent.

Hard requirements (these are the only constraints):
1. Cite EVERY substantive claim inline — every sentence that asserts a fact, number,
   mechanism, or comparison must carry at least one [paper_key]; a factual sentence with
   no inline cite is a defect. Frame/transition sentences need no cite. The cite is the
   source key shown in each chunk's source label, unchanged, wrapped in square brackets
   (e.g. "[Reames2023]"; a textbook/notebook/web key keeps its colon, e.g.
   "[textbook:Griffiths]"), no author names, no separate years.
2. Use ALL the relevant retrieved material, not just the single most-relevant chunk. When
   several papers bear on the intent, SYNTHESIZE across them — corroborate where they agree
   (cite each, e.g. "[A2021][B2022]"), and surface disagreements or complementary angles
   explicitly rather than reporting only one source. Cover the distinct sub-points the
   retrieved set supports; do not collapse a multi-paper body of evidence into a single-paper
   summary. (Requirement 3's honesty clause still bounds this — refuse cleanly on thin coverage.)
3. Be honest: if the retrieved material doesn't cover the intent well, say so
   cleanly (do not pad with hand-wavy claims).
4. Explain MECHANISM where the material supports it (not just "X fails" — say WHY).
5. If the intent mentions a specific context (e.g. "I do Voyager XPINN inversion"),
   address that context explicitly somewhere in the answer.
6. Respect each chunk's credibility tag. Weight established / empirical content
   (textbook, peer-reviewed results) for factual claims. Content tagged
   credibility=preliminary (a paper's speculation, a preprint, a web/satellite page)
   MUST be presented AS tentative — never asserted as settled. Flag it inline
   (e.g. "one preprint suggests…", "per NASA's spec page…").
6b. A chunk tagged credibility=abstract-only is only that paper's abstract (its full
   text is not in the knowledge base): cite it only for what its abstract states, never
   for methods, numbers or conclusions the abstract does not contain.

Format is your call: prose, bullets, sections, table — whatever fits the question +
the depth of coverage. Length is your call too: brief if coverage is thin or the
question is narrow; deep if coverage is rich + the question is broad."""

# V-SR variant (2026-06-11, #5 loop): STRICT refusal discipline, flag-gated so the baseline
# path is untouched unless KS_SYNTH_STRICT_REFUSAL=1. Motivation (BASELINE_FULLCORPUS.md):
# measured trap behaviour shows most "refusals" are refuse-then-explain hybrids — the answer
# says "the retrieved material does not cover this" and then answers anyway from model-internal
# knowledge. Real refusal discipline ≈ 1/3. This rule hardens requirement 3 into a hard stop.
_STRICT_REFUSAL_RULE = """
3b. HARD STOP after a coverage refusal: if you determine the retrieved material does not
   cover the intent (requirement 3), say so and STOP. You may briefly state what the
   retrieved material IS about (one or two sentences, to show the mismatch), but you MUST
   NOT then answer the question from your own background knowledge — no formulas, no
   mechanisms, no numbers responsive to the intent that do not come from the retrieved
   chunks. An un-cited substantive answer after a refusal is a defect, not a service."""

if os.getenv("KS_SYNTH_STRICT_REFUSAL", "1") == "1":   # #5 default-promote 2026-06-14: V-SR ON by default (set =0 to disable)
    _SYNTH_SYSTEM = _SYNTH_SYSTEM.replace(
        "4. Explain MECHANISM",
        _STRICT_REFUSAL_RULE.strip("\n") + "\n4. Explain MECHANISM",
    )

# Coverage variant (#5 drill L3+L4, 2026-06-14, flag KS_SYNTH_COVERAGE=1). The drill found the synth
# over-optimizes a coherent thesis and SHEDS distinct sub-claims that ARE verbatim in the served
# chunks (present-not-used), and occasionally upgrades a source's stated uncertainty into a settled
# verdict (a faithfulness inversion). This rule forces enumerate-then-cover + preserve-uncertainty,
# BOUNDED by req 3/3b (every claim must trace to a served chunk → does NOT license hallucination to
# fill coverage). Default OFF = byte-identical baseline. Composes with strict-refusal (the floor).
_COVERAGE_RULE = """
7. COVERAGE & UNCERTAINTY (this does NOT relax requirement 3/3b — every claim below MUST trace to a
   retrieved chunk; never add background-knowledge claims just to satisfy coverage):
   - Before writing, enumerate every DISTINCT substantive sub-claim the retrieved chunks support:
     mechanism; named quantitative head-to-head comparisons (give the specific numbers / table rows,
     not just the qualitative direction); applicability / when-each-approach-wins; and any cited
     paper's own qualifying or contradicting conclusion. Cover each, or explicitly note it absent.
   - Every retrieved paper with on-topic content must be cited at least once, or explicitly dismissed.
   - PRESERVE UNCERTAINTY: when a source states a finding is unresolved, or that two hypotheses are
     currently indistinguishable, keep it unresolved — do not upgrade it to a settled conclusion, and
     state what new evidence would resolve it. Never contradict a cited paper's own headline conclusion."""

if os.getenv("KS_SYNTH_COVERAGE") == "1":
    _SYNTH_SYSTEM = _SYNTH_SYSTEM.rstrip() + "\n" + _COVERAGE_RULE


# ingest_source prefix → credibility band (SDD §12). paper defaults to empirical
# (peer-reviewed result is the common case; the synth LLM down-weights a paper's
# own speculation per requirement 5).
_CRED_BY_SOURCE = {
    "textbook": "established",
    "paper": "empirical",
    "web": "preliminary",
    # notebook (#47): the lab's own executor notebooks — unpublished, in-progress reasoning.
    # Lowest band: preliminary (never let a notebook out-rank a peer-reviewed paper/textbook).
    "notebook": "preliminary",
}


# #144: credibility band of a chunk from an abstract-only paper doc (rule 6b).
ABSTRACT_ONLY_CRED = "abstract-only"


def _source_label(file_path: str) -> tuple[str, str]:
    """(citeable_key_label, credibility) from a chunk/entity file_path.

    Two provenance conventions coexist:
      - paper distill: 'paper/<key>' (1.4) or the bare '<key>' basename (1.5);
        both expose the same citeable token with credibility=empirical.
      - operator docs (#45/#47): the COLON provenance key itself ('textbook:AuthorYear',
        'notebook:idea-scope') — slash-free so it survives LightRAG 1.5 basenaming. The
        source class in the prefix sets the credibility band (textbook→established); the
        whole key is the label, cited bracketed with its colon ([textbook:Key], #122). A multi-section
        doc carries a per-section file_path (`<key>#s<N>`, issue #79) — strip the `#s<N>` so
        the label attributes to the WHOLE book/notebook, not one section.
    """
    if not file_path.strip() or file_path in {"unknown", "unknown_source", "paper/"}:
        return ("unknown", "preliminary")
    # Operator-doc colon key: '<source>:<id>' with no slash (e.g. 'textbook:Schlickeiser2002').
    if "/" not in file_path and ":" in file_path:
        source = file_path.split(":", 1)[0]
        if source in _CRED_BY_SOURCE:
            # #79: drop the per-section suffix so the cite is the book-level key.
            return (strip_section_suffix(file_path), _CRED_BY_SOURCE[source])
    # Match aquery._cited_papers: operator sources first, then both paper eras.
    if file_path.startswith("paper/"):
        return (file_path.split("/", 1)[1], _CRED_BY_SOURCE["paper"])
    if "/" not in file_path:
        if ":" in file_path:
            return ("unknown", "preliminary")
        # Distill owns these basenames; do not impose an author/year key regex.
        return (file_path, _CRED_BY_SOURCE["paper"])
    source, _, sid = file_path.partition("/")
    cred = _CRED_BY_SOURCE.get(source, "preliminary")
    return (f"{source}:{sid}", cred)


def _build_prompt(data: dict[str, Any], intent: str) -> str:
    """Assemble the synth prompt from aquery_data's structured `data` dict.

    Sends entities (typed nodes), relationships (edges), and source chunks — each
    chunk prefixed with its citeable source label + credibility band so the LLM can
    cite real [paper_key]s and honor tentativeness.
    """
    blocks: list[str] = []

    entities = data.get("entities") or []
    if entities:
        ent_lines = []
        for e in entities:
            name = e.get("entity_name", "?")
            etype = e.get("entity_type", "?")
            desc = (e.get("description") or "").strip()
            ent_lines.append(f"- ({etype}) {name}: {desc}")
        blocks.append("Knowledge-graph entities:\n" + "\n".join(ent_lines))

    rels = data.get("relationships") or []
    if rels:
        rel_lines = []
        for r in rels:
            src = r.get("src_id", "?")
            tgt = r.get("tgt_id", "?")
            desc = (r.get("description") or "").strip()
            rel_lines.append(f"- {src} — {tgt}: {desc}")
        blocks.append("Knowledge-graph relationships:\n" + "\n".join(rel_lines))

    chunks = data.get("chunks") or []
    if chunks:
        chunk_blocks = []
        for c in chunks:
            label, cred = _source_label(c.get("file_path") or "")
            content = (c.get("content") or "").strip()
            if label != "unknown" and cred == _CRED_BY_SOURCE["paper"] and is_abstract_text(content):
                # #144: the label carries the class; the header line (square brackets) is dropped so
                # it never reads as a citation lookalike next to the one bracketed-key rule (#122).
                cred = ABSTRACT_ONLY_CRED
                content = strip_abstract_header(content)
            if label == "unknown":
                header = f"(no citeable source key; do not invent a citation | credibility: {cred})"
            else:
                header = f"(source key: {label} | credibility: {cred})"
            chunk_blocks.append(f"{header}\n{content}")
        blocks.append(
            "Source chunks (each chunk below has a source key or an explicit no-citation marker; to "
            "cite it, wrap ONLY the key, unchanged, in square brackets, e.g. [Reames2023] for a "
            "paper or [textbook:Baumjohann2012] for a textbook/notebook/web source (keep its "
            "colon) — never write the word paper_key or the credibility tag inside the brackets):\n"
            + "\n\n---\n\n".join(chunk_blocks)
        )

    context = "\n\n====\n\n".join(blocks) if blocks else "(no material retrieved)"

    return f"""User intent:
{intent}

Retrieved knowledge:
====
{context}
====

Answer the intent using the retrieved knowledge above. Cite EVERY substantive claim
inline with the bracketed source key shown in each chunk's source label.
Use ALL the relevant material and synthesize ACROSS sources — don't collapse a multi-paper body
of evidence into a single-paper summary. Pick whatever structure (prose / bullets /
sections / table) and length best fits the question + the depth of coverage. If the
material doesn't support the intent well, say so cleanly."""


# Bounded internal retry for the synth MiMo call (issue #70). By the time synth runs, RETRIEVAL
# already SUCCEEDED — so a single TRANSIENT synth hiccup (a timeout, a connection reset, a 5xx /
# 429 from the gateway, or a silent empty-200) must NOT burn that successful retrieval and push
# retry logic onto every caller. Retry a few times with exponential backoff BEFORE the honest
# SYNTH_FAILED fallback. Mirrors the library judge path (services/judge.py), which already retries
# transient LLM hiccups. DETERMINISTIC failures (400 validation, or an all-deployments-blocked
# 401/403 — the KeyPool's terminal "All configured LLM keys failed" RuntimeError, and the gateway's
# raw 4xx) can't recover on an immediate re-send, so they FAIL FAST to the same fallback. A true
# full outage of transient errors still exhausts the retries and then falls back — that terminal
# graceful-degradation behavior is unchanged (issue #70: "a true full outage SHOULD still exhaust
# retries and fall back"). Both knobs env-tunable (a test shrinks the backoff to 0).
_SYNTH_MAX_RETRIES = int(os.getenv("KS_SYNTH_MAX_RETRIES", "2"))     # → up to 3 attempts
_SYNTH_BACKOFF_BASE = float(os.getenv("KS_SYNTH_BACKOFF_BASE", "2.0"))


def _is_transient_synth_status(code: int | None) -> bool:
    """HTTP status → is it worth re-sending the SAME synth request? 429 (rate-limit) and 5xx
    (server / gateway) recover on retry; 4xx (400 validation / 401 / 403 all-blocked / 404 / 422)
    do NOT. Mirrors store/llm.py's transient/permanent split (_PERMANENT_CODES = {401, 403})."""
    if code is None:
        return False
    return code == 408 or code == 429 or 500 <= code <= 599


def _is_transient_synth_error(exc: Exception) -> bool:
    """True iff a synth MiMo failure is a TRANSIENT hiccup worth a bounded internal retry.

    Transient (RETRY): a timeout (synth's own business-deadline ``wait_for``, or a transport-level
    ``APITimeoutError``), a connection reset / gateway-unreachable (``APIConnectionError``), a
    streamed completion cut before its finish_reason (``StreamTruncated``), or an
    HTTP 429 / 5xx from the gateway. Deterministic (FAIL FAST): a 4xx validation/auth error (400 /
    401 / 403 / 404 / 422), and anything else — notably the KeyPool's terminal ``RuntimeError("All
    configured LLM keys failed …")``, which already represents EXHAUSTED internal failover (its
    worst case is every deployment 403-blocked, a deterministic outage that a re-send won't fix).
    ``asyncio.CancelledError`` is a ``BaseException`` and is never reached here (never retried)."""
    # A timeout — synth's own wait_for deadline (asyncio.TimeoutError) or the transport-level
    # APITimeoutError (a subclass of APIConnectionError, handled just below).
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    # Connection reset / gateway unreachable (also catches APITimeoutError).
    if isinstance(exc, APIConnectionError):
        return True
    # A streamed completion cut before its finish_reason (issue #100): no HTTP status, not a
    # connection error type — the relay's idle cut / a dropped stream. Same class as a reset.
    if isinstance(exc, StreamTruncated):
        return True
    # HTTP-status-bearing errors: reuse the store layer's best-effort status extraction so the
    # transient/permanent classification convention lives in ONE place.
    return _is_transient_synth_status(_error_code(exc))


async def synth_answer(
    data: dict[str, Any],
    intent: str,
    timeout_s: float = _SYNTH_TIMEOUT_S,
) -> str:
    """Exit LLM → prose answer string (NO cited_papers here; aquery.py owns that).

    On LLM failure, returns a clean honesty fallback string rather than raising —
    the out-feed must still return a well-formed {answer, cited_papers, kb_coverage}.
    """
    prompt = _build_prompt(data, intent)
    # Model/thinking routing (issue #8): the research-plane synth call. Default routes to the
    # SYNTH slot with no thinking param (today's behavior); an operator can pin the strong model
    # + thinking ON via PAPERVAULT_LLM_SYNTH. Pass model only when non-empty (empty = ride the
    # pool default, byte-identical to today) and enable_thinking only when the route pins it.
    _model, _thinking = route("synth")
    _route_kw: dict[str, Any] = {}
    if _model:
        _route_kw["model"] = _model
    if _thinking is not None:
        _route_kw["enable_thinking"] = _thinking
    # Bounded retry (issue #70): retry ONLY transient hiccups (see _is_transient_synth_error);
    # fail fast on deterministic ones; fall back to _SYNTH_FAILED_MSG once retries are exhausted.
    attempts = _SYNTH_MAX_RETRIES + 1
    for attempt in range(attempts):
        # Timing log (2026-07-16, issue #3): synth wall-clock was an observability blind spot
        # — only the generic >=60s slow-call log in store/llm.py ever recorded it.
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                mimo_complete(
                    prompt,
                    system_prompt=_SYNTH_SYSTEM,
                    temperature=0.2,
                    max_tokens=_SYNTH_MAX_TOKENS,
                    **_route_kw,
                ),
                timeout=timeout_s,
            )
        except Exception as e:  # noqa: BLE001 — CancelledError is BaseException, never caught here
            dt = time.monotonic() - t0
            transient = _is_transient_synth_error(e)
            if transient and attempt < attempts - 1:
                backoff = _SYNTH_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "synth MiMo call failed after %.0fs (transient, attempt %d/%d) — retrying "
                    "in %.1fs: %s", dt, attempt + 1, attempts, backoff, e)
                await asyncio.sleep(backoff)
                continue
            # Deterministic (fail fast), OR transient with retries exhausted (true outage) → the
            # honest fallback. Retrieval still succeeded, so the caller gets cited_papers.
            reason = "transient, retries exhausted" if transient else "deterministic — fail fast"
            logger.warning(
                "synth MiMo call failed after %.0fs (%s, attempt %d/%d): %s",
                dt, reason, attempt + 1, attempts, e)
            return _SYNTH_FAILED_MSG

        answer = raw.strip()
        if not answer:
            # S16 (SDD §6.4): an empty/whitespace-200 is a SILENT failure — OK status, no usable
            # text (content filter, degenerate generation, gateway hiccup). This is exactly the
            # "malformed response" transient class from issue #70, so RETRY it like a transient
            # exception before the honesty fallback (rather than handing back a blank answer).
            if attempt < attempts - 1:
                backoff = _SYNTH_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "synth MiMo returned empty/whitespace 200 (S16, attempt %d/%d) — retrying "
                    "in %.1fs", attempt + 1, attempts, backoff)
                await asyncio.sleep(backoff)
                continue
            logger.warning(
                "synth MiMo returned empty/whitespace 200 (S16) after %d attempts — "
                "treating as failure", attempts)
            return _SYNTH_FAILED_MSG

        logger.info(
            "synth done in %.0fs (prompt_chars=%d, attempt %d/%d)",
            time.monotonic() - t0, len(prompt), attempt + 1, attempts)
        return answer

    # Unreachable: the final loop iteration always returns. Defensive fallback for total safety.
    return _SYNTH_FAILED_MSG
