"""PDF → markdown extraction (single-engine, whole-PDF MinerU2.5-Pro).

2026-06-06: the dots/chandra/marker per-chunk OCR cascade + per-paper GPU
pin + GPU-eviction machinery is replaced by ONE whole-PDF call to a
persistent MinerU2.5-Pro vLLM server (SDD §2.1).

**Model**: ``classify==EXTRACT → extract_md`` produces markdown from a SINGLE
whole-PDF MinerU call against a persistent vLLM server (no chunking, no page
cap, no per-paper exclusive GPU lock). Concurrency is an explicit semaphore
decoupled from GPU count; vLLM batches the per-page sub-requests server-side.

**Pipeline** (SDD §2.1): ``pdf_probe`` (isolated subprocess guard) → acquire a
concurrency slot + a round-robin endpoint → ``mineru_client.extract_mineru``
(one whole-PDF call) → doc-level ``review_extract`` clarity check → whole-doc
``completeness_gate`` → ``_save_md(engine="mineru2.5-pro")``.

**Failure model (C1, SDD §2.2)**: ``mineru_client`` raises two typed
exceptions. ``MineruTransportError`` (server unreachable / connection death /
bare-500) → NO md, leave status ``ok``/``pending``, do NOT charge
``extract_attempts``, do NOT set ``extract_failed`` (reconcile re-routes it
next sweep). ``MineruExtractionError`` (per-doc defect: 400/422/structured-500/
truncation/thin md) and a completeness-gate reject charge toward
``MAX_EXTRACT_ATTEMPTS`` → ``extract_failed`` only when the budget is
exhausted. There is NO instant-terminal-on-first-failure (the old
``n_chunks == 1`` disjunct is dropped — terminal is budget-only).

``review_extract`` is KEPT — its per-chunk OCR call site is gone, but it is
ALSO the firecrawl post-download pre-curator clarity judge (``download.py``);
here it runs ONCE on the whole-doc md.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from papervault.llm_routing import route

from .mineru_client import (
    MineruExtractionError,
    MineruTransportError,
    endpoints_from_env,
    extract_mineru,
)
from .services.mineru_server import get_server_controller
from .services import concurrency
from .models import (
    DOWNLOAD_STATUS_EXTRACT_FAILED,
    DOWNLOAD_STATUS_OK,
    MAX_EXTRACT_ATTEMPTS,
    Paper,
)
from .store import Library

# Below this many chars an extraction's text is treated as essentially blank —
# the engine produced nothing readable (a refused / blank rendering), a CLARITY
# failure, NOT a "this is short" verdict (D4 forbids the latter). Used by the
# surviving ``review_extract`` clarity judge (now a firecrawl-path consumer): a
# legitimately short-but-clean text (>= this floor) is ABOVE the floor and is
# judged by the LLM clarity reviewer, which is told "short is fine".
_BLANK_CHUNK_FLOOR = int(
    os.environ.get("PAPER_LIBRARY_BLANK_CHUNK_FLOOR_CHARS", "100"))

# Hard wall-clock cap for the isolated pdf_probe subprocess (SDD §2.1).
# A malformed / adversarial PDF can wedge pypdf for an unbounded time; the
# probe runs in a throwaway subprocess so the worst case is THIS one paper's
# probe being killed, never the daemon's main thread.
_PROBE_TIMEOUT_SECONDS = float(
    os.environ.get("PAPER_LIBRARY_PDF_PROBE_TIMEOUT_SECONDS", "30"))

# Daemon wall-clock cap (SDD §4.2 D10-replacement): a deadline threaded into the
# MinerU client as ``t_pinned_start + _CEILING_SECONDS``. Crossing it WHILE the
# server is alive and answering = per-doc pathology (a pathological PDF) →
# ``MineruExtractionError`` → charge → terminal at budget. Time spent in
# transport backoff because the server is dead is classified transport (don't
# charge). The clock starts only AFTER concurrency admission, so queue-wait
# never counts (D9 preserved).
_CEILING_SECONDS = float(
    os.environ.get("PAPER_LIBRARY_EXTRACT_CEILING_SECONDS", str(30 * 60)))

# Explicit whole-PDF extraction concurrency (SDD §4.1), decoupled from GPU
# count: a plain asyncio semaphore admits this many extractions in flight; vLLM
# batches the per-page sub-requests server-side via ``--max-num-seqs``.
_EXTRACT_CONCURRENCY = int(
    os.environ.get("PAPER_LIBRARY_EXTRACT_CONCURRENCY", "8"))

# Worker-pool size, DECOUPLED from the OCR/GPU concurrency cap above. The OCR slot
# (``concurrency.extract_slots``) caps how many whole-PDF MinerU calls hit the GPU
# at once (~8); we run MORE workers so that when several are in their (now off-loop)
# post-OCR LLM-gate phase, OTHER workers still hold the OCR slots and keep the 3090
# fed. This is the structural half of the GPU-idle fix. Must be >= _EXTRACT_CONCURRENCY.
_EXTRACT_WORKERS = int(
    os.environ.get("PAPER_LIBRARY_EXTRACT_WORKERS", "16"))

# Long-paper LOW-priority threshold in pages (re-homed from the old
# ``_DOTS_CHUNK_SIZE * 3``, SDD §6.4): reconcile / extract_queue de-prioritize a
# paper longer than this. The priority-by-pages behavior is unchanged — still
# keyed off ``pdf_probe(...).n_pages``.
_LONG_PAPER_PAGE_THRESHOLD = int(
    os.environ.get("PAPER_LIBRARY_LONG_PAPER_PAGE_THRESHOLD", "90"))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _pkg_version(name: str) -> str:
    if not name:
        return ""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return ""


# ----------------------------- pdf_probe -----------------------------------


@dataclass(frozen=True)
class PDFProbe:
    """Result of the isolated PDF health-check (SDD §2.1).

    ``bad`` is the single routing bit ``extract_md`` reads — True when the
    file is not a usable multi-page PDF (wrong magic / 0 pages / a parse
    error / the probe subprocess timed out or crashed). ``n_pages`` sizes the
    work (consumers de-prioritize a long paper off ``n_pages``). There is no
    chunking any more, so the old ``n_chunks`` / ``single_chunk`` fields are
    gone (whole-PDF call — terminal is budget-only, SDD §2.1).
    """

    bad: bool
    n_pages: int
    reason: str = ""


# A throwaway program run in a child interpreter. It MUST NOT import
# papervault.library (keeps the child cheap + isolated): it only checks the PDF
# magic bytes and counts pages via pypdf, then prints the page count. Any
# exception / hang is the parent's problem to time out — the child never
# blocks the daemon's main thread (one bad PDF can't wedge startup).
_PROBE_PROGRAM = (
    "import sys\n"
    "p = sys.argv[1]\n"
    "try:\n"
    "    with open(p, 'rb') as f:\n"
    "        head = f.read(5)\n"
    "    if not head.startswith(b'%PDF'):\n"
    "        print('-1'); sys.exit(0)\n"
    "    from pypdf import PdfReader\n"
    "    n = len(PdfReader(p).pages)\n"
    "    print(n)\n"
    "except Exception:\n"
    "    print('-1')\n"
)


def pdf_probe(pdf_path: str, *,
              timeout: float = _PROBE_TIMEOUT_SECONDS) -> PDFProbe:
    """Isolated PDF health-check: is-PDF? how many pages?

    SDD §2.1. Runs the parse in a **throwaway subprocess** with a hard
    wall-clock alarm so a malformed / adversarial PDF that wedges pypdf can
    only kill THIS probe, never the daemon's main thread. The decision logic
    (``extract_md``) lives in the parent; this returns a plain verdict.

    ``bad=True`` ⟺ not a usable PDF: bad magic bytes, 0 pages, a parse
    error in the child, or the child timed out / crashed (we judge those
    conservatively bad so the caller charges an attempt and stops hot-
    looping on a file that can't even be opened).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_PROGRAM, pdf_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return PDFProbe(bad=True, n_pages=0, reason="probe_timeout")
    except Exception as exc:  # spawn failure — treat as unprobeable
        return PDFProbe(bad=True, n_pages=0,
                        reason=f"probe_spawn_error:{type(exc).__name__}")

    if proc.returncode != 0:
        return PDFProbe(bad=True, n_pages=0, reason="probe_crash")

    out = (proc.stdout or "").strip().splitlines()
    try:
        n_pages = int(out[-1]) if out else -1
    except (ValueError, IndexError):
        n_pages = -1

    if n_pages < 0:
        return PDFProbe(bad=True, n_pages=0, reason="not_pdf")
    if n_pages == 0:
        return PDFProbe(bad=True, n_pages=0, reason="zero_pages")

    return PDFProbe(bad=False, n_pages=n_pages, reason="ok")


# --------------------------- LLM extract review ----------------------------
#
# review_extract is the clarity reviewer (D4). The per-chunk OCR caller is gone
# (single whole-PDF MinerU call, no chunks); the SURVIVING consumer is the
# firecrawl post-download pre-curator check (download.py), which asks "can a
# human read this rendering?" before treating a firecrawl md as real. It is
# deliberately NARROW (D4):
#
#   - It does NOT judge completeness. Whether the text is truncated, a paywall
#     stub, a publisher landing page, or missing an interior section is the
#     completeness_gate's job (D3).
#   - It does NOT judge by length (D4). A 2-page letter or a references-only
#     tail is short but perfectly readable; "short" is never a fail. The only
#     length rule is the <_BLANK_CHUNK_FLOOR-char floor below, which catches an
#     engine that produced essentially-nothing (a blank/refused rendering), not
#     "shortness". A legitimately-short-but-clean text (100-499 chars: a short
#     acknowledgements page, a brief reference list, a figure-only page) is NOT
#     condemned by the floor — it goes to the LLM clarity judge, which is told
#     "short is fine".
#
# So this prompt asks ONLY the clarity question and there is NO
# ``broken_pdf_suspected`` axis (that was completeness leakage, now fully owned
# by completeness_gate, D3/D4).
_REVIEW_PROMPT = (
    "You are checking ONE chunk of a PDF→Markdown extraction of an academic "
    "paper for READABILITY. Below is the text of a single chunk (not the whole "
    "paper). Return ONLY a JSON object on a single line, no prose:\n"
    '{"ok": true|false, "issues": ["..."], "confidence": 0.0-1.0}\n\n'
    # === THE ONE QUESTION (clarity only) ===
    "Your single test: **can a human read this chunk?** If the text is "
    "legible prose / equations / tables a reader can follow → ok=true. Flag "
    "ok=false ONLY when the extraction itself is broken so the chunk is "
    "UNREADABLE:\n"
    "  - Garbled / looping text: e.g., 'Hull, Hull, Hulf, Hull, Hulf, ...', "
    "'Galathe' for 'Galactic', or μ rendered as 'TO GUO GUILLOS'\n"
    "  - Per-character spacing corruption: 'a n d t h a t' instead of "
    "'and that'\n"
    "  - Wrong codepoint mapping: Braille glyphs U+2800-28FF where math "
    "symbols should be; random replacement chars (U+FFFD ⊙) throughout\n"
    "  - Engine-refusal text replacing real content: 'I cannot…', "
    "'As an AI model…'\n"
    "  - Essentially blank: almost no actual text came out of this chunk\n\n"
    # === DO NOT FLAG (clarity false-positive traps) ===
    "**Do NOT flag the following — these do NOT make a chunk unreadable:**\n"
    "  - The chunk being SHORT. Length is NOT a clarity problem. A short "
    "chunk (a 2-page letter, a references-only tail, an appendix) is fine.\n"
    "  - The chunk lacking an Introduction / Conclusion / some section. This "
    "is ONE chunk of a larger paper; it is NOT supposed to be self-contained. "
    "Missing sections are NEVER your concern (that is the whole-paper gate's "
    "job, not yours).\n"
    "  - The chunk ending mid-sentence or mid-paragraph. Chunks are cut by "
    "page count, so they routinely start and end mid-flow — that is normal "
    "chunking, NOT a truncated paper. Do NOT report 'truncated'.\n"
    "  - Whether the paper looks like a paywall page / landing page / is "
    "complete overall — NONE of that is your job. Judge ONLY readability.\n"
    "  - HTML residue (`<sup>`, `<span>`, `<br>`, `<table>`) from VLM "
    "output — cosmetic only, readability fine\n"
    "  - Equations rendered as LaTeX (`$...$`, `\\frac{}{}`) — that's the "
    "correct format\n"
    "  - Page-break markers (`<!-- page N -->`, `* * *`)\n"
    "  - Ligatures ('fi', 'fl' joined as single char) / Unicode superscripts "
    "/ Greek letters / arrows\n"
    "  - Author affiliation lists with superscript numbers / institutional "
    "long names\n\n"
    "Issues list must be concrete (quote the specific garbled chars or "
    "phrases). Empty if ok."
)


def review_extract(extract_text: str, *, llm=None) -> dict:
    """Clarity reviewer (D4) — now used by the firecrawl pre-curator check.

    The per-chunk OCR caller is gone (single whole-PDF MinerU call). The live
    consumer is the firecrawl post-download pre-curator check (``download.py``),
    which runs this on a web-scraped rendering before treating it as real full
    text. It answers ONE question — **can a human read this text?** (garble /
    loop / engine-refusal / blank). It is deliberately narrow:

    - It does NOT judge completeness (truncation / paywall stub / a missing
      interior section). That is ``completeness_gate``'s job (D3).
    - It NEVER judges by length (D4). The ONE length rule is the
      <_BLANK_CHUNK_FLOOR-char floor below, which catches a blank / refused
      rendering (essentially no text came out), not "this is short". A
      legitimately short-but-clean text (100-499 chars) is ABOVE the floor and
      is judged by the LLM clarity reviewer.

    Returns ``{"ok": bool, "issues": [str], "confidence": float}`` — there is
    no ``broken_pdf_suspected`` axis (that completeness/PDF-source judgment now
    lives entirely in ``completeness_gate``). Network or parse failure →
    ``ok=True`` (fail-open) so a flaky API never discards a readable text.
    """
    if not extract_text or len(extract_text) < _BLANK_CHUNK_FLOOR:
        # Blank / near-blank chunk: the engine produced essentially nothing,
        # which is a CLARITY failure (nothing readable came out), not a
        # "shortness" judgment. The floor matches the engine-layer thin-output
        # guard, so a legitimately short-but-clean tail (100-499 chars) is NOT
        # condemned here — it goes to the LLM clarity judge below (D4: short is not an error).
        return {
            "ok": False,
            "issues": ["extract_too_short"],
            "confidence": 1.0,
        }

    try:
        if llm is None:
            from .llm import get_llm

            # gate role (issue #8): the extract clarity/completeness gates. Default = the SYNTH
            # slot (today's get_llm() default); operator-overridable via PAPERVAULT_LLM_GATE.
            llm = get_llm(model=route("gate")[0])
        raw = llm.call([
            {"role": "system", "content": _REVIEW_PROMPT},
            {"role": "user", "content": extract_text},
        ])
    except Exception:
        return {
            "ok": True,
            "issues": ["review_llm_unavailable"],
            "confidence": 0.0,
        }

    import json
    import re
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {
            "ok": True,
            "issues": ["review_parse_failed"],
            "confidence": 0.0,
        }
    try:
        verdict = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {
            "ok": True,
            "issues": ["review_parse_failed"],
            "confidence": 0.0,
        }
    # Coerce ``ok``: LLMs sometimes emit the JSON string "false" instead of a
    # boolean, and ``bool("false")`` is True — which would pass a genuinely
    # unreadable chunk. Read truthiness words explicitly (matches the same
    # guard in completeness_gate).
    ov = verdict.get("ok", True)
    if isinstance(ov, bool):
        ok = ov
    else:
        ok = str(ov).strip().lower() not in ("false", "0", "no", "")
    return {
        "ok": ok,
        "issues": verdict.get("issues", []) or [],
        "confidence": verdict.get("confidence", 0.0),
    }


# ----------------------- LLM completeness gate (D3) ------------------------
#
# The ONE whole-document completeness gate (D3 / SDD §5/§6.1/§6.3). It runs
# AFTER all per-chunk ``review_extract`` passes, on the FULL assembled text,
# exactly once per paper. Two distinct jobs, never conflated:
#
#   review_extract (D4)  — per CHUNK, "can a human READ this block?" (local
#                          clarity: garble / loop / strike / blank). Length-blind
#                          except a <500-char floor.
#   completeness_gate(D3) — per WHOLE PAPER, "is this a COMPLETE article?" (global
#                          completeness: truncation / login-or-paywall stub /
#                          body cut mid-sentence). NEVER judges by length — a
#                          2-page letter or a pure-references tail block is
#                          legitimately complete.
#
# It is JUDGE-ONLY (D3 lossless): it inspects the text and returns a verdict;
# it NEVER edits a single character (downstream consumers need the verbatim
# original). The caller (extract_md / firecrawl fallback) acts on the verdict —
# pass → serve as full text; fail → terminal (no md saved / md removed), the
# abstract still serves from metadata (D6).
_GATE_PROMPT = (
    "You are the FINAL completeness gate for an academic paper's full text "
    "(already proven readable). Below is the ENTIRE assembled text. Decide ONE "
    "thing: **is this a COMPLETE article, or is it cut off / a non-article "
    "stub?** Return ONLY a JSON object on a single line, no prose:\n"
    '{"complete": true|false, "reason": "..."}\n\n'
    # === THE ONLY THREE WAYS TO BE INCOMPLETE ===
    "Set ``complete: false`` ONLY for one of these EVIDENCE-BASED conditions:\n"
    "  1. TRUNCATED — the text stops mid-stream: the body ends mid-word "
    "(e.g. 'we therefore conclu' with nothing after) or mid-equation "
    "('\\frac{a}{' with no close), or a chunk/page is obviously missing in "
    "the middle (a section header followed immediately by the next paper's "
    "title, an abrupt jump that drops whole sections).\n"
    "  2. LOGIN / PAYWALL STUB — the 'text' is not the paper but a gate page: "
    "'Sign in to access', 'Subscribe to read', 'Access denied', 'Purchase "
    "PDF', 'You do not have access', a cookie/consent wall, a CAPTCHA, or only "
    "title + authors + DOI + 'Abstract' with NO body (Introduction / Methods / "
    "Results / Discussion) for a paper that must be multi-page.\n"
    "  3. BODY CUT MID-SENTENCE — the prose genuinely breaks off in the middle "
    "of a sentence at the very end with no terminal punctuation AND no natural "
    "stopping point (not a heading, not a reference entry, not a figure "
    "caption).\n\n"
    # === NEVER JUDGE BY LENGTH (the core D3 rule) ===
    "**CRITICAL: do NOT call a SHORT text incomplete.** Length is NOT evidence "
    "of incompleteness. ALL of these are COMPLETE (complete=true):\n"
    "  - A 2-page letter / research note / proceedings extended abstract that "
    "is short but whole.\n"
    "  - A paper that ends on its references section (long or short).\n"
    "  - A paper that ends right after Conclusions / Acknowledgements with no "
    "references at all.\n"
    "  - A chunk that is ONLY a references list or ONLY an appendix — that is a "
    "legitimate tail of a longer whole, not a truncation.\n"
    "  - Prose that ends a paragraph mid-thought but at a real sentence "
    "boundary (academic writing is not always tidy) → COMPLETE.\n\n"
    # === ANTI-HALLUCINATION ===
    "**Do NOT claim 'truncated' unless you can prove it from the LAST "
    "characters of the text.** Before flagging truncation: does the text end at "
    "sentence punctuation ('.', '!', '?', '\"', ')'), a heading, or a reference "
    "entry? → it is COMPLETE. Only if it ends mid-word / mid-equation with "
    "nothing after is it truncated. If you cannot point to the exact broken-off "
    "ending, default to complete=true.\n\n"
    "When complete=true, ``reason`` is a short phrase like 'complete' or "
    "'short but whole letter'. When complete=false, ``reason`` must name which "
    "of the three conditions and quote the evidence (e.g. 'truncated: ends "
    "\\'we therefore conclu\\'' or 'paywall: \\'Sign in to access\\'')."
)


def completeness_gate(full_text: str, *, llm=None) -> dict:
    """The one whole-document completeness gate (D3 / SDD §5/§6.1/§6.3).

    Runs once on the FULL assembled text, after all per-chunk reviews pass.
    Judge-only and lossless — inspects, never edits a character.

    Question: *is this a COMPLETE article?* Judges three things only —
    truncation, a login/paywall stub, or body cut mid-sentence — and
    **never judges by length** (a 2-page letter or a pure-references tail
    is legitimately complete; see ``_GATE_PROMPT``).

    Returns ``{"complete": bool, "reason": str}``.

    Failure policy (SDD §5): a *real* LLM / parse failure → ``complete=True``
    (fail-open). The text reaching this gate has already passed every
    per-chunk ``review_extract``; we do not let a flaky API or an
    unparseable reply throw away genuinely-good work. ``reason`` records the
    fail-open cause (``gate_llm_unavailable`` / ``gate_parse_failed``) for
    audit. (Empty text is the one hard-coded incomplete: there is nothing
    to serve.)
    """
    if not full_text or not full_text.strip():
        return {"complete": False, "reason": "empty_text"}

    try:
        if llm is None:
            from .llm import get_llm

            # gate role (issue #8): the extract clarity/completeness gates. Default = the SYNTH
            # slot (today's get_llm() default); operator-overridable via PAPERVAULT_LLM_GATE.
            llm = get_llm(model=route("gate")[0])
        raw = llm.call([
            {"role": "system", "content": _GATE_PROMPT},
            {"role": "user", "content": full_text},
        ])
    except Exception:
        # Fail-open: don't lose review-passed text over a flaky API.
        return {"complete": True, "reason": "gate_llm_unavailable"}

    import json
    import re
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {"complete": True, "reason": "gate_parse_failed"}
    try:
        verdict = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"complete": True, "reason": "gate_parse_failed"}

    # Coerce ``complete``. LLMs commonly emit the JSON string "false" instead
    # of a boolean; ``bool("false")`` is True, which would fail OPEN and serve a
    # genuinely-incomplete paper as complete — the exact direction this gate
    # exists to catch (distinct from the intentional fail-open on API/parse
    # errors above). So treat a string explicitly via its truthiness words.
    cv = verdict.get("complete", True)
    if isinstance(cv, bool):
        complete = cv
    else:
        complete = str(cv).strip().lower() not in ("false", "0", "no", "")
    reason = verdict.get("reason") or ("complete" if complete else "incomplete")
    return {"complete": complete, "reason": str(reason)}


def confirmed_completeness_gate(full_text: str, *, llm=None) -> dict:
    """D3 hardening (2026-06-10): double-judge a gate REJECT before acting on it.

    The gate is an LLM judge and measurably flaky: the 2026-06-07 whole-vault
    backfill produced 30 rejects of which 24 (80%) were re-verified as FALSE
    POSITIVES (adversarial re-read of the actual mds — ADR
    2026-06-08-mineru-ondemand-lifecycle's sibling investigation). A single
    "incomplete" verdict is therefore weak evidence against an md that already
    survived extraction. This wrapper re-judges a reject ONCE and confirms it
    only if BOTH passes independently agree; a split verdict → complete, which
    matches the gate's own fail-open bias (never discard good work on one flaky
    judgment). A PASS is accepted immediately — the measured failure mode is
    false-reject, not false-accept. The hard-coded empty-text reject re-confirms
    deterministically at zero cost (no LLM call on that branch).

    Returns the same ``{"complete": bool, "reason": str}`` shape; a confirmed
    reject carries both reasons, an overturned one records the overruled first
    reason for audit.
    """
    first = completeness_gate(full_text, llm=llm)
    if first.get("complete", True):
        return first
    second = completeness_gate(full_text, llm=llm)
    if second.get("complete", True):
        # A confirmation pass that can be satisfied by its OWN failure is not a
        # confirmation (review must-fix): the gate fail-opens (complete=true)
        # on an LLM/parse error, and letting that overturn a genuine pass-1
        # reject would serve a real stub forever. A fail-open second pass is
        # "no second opinion obtained" → keep the pass-1 reject.
        if second.get("reason") in ("gate_llm_unavailable", "gate_parse_failed"):
            return {"complete": False,
                    "reason": f"reject_stands_no_second_opinion "
                              f"({second.get('reason', '')}): "
                              f"{first.get('reason', '')}"}
        return {"complete": True,
                "reason": f"reject_overturned_on_rejudge "
                          f"(first: {first.get('reason', '')})"}
    return {"complete": False,
            "reason": f"confirmed_x2: {first.get('reason', '')} "
                      f"| {second.get('reason', '')}"}


# ------------------------- save helper -------------------------------------


def _save_md(paper: Paper, library: Library, engine: str, text: str) -> None:
    """Write the md body RAW (NO YAML frontmatter — OCR md has never carried
    any; serve / reconcile-abstract-feed / consistency all read the file head
    raw and would mis-parse a YAML block) and stamp provenance in
    ``md_engine`` / ``md_engine_version`` (index.json).

    ``md_engine_version`` for the current engine ``mineru2.5-pro`` is the
    HARDCODED model id (SDD §3.3 item 4c): do NOT route mineru through
    ``_pkg_version`` (``importlib.metadata.version`` returns ``""`` if mineru
    is absent from the daemon import path, and the model id is the meaningful
    provenance anyway — consumed by ``extract_quality_tier``, audits, the
    backfill resume marker). The dots/chandra/marker pkg-map is kept
    read-frozen for back-compat (no new rows are produced by them).
    """
    md_path = library.md_path(paper.key)
    _atomic_write(md_path, text)
    paper.md_path = str(md_path.relative_to(library.root))
    paper.md_engine = engine
    if engine == "mineru2.5-pro":
        paper.md_engine_version = "MinerU2.5-Pro-2605-1.2B"
        return
    pkg = {"chandra": "chandra-ocr", "dots": "dots_mocr",
           "marker": "marker-pdf"}.get(engine, "")
    paper.md_engine_version = _pkg_version(pkg)


# ------------------------- main entry point --------------------------------


async def extract_md(paper: Paper, library: Library, *,
                     force: bool = False,
                     llm=None,
                     on_progress=None) -> Optional[str]:
    """Single-engine whole-PDF extraction via the persistent MinerU server.

    Pipeline (SDD §2.1):

      1. ``pdf_probe`` (isolated subprocess) — a malformed / non-PDF file is a
         genuine per-doc defect: charge an attempt, terminal at budget.
      2. Acquire a round-robin endpoint (the concurrency-slot admission lands
         here in slice 2 — see the SLICE-2 note below) and start the daemon
         wall-clock clock.
      3. ONE whole-PDF ``mineru_client.extract_mineru`` call (no chunking, no
         page cap), bounded by a per-request HTTP timeout + the daemon
         wall-clock cap. It raises ``MineruTransportError`` |
         ``MineruExtractionError`` (the C1 split, SDD §2.2).
      4. Doc-level ``review_extract`` clarity sanity (belt-and-suspenders for a
         200-OK-but-thin body; ``extract_mineru`` already raises on thin/empty).
      5. Whole-doc ``completeness_gate`` (the write-time chokepoint).
      6. ``_save_md(engine="mineru2.5-pro")`` + ``download_status = ok``.

    C1 terminalization rule (SDD §2.2/§2.4):
      - ``MineruTransportError`` → NO md, leave status ``ok``/``pending``, do
        NOT charge ``extract_attempts``, do NOT set ``extract_failed``.
        Non-mutation lets classify re-route the paper next reconcile sweep.
      - ``MineruExtractionError`` (per-doc defect) and a completeness-gate
        reject charge toward ``MAX_EXTRACT_ATTEMPTS``; terminal at budget
        ONLY (there is no instant-terminal-on-first-failure — the old
        ``n_chunks == 1`` disjunct is dropped).

    Idempotent: an md already on disk is terminal truth and is never re-OCR'd.
    """
    md_path = library.md_path(paper.key)

    # An md already on disk is terminal truth. We do NOT re-OCR a firecrawl-md
    # paper just because a real PDF later appears: per D5 ("if firecrawl's grab
    # is no good, it stays no good") a firecrawl text-only paper that passed the
    # completeness gate is served as-is for good — there is no PDF-upgrade
    # re-hunt. classify() routes such a paper to rule-2 TERMINAL (has_pdf ∧
    # has_md) and serve-safety keeps surfacing the firecrawl md via text_path;
    # nothing forces a re-OCR.
    if not force and library.has_extract(paper.key, "md"):
        return md_path.read_text()
    if not library.has_pdf(paper.key):
        return None

    pdf_path = library.pdf_path(paper.key)

    def _p(msg: str) -> None:
        """Best-effort progress callback emit."""
        if on_progress is None:
            return
        try:
            on_progress(msg)
        except Exception:
            pass

    # ---- 1. Isolated PDF health-check (SDD §2.1) — BEFORE any server call ----
    # A malformed PDF must be caught in a throwaway subprocess (never on the
    # daemon's main thread). A bad local file is a genuine per-doc defect →
    # charge an attempt; terminal at BUDGET only (the old `∨ single_chunk`
    # disjunct is dropped — there is no chunking).
    probe = pdf_probe(str(pdf_path))
    if probe.bad:
        paper.extract_attempts += 1
        if paper.extract_attempts >= MAX_EXTRACT_ATTEMPTS:
            paper.download_status = DOWNLOAD_STATUS_EXTRACT_FAILED
        library.log({"event": "extract_md_probe_bad", "key": paper.key,
                     "reason": probe.reason,
                     "attempts": paper.extract_attempts,
                     "status": paper.download_status})
        _p(f"extract: pdf_probe BAD ({probe.reason}) — "
           f"attempt {paper.extract_attempts}/{MAX_EXTRACT_ATTEMPTS}")
        return None

    # ---- 2. Endpoint set + wall-clock clock (SDD §2.1 step 2/§4.1) ----------
    # Concurrency admission (the asyncio.Semaphore permit) lives ONE layer up in
    # ``ExtractQueue._process_one_inner`` (SDD §4.1): the worker pool is sized to
    # ``_EXTRACT_CONCURRENCY`` and the semaphore gates dispatch, so a permit is
    # already held before ``extract_md`` is awaited. Here we only resolve the
    # endpoint set from the env (``MINERU_URL``, comma-sep) — ``extract_mineru``
    # round-robins / fails over across it internally. The wall-clock clock starts
    # at the call boundary (after admission), so queue-wait is never metered (D9
    # invariant preserved).
    endpoints = endpoints_from_env()
    t_pinned_start = time.time()
    wall_clock_deadline = t_pinned_start + _CEILING_SECONDS

    pdf_bytes = pdf_path.read_bytes()

    # ---- 3. ONE whole-PDF MinerU call (no chunking, no page cap) ----
    library.log({"event": "extract_md_start", "key": paper.key,
                 "n_pages": probe.n_pages,
                 "endpoints": [e.url for e in endpoints]})
    _p(f"extract: MinerU whole-PDF call ({probe.n_pages} pages)")
    # OCR-slot admission (SDD §4.1): hold the GPU/OCR permit ONLY around the MinerU
    # call and release it in the ``finally`` below — BEFORE the LLM gates — so a
    # worker in its off-loop gate phase never pins an OCR slot. With _EXTRACT_WORKERS
    # > _EXTRACT_CONCURRENCY, a freed slot is immediately re-dispatched → GPU stays fed.
    await concurrency.acquire_extract_slot()
    try:
        # On-demand server gate (no-op unless PAPER_LIBRARY_MINERU_ONDEMAND=1):
        # start the MinerU server if it was stopped on idle and WAIT for it to be
        # model-ready. INSIDE the try so a readiness timeout (MineruServerUnavailable,
        # a MineruTransportError subclass) lands in the C1 transport arm below —
        # NO attempt charged, reconcile retries. The wall-clock deadline above was
        # set at the call boundary; the cold-start wait is transport, not metered
        # against the per-doc budget.
        await get_server_controller().ensure_ready()
        mineru_md = await extract_mineru(
            pdf_bytes, endpoints,
            stem=paper.key,
            wall_clock_deadline=wall_clock_deadline,
        )
    except MineruTransportError as exc:
        # ── C1 CORE: transport → DO NOT charge, DO NOT terminalize. ──
        # status & attempts deliberately UNMUTATED → classify re-routes the
        # paper to EXTRACT on the next reconcile sweep. Non-mutation IS the
        # mechanism (SDD §2.4). A bare 500 / a `systemctl restart` window /
        # both endpoints down all land here and never condemn the paper.
        library.log({"event": "extract_md_transport_retry", "key": paper.key,
                     "error": repr(exc)[:200],
                     "attempts": paper.extract_attempts})
        _p(f"extract: TRANSPORT failure ({repr(exc)[:80]}) — "
           f"no charge, will retry next sweep")
        return None
    except MineruExtractionError as exc:
        # ── genuine per-doc failure → charge, terminal at BUDGET ONLY ──
        # 400/422 / structured-500 / truncation / thin md / the wall-clock cap
        # firing while the server is alive all land here.
        paper.extract_attempts += 1
        if paper.extract_attempts >= MAX_EXTRACT_ATTEMPTS:
            paper.download_status = DOWNLOAD_STATUS_EXTRACT_FAILED
        library.log({"event": "extract_md_engine_failed", "key": paper.key,
                     "error": repr(exc)[:200],
                     "attempts": paper.extract_attempts,
                     "status": paper.download_status})
        _p(f"extract: EXTRACTION failure ({repr(exc)[:80]}) — "
           f"attempt {paper.extract_attempts}/{MAX_EXTRACT_ATTEMPTS}")
        return None
    finally:
        # Release the OCR/GPU slot the instant MinerU is done (success OR raise),
        # BEFORE the LLM gates run — the gates are off-loop and unslotted, so they
        # never starve OCR. (Early returns above this try never acquired the slot.)
        concurrency.release_extract_slot()

    # ---- 4. Doc-level clarity sanity (belt-and-suspenders) ----
    # `extract_mineru` already raises MineruExtractionError on thin/empty md;
    # this catches a 200-OK-but-empty body that slipped through. Treat as a
    # per-doc extraction failure (charge, terminal at budget).
    if not mineru_md or not mineru_md.strip():
        paper.extract_attempts += 1
        if paper.extract_attempts >= MAX_EXTRACT_ATTEMPTS:
            paper.download_status = DOWNLOAD_STATUS_EXTRACT_FAILED
        library.log({"event": "extract_md_empty", "key": paper.key,
                     "attempts": paper.extract_attempts,
                     "status": paper.download_status})
        _p("extract: empty md — "
           f"attempt {paper.extract_attempts}/{MAX_EXTRACT_ATTEMPTS}")
        return None

    # The whole-doc clarity judge (the firecrawl-path consumer's twin, run here
    # once on the assembled md). Fail-open on LLM/parse error (never discards
    # readable text). A genuine clarity FAIL is treated like the gate reject
    # below: a bad rendering of a real PDF is a source/extraction problem →
    # terminal WITHOUT charging an attempt (a retry hits the same wall; the
    # judge already proved the bytes render to garble).
    async with concurrency.gate_sem:
        clarity = await asyncio.to_thread(review_extract, mineru_md, llm=llm)
    if not clarity.get("ok", True):
        paper.download_status = DOWNLOAD_STATUS_EXTRACT_FAILED
        library.log({"event": "extract_md_clarity_reject", "key": paper.key,
                     "issues": clarity.get("issues", [])[:5],
                     "chars": len(mineru_md)})
        _p(f"extract: clarity REJECT ({clarity.get('issues', [])[:3]}) — "
           f"extract_failed")
        return None

    # ---- 5. Completeness gate (SDD §2.1 step 5) — the WRITE-TIME chokepoint --
    # The ONLY thing that catches a paywall stub / mid-sentence truncation / a
    # missing interior block. md MUST NOT land on disk until it passes, because
    # serve-safety (mcp.server) treats "md on disk" as "real + complete" with no
    # re-check. Fail-open (LLM/parse error) → complete=true, so a flaky API
    # never discards good work. A reject must be CONFIRMED by a second judging
    # pass (the gate false-rejects ~0.8% per single pass — D3 hardening, see
    # confirmed_completeness_gate). A confirmed reject → terminal, NO md on
    # disk, attempts NOT bumped (D5: the source is bad, no retry helps).
    async with concurrency.gate_sem:
        verdict = await asyncio.to_thread(confirmed_completeness_gate, mineru_md, llm=llm)
    if not verdict.get("complete", True):
        paper.download_status = DOWNLOAD_STATUS_EXTRACT_FAILED
        library.log({"event": "extract_md_gate_reject", "key": paper.key,
                     "reason": verdict.get("reason", ""),
                     "chars": len(mineru_md)})
        _p(f"extract: completeness_gate REJECT "
           f"({verdict.get('reason', '')}) — extract_failed")
        return None

    # ---- 6. Save (label → mineru2.5-pro) ----
    _save_md(paper, library, "mineru2.5-pro", mineru_md)
    paper.download_status = DOWNLOAD_STATUS_OK
    library.log({"event": "extract_md_done", "key": paper.key,
                 "engine": "mineru2.5-pro",
                 "n_pages": probe.n_pages,
                 "chars": len(mineru_md),
                 "seconds": round(time.time() - t_pinned_start, 1)})
    _p(f"extract: done (mineru2.5-pro, {probe.n_pages} pages, "
       f"{len(mineru_md)} chars)")
    return mineru_md
