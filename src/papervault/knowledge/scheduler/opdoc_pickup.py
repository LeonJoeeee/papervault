"""Windowless in-service operator-doc pickup (#81).

The SCHEDULER-side of operator-doc ingest. The standalone `papervault ingest-doc` CLI runs
as a SEPARATE process writing the SAME ledger + LightRAG graph the live service serves, so
it needs a maintenance window (`require_services_stopped`, cli.py) to avoid two-writer
corruption. That takes PL offline for the whole bulk (~15h for 45 textbooks). This module
makes operator-doc ingest WINDOWLESS by routing it through the SAME single writer papers use:

  An operator DROPS a `textbook-<Key>.md` / `notebook-<Key>.md` file into the pending dir
  (env `KS_OPDOC_PENDING_DIR`, default `/data/paper-vault/operator-drop/pending`). The live
  KS scheduler's serial `main_loop` calls `drain_pending(rag)` after each paper `run_round`,
  which scans the dir and `await ingest_document(...)`s each file against the process-wide
  singleton LightRAG — the very same instance (and single writer) that builds papers.

THE GENERAL "any input → document → RAG" QUEUE. This is one queue with one single writer;
the input type just picks a preprocessing front-end that all converge on the SAME
`ingest_document` call:
  - `textbook-<Key>.md` / `.markdown`  → ingested directly (heading-aware chunking).
  - `textbook-<Key>.pdf`               → OCR'd to markdown via the SHARED MinerU pipeline
                                         (`mineru_client.extract_mineru`, the very engine the
                                         paper-library extract path uses), then the produced
                                         md is fed through `ingest_document` exactly like a
                                         native `.md` drop. Same queue, same single writer —
                                         OCR is just a preprocessing step for PDF inputs.
  - `.url` (FUTURE, NOT built)          → a fetch front-end would download → md → ingest.
                                         The dispatch below is shaped so adding it is a new
                                         suffix branch that also converges on `ingest_document`.

PDF OCR error routing mirrors the paper extract path's transport-vs-extraction split (C1):
  - `MineruTransportError` (MinerU down/unreachable — TRANSIENT) → LEAVE the pdf in pending
    and retry next round when MinerU is back; log ONCE (the SourceDisabled pattern). Never
    move to failed/ — a server outage must not condemn a good PDF.
  - `MineruExtractionError` (per-doc OCR defect — TERMINAL) → move to failed/, log LOUD.

INVARIANTS INHERITED FOR FREE (why this needs no new queue / lock / endpoint):
  - SINGLE WRITER: `main_loop` is ONE serial `asyncio.Task`; `drain_pending` runs AFTER
    `run_round` in that same task, so it never overlaps a paper round → no two-writer race.
  - OFF-LOOP: `ingest_document` already offloads its GPU/DB work to threads (the #31/#75 fix)
    and, since #81, its synchronous `build_sections` tiktoken pass to `asyncio.to_thread`, so
    picking up a 1000-page book never blocks the event loop that serves live queries.

Filename → provenance key: the kind is the prefix before the first hyphen, the key body is
the rest (notebook keys legitimately contain hyphens, e.g. `notebook-idea23-c12`):
  `textbook-Schlickeiser2002.md`  → kind=textbook, key=`textbook:Schlickeiser2002`
  `notebook-idea23-c12.md`        → kind=notebook, key=`notebook:idea23-c12`
A sidecar `<file>.force` present ⇒ `force=True` (purge + re-ingest, to REPLACE an older
version); absent ⇒ push-once (an already-ingested key is refused and treated as done).

OUTCOME ROUTING (a bad file must NEVER crash the scheduler loop). `ingest_document` RETURNS
a `{done, error, sections, ...}` split (it can succeed, partly succeed, or build NOTHING all
WITHOUT raising), so a non-exception return is inspected — not assumed to be success (#84):
  full success (done>0, error==0)  → move file (+ any `.force` sidecar) to `processed/`
  PARTIAL (done>0, error>0)        → move to `processed/` + log LOUD (some sections landed;
                                     a `.force` re-drop rebuilds the errored ones)
  NOTHING built (done==0)          → move to `failed/` + write a `<file>.reason` note + log
                                     LOUD, count `failed`. #84: `ingest_document` can RETURN
                                     (no exception) with done=0 when EVERY section failed to
                                     build — e.g. a transient embedding/LLM outage errored the
                                     whole book (textbook:Bubeck2015 = done=0/error=51 during
                                     the #84 embedding break). That is a REAL FAILURE, never a
                                     silent success — it stays visibly failed + re-droppable
                                     instead of being marked done and lost.
  AlreadyIngestedError             → move to `processed/` (push-once: it's already done)
  SourceDisabledError              → LEAVE in place, log ONCE (source flag OFF; a re-enable picks it up)
  any other error                  → move to `failed/` + log LOUD (malformed key, enqueue blip, …)

RESTART RECOVERY (issue #81 known gap): an in-flight ingest interrupted by a restart leaves a
`processing` ledger row the paper `reconcile_terminal` (scans only `ingest_source="paper"`)
won't clean. MVP recovery = re-drop the file with a `.force` sidecar: force purges the stuck
rows then re-ingests clean. A boot-time operator-doc reconcile pass is durable follow-up.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from papervault.knowledge.ingest.operator_docs import (
    KINDS,
    AlreadyIngestedError,
    SourceDisabledError,
    check_source_enabled,
    ingest_document,
)
from papervault.library.mineru_client import (
    MineruExtractionError,
    MineruTransportError,
    endpoints_from_env,
    extract_mineru,
)

log = logging.getLogger("ks.scheduler.opdoc_pickup")

DEFAULT_PENDING_DIR = "/data/paper-vault/operator-drop/pending"
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_PDF_SUFFIX = ".pdf"
# Every suffix the pickup recognizes as an operator-doc drop: markdown (ingested directly) or
# PDF (OCR'd to markdown first, then ingested through the SAME path). A future `.url` front-end
# would add its suffix here + a branch in `drain_pending` that also lands on `ingest_document`.
_DROP_SUFFIXES = _MARKDOWN_SUFFIXES | {_PDF_SUFFIX}
_FORCE_SIDECAR_SUFFIX = ".force"
# Staging for OCR'd markdown: a dot-prefixed SUBDIR under the pending dir. The drain lists
# FILES only (`is_file()`), so a subdir is never scanned — the produced `textbook-<Key>.md`
# can therefore carry a real `.md` suffix (→ heading-aware chunking) without ever being
# mistaken for its own operator-doc drop on a later round.
_OCR_STAGING_SUBDIR = ".ocr"

# Files whose source class is currently DISABLED that we've already logged, so a disabled
# source doesn't spam the log once per round (the file legitimately stays in pending until the
# flag is turned on). Cleared for a file once it successfully ingests. Process-local by design.
_disabled_logged: set[str] = set()

# PDFs left in pending because MinerU is DOWN/unreachable that we've already logged, so a
# server outage doesn't spam the log once per round (the pdf legitimately stays in pending
# until MinerU is back). A SEPARATE set from `_disabled_logged` so a source-flag warning and
# a MinerU-down warning never suppress each other for the same path. Cleared once the pdf
# OCRs (MinerU answered) or is otherwise moved out of pending. Process-local by design.
_mineru_down_logged: set[str] = set()


def _pending_dir() -> Path:
    return Path(os.getenv("KS_OPDOC_PENDING_DIR", DEFAULT_PENDING_DIR))


def parse_drop_name(name: str) -> tuple[str, str] | None:
    """`textbook-<Key>.md` / `textbook-<Key>.pdf` → (kind, `<kind>:<Key>`); None otherwise.

    The kind is the prefix before the FIRST hyphen; the key body is everything after (notebook
    keys legitimately contain hyphens, e.g. `notebook-idea23-c12` → `idea23-c12`). Recognizes
    both markdown (`.md`/`.markdown`) and PDF (`.pdf`) drops — key derivation is IDENTICAL for
    both, only `drain_pending` branches on the suffix (a `.pdf` is OCR'd to md first). Returns
    None for any other suffix or any name without a recognized `<kind>-` prefix (so `.force`
    sidecars, `paper-*.md`, `README.md`, plain `.txt`, etc. are silently skipped).
    """
    p = Path(name)
    if p.suffix.lower() not in _DROP_SUFFIXES:
        return None
    stem = p.stem
    for kind in KINDS:
        prefix = f"{kind}-"
        if stem.startswith(prefix):
            body = stem[len(prefix):]
            if body:
                return kind, f"{kind}:{body}"
    return None


def _move(path: Path, dest_dir: Path) -> None:
    """Move an operator-doc file (+ any `.force` sidecar) into dest_dir. Best-effort: a move
    failure is logged LOUD but NEVER raised — the scheduler loop must survive it (the file is
    simply re-scanned next round, hits AlreadyIngestedError, and is re-moved)."""
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        sidecar = path.with_name(path.name + _FORCE_SIDECAR_SUFFIX)
        shutil.move(str(path), str(dest_dir / path.name))
        if sidecar.exists():
            shutil.move(str(sidecar), str(dest_dir / sidecar.name))
    except Exception as e:  # noqa: BLE001 — a move failure must not abort the drain
        log.exception("opdoc pickup: failed to move %s to %s/ (%r)", path.name, dest_dir.name, e)


async def _ocr_to_staged_md(pdf_path: Path) -> Path:
    """OCR a dropped PDF to markdown via the SHARED MinerU pipeline and stage the md.

    The PDF front-end of the general input→document→RAG queue. Reads the pdf bytes OFF-LOOP
    (`to_thread` — a book PDF is large), awaits the async whole-PDF `extract_mineru` (the very
    engine paper-library's extract path uses; it round-robins / fails over across the
    `MINERU_URL` endpoints and never blocks the loop), then writes the produced markdown
    OFF-LOOP to `<pending>/.ocr/<pdf-stem>.md` and returns that path. The `.md` suffix routes
    `ingest_document` to heading-aware chunking; the `.ocr` staging subdir is never scanned by
    the drain. `stem=pdf_path.stem` (e.g. `textbook-Foo2020`) is a real filesystem name, so it
    is inherently fs-safe for MinerU's `<out>/<stem>/vlm/<stem>.md` read-back path.

    Raises `MineruTransportError` (server down/unreachable — TRANSIENT, caller leaves the pdf
    in pending) or `MineruExtractionError` (per-doc OCR defect — TERMINAL, caller → failed/).
    """
    pdf_bytes = await asyncio.to_thread(pdf_path.read_bytes)
    md_text = await extract_mineru(pdf_bytes, endpoints_from_env(), stem=pdf_path.stem)

    def _write() -> Path:
        staging = pdf_path.parent / _OCR_STAGING_SUBDIR
        staging.mkdir(parents=True, exist_ok=True)
        md_path = staging / f"{pdf_path.stem}.md"
        md_path.write_text(md_text, encoding="utf-8")
        return md_path

    return await asyncio.to_thread(_write)


def _discard_staged(staged_md: Path | None, dest_dir: Path) -> None:
    """Move a staged OCR md (if any) into dest_dir alongside its PDF, best-effort.

    Keeping the OCR output next to the processed/failed PDF preserves EXACTLY what was ingested
    (or what failed to ingest) without re-OCRing — a cheap debugging artifact. A no-op for a
    native `.md` drop (`staged_md is None`). Never raises: a stray staged md lives in the
    unscanned `.ocr` subdir, so a failed move here can never leak back into the drain."""
    if staged_md is None:
        return
    try:
        if staged_md.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged_md), str(dest_dir / staged_md.name))
    except Exception as e:  # noqa: BLE001 — a staged-md move failure must not abort the drain
        log.warning(
            "opdoc pickup: failed to move staged md %s to %s/ (%r)",
            staged_md.name, dest_dir.name, e,
        )


def _write_reason(dest_dir: Path, filename: str, key: str, result: dict) -> None:
    """Drop a small `<filename>.reason` note beside a doc routed to failed/ on a done==0 build.

    Records WHY it failed — the `done`/`error`/`sections` split from `ingest_document` — so an
    operator can see the file built NOTHING (every section errored, e.g. a transient
    embedding/LLM outage) and is RE-DROPPABLE (optionally with a `.force` sidecar to rebuild),
    not a permanent per-doc defect. Best-effort: never raises — a missing reason note must not
    abort the drain (the failed/ move already stands on its own)."""
    try:
        done = result.get("done", 0) if isinstance(result, dict) else 0
        error = result.get("error", 0) if isinstance(result, dict) else 0
        sections = result.get("sections", done + error) if isinstance(result, dict) else 0
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / (filename + ".reason")).write_text(
            f"{key}: build produced NOTHING — done={done} error={error} sections={sections}. "
            "Every section failed to build (likely a transient embedding/LLM/build outage, not "
            "a per-doc defect). Re-drop this file (optionally with a .force sidecar) to retry.\n",
            encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001 — a reason-note write must not abort the drain
        log.warning("opdoc pickup: failed to write reason note for %s (%r)", filename, e)


async def drain_pending(rag) -> dict:
    """Scan the pending dir once and ingest each dropped operator doc against `rag`.

    Called from `main_loop` after `run_round` — same serial task, so single-writer/off-loop
    invariants hold (see the module docstring). Returns a small counter summary. NEVER raises:
    one bad file is routed to `failed/` and logged; the drain always completes so the scheduler
    loop keeps turning. (The caller in `main_loop` also wraps this in a belt-and-braces except.)
    """
    pending = _pending_dir()
    # `deferred` = left in pending on a TRANSIENT block (MinerU down): neither ingested nor
    # failed, will retry next round. Distinct from `disabled` (source flag OFF) so the summary
    # tells a server-outage backlog apart from a flag-gated one. `partial` = some sections built
    # and some errored (moved to processed/ but LOUD-warned); kept disjoint from `ingested`
    # (a FULL build) and `failed` (done==0, NOTHING built) so each drained file counts once (#84).
    counts = {"ingested": 0, "already": 0, "failed": 0, "partial": 0, "disabled": 0, "deferred": 0}
    if not pending.is_dir():
        return counts
    processed_dir = pending / "processed"
    failed_dir = pending / "failed"

    for path in sorted(pending.iterdir()):
        if not path.is_file():
            continue  # skip the processed/, failed/ and .ocr/ subdirs
        parsed = parse_drop_name(path.name)
        if parsed is None:
            continue  # not an operator-doc drop (.force sidecar, README.md, .txt, …)
        kind, key = parsed
        force = path.with_name(path.name + _FORCE_SIDECAR_SUFFIX).exists()

        # ── PDF front-end: OCR → markdown BEFORE ingest. A native .md/.markdown drop skips
        #    this entirely (staged_md stays None) and is ingested from its own path. ──
        ingest_path = str(path)
        staged_md: Path | None = None
        if path.suffix.lower() == _PDF_SUFFIX:
            try:
                # Gate BEFORE OCR: a disabled source must not burn a MinerU/GPU pass every
                # round. ingest_document re-checks (harmlessly) once we reach it.
                check_source_enabled(kind)
                staged_md = await _ocr_to_staged_md(path)
            except SourceDisabledError as e:
                if str(path) not in _disabled_logged:
                    _disabled_logged.add(str(path))
                    log.warning(
                        "opdoc pickup: %s source disabled — leaving %s in pending until enabled (%s)",
                        key, path.name, e,
                    )
                counts["disabled"] += 1
                continue
            except MineruTransportError as e:
                # TRANSIENT: MinerU down/unreachable → LEAVE the pdf in pending (retry next
                # round when MinerU is back). Do NOT move to failed/. Log ONCE (SourceDisabled
                # pattern) so a persistent outage doesn't spam the log every round.
                if str(path) not in _mineru_down_logged:
                    _mineru_down_logged.add(str(path))
                    log.warning(
                        "opdoc pickup: %s MinerU unreachable — leaving %s in pending until "
                        "MinerU is back (%s)", key, path.name, e,
                    )
                counts["deferred"] += 1
                continue
            except MineruExtractionError as e:
                # TERMINAL per-doc OCR defect (400/422 / structured-500 / thin md) → failed/.
                log.error("opdoc pickup: %s MinerU OCR failed (per-doc defect) → failed/ (%r)", key, e)
                _mineru_down_logged.discard(str(path))
                _move(path, failed_dir)
                counts["failed"] += 1
                continue
            # MinerU answered → the pdf is no longer "down"; clear its once-log latch.
            _mineru_down_logged.discard(str(path))
            ingest_path = str(staged_md)

        try:
            result = await ingest_document(rag, kind, key, ingest_path, force=force)
        except AlreadyIngestedError:
            # Push-once: the key is already in the graph — nothing to do, it's DONE.
            log.info("opdoc pickup: %s already ingested (push-once) → processed/", key)
            _move(path, processed_dir)
            _discard_staged(staged_md, processed_dir)
            _disabled_logged.discard(str(path))
            counts["already"] += 1
        except SourceDisabledError as e:
            # Source class flag is OFF — LEAVE the file so a later re-enable picks it up; log ONCE.
            if str(path) not in _disabled_logged:
                _disabled_logged.add(str(path))
                log.warning(
                    "opdoc pickup: %s source disabled — leaving %s in pending until enabled (%s)",
                    key, path.name, e,
                )
            counts["disabled"] += 1
        except Exception as e:  # noqa: BLE001 — one bad file must NOT crash the scheduler loop
            log.exception("opdoc pickup: %s FAILED to ingest → failed/ (%r)", key, e)
            _move(path, failed_dir)
            _discard_staged(staged_md, failed_dir)
            _disabled_logged.discard(str(path))
            counts["failed"] += 1
        else:
            # `ingest_document` RETURNED — but a return is NOT proof of success. Inspect the
            # done/error section split (#84): a book where EVERY section failed to build comes
            # back done=0 WITHOUT raising, and must not be silently marked done + lost.
            done = result.get("done", 0) if isinstance(result, dict) else 0
            error = result.get("error", 0) if isinstance(result, dict) else 0
            sections = result.get("sections", done + error) if isinstance(result, dict) else 0
            if done > 0 and error == 0:
                # FULL success: every section built.
                log.info("opdoc pickup: ingested %s (force=%s) → processed/ %s", key, force, result)
                _move(path, processed_dir)
                _discard_staged(staged_md, processed_dir)
                _disabled_logged.discard(str(path))
                counts["ingested"] += 1
            elif done > 0:
                # PARTIAL: some sections landed in the graph, some errored. The landed sections
                # ARE ingested (re-scanning would hit push-once), so move to processed/ — but
                # LOUD-warn with the split so an operator can re-drop with a `.force` sidecar to
                # rebuild the errored sections.
                log.warning(
                    "opdoc pickup: %s PARTIAL ingest (done=%d error=%d of %d section(s)) → "
                    "processed/ — some sections landed; re-drop with a .force sidecar to rebuild "
                    "the errored ones. %s", key, done, error, sections, result,
                )
                _move(path, processed_dir)
                _discard_staged(staged_md, processed_dir)
                _disabled_logged.discard(str(path))
                counts["partial"] += 1
            else:
                # done == 0: NOTHING built (a transient build outage errored every section — e.g.
                # the #84 embedding break took textbook:Bubeck2015 to done=0/error=51). This is a
                # REAL FAILURE, NOT a success: route to failed/ + drop a `.reason` note so it's
                # visibly failed + re-droppable, never silently "done" in processed/ and lost.
                log.error(
                    "opdoc pickup: %s built NOTHING (done=0 error=%d of %d section(s)) → failed/ "
                    "— likely a transient build outage; re-drop to retry. %s",
                    key, error, sections, result,
                )
                _move(path, failed_dir)
                _discard_staged(staged_md, failed_dir)
                _write_reason(failed_dir, path.name, key, result)
                _disabled_logged.discard(str(path))
                counts["failed"] += 1

    if any(counts.values()):
        log.info("opdoc pickup drain: %s", counts)
    return counts
