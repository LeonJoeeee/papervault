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

OUTCOME ROUTING (a bad file must NEVER crash the scheduler loop):
  success                → move file (+ any `.force` sidecar) to `processed/`
  AlreadyIngestedError   → move to `processed/` (push-once: it's already done)
  SourceDisabledError    → LEAVE in place, log ONCE (source flag OFF; a re-enable picks it up)
  any other error        → move to `failed/` + log LOUD (malformed key, enqueue blip, …)

RESTART RECOVERY (issue #81 known gap): an in-flight ingest interrupted by a restart leaves a
`processing` ledger row the paper `reconcile_terminal` (scans only `ingest_source="paper"`)
won't clean. MVP recovery = re-drop the file with a `.force` sidecar: force purges the stuck
rows then re-ingests clean. A boot-time operator-doc reconcile pass is durable follow-up.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from papervault.knowledge.ingest.operator_docs import (
    KINDS,
    AlreadyIngestedError,
    SourceDisabledError,
    ingest_document,
)

log = logging.getLogger("ks.scheduler.opdoc_pickup")

DEFAULT_PENDING_DIR = "/data/paper-vault/operator-drop/pending"
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_FORCE_SIDECAR_SUFFIX = ".force"

# Files whose source class is currently DISABLED that we've already logged, so a disabled
# source doesn't spam the log once per round (the file legitimately stays in pending until the
# flag is turned on). Cleared for a file once it successfully ingests. Process-local by design.
_disabled_logged: set[str] = set()


def _pending_dir() -> Path:
    return Path(os.getenv("KS_OPDOC_PENDING_DIR", DEFAULT_PENDING_DIR))


def parse_drop_name(name: str) -> tuple[str, str] | None:
    """`textbook-<Key>.md` → (kind, `<kind>:<Key>`); None if not an operator-doc drop.

    The kind is the prefix before the FIRST hyphen; the key body is everything after (notebook
    keys legitimately contain hyphens, e.g. `notebook-idea23-c12` → `idea23-c12`). Returns None
    for a non-markdown file or any name without a recognized `<kind>-` prefix (so `.force`
    sidecars, `paper-*.md`, `README.md`, plain `.txt`, etc. are silently skipped).
    """
    p = Path(name)
    if p.suffix.lower() not in _MARKDOWN_SUFFIXES:
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


async def drain_pending(rag) -> dict:
    """Scan the pending dir once and ingest each dropped operator doc against `rag`.

    Called from `main_loop` after `run_round` — same serial task, so single-writer/off-loop
    invariants hold (see the module docstring). Returns a small counter summary. NEVER raises:
    one bad file is routed to `failed/` and logged; the drain always completes so the scheduler
    loop keeps turning. (The caller in `main_loop` also wraps this in a belt-and-braces except.)
    """
    pending = _pending_dir()
    counts = {"ingested": 0, "already": 0, "failed": 0, "disabled": 0}
    if not pending.is_dir():
        return counts
    processed_dir = pending / "processed"
    failed_dir = pending / "failed"

    for path in sorted(pending.iterdir()):
        if not path.is_file():
            continue  # skip the processed/ and failed/ subdirs
        parsed = parse_drop_name(path.name)
        if parsed is None:
            continue  # not an operator-doc drop (.force sidecar, README.md, .txt, …)
        kind, key = parsed
        force = path.with_name(path.name + _FORCE_SIDECAR_SUFFIX).exists()
        try:
            result = await ingest_document(rag, kind, key, str(path), force=force)
        except AlreadyIngestedError:
            # Push-once: the key is already in the graph — nothing to do, it's DONE.
            log.info("opdoc pickup: %s already ingested (push-once) → processed/", key)
            _move(path, processed_dir)
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
            _disabled_logged.discard(str(path))
            counts["failed"] += 1
        else:
            log.info("opdoc pickup: ingested %s (force=%s) → processed/ %s", key, force, result)
            _move(path, processed_dir)
            _disabled_logged.discard(str(path))
            counts["ingested"] += 1

    if any(counts.values()):
        log.info("opdoc pickup drain: %s", counts)
    return counts
