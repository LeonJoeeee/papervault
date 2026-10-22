"""On-disk paper library: load / save / dedupe / key allocation."""

from __future__ import annotations

import datetime as _dt
import asyncio
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from papervault.config import merge_v2_enabled, reopen_v2_enabled, terminal_v2_enabled

from .download_telemetry import enrich, manifest_lock
from .models import DOWNLOAD_STATUS_PENDING, IdentityRecovery, Paper, base_key, normalize_title
from .sources.inspire import merge_inspire_fields, retained_inspire_fields
from .sources.identity import merge_source_identities
from .state_v2 import Policy, TerminalState, after_status_write, retain_observed_restrictions


def _is_nonempty_file(path: Path) -> bool:
    """True iff ``path`` is a regular file with size > 0.

    TOCTOU-safe: probes with a SINGLE ``stat()`` (no is_file()+stat() gap a
    background worker / janitor could unlink the file inside). A missing file /
    deleted-mid-call file returns False rather than raising FileNotFoundError —
    closing the race that could crash a get_paper batch or a serve path.
    """
    try:
        st = path.stat()
    except OSError:
        return False
    import stat as _stat
    return _stat.S_ISREG(st.st_mode) and st.st_size > 0


# ============== ingress quality gate ==============
# These patterns catch metadata that schemas accept but are obviously not
# papers — PDF preamble fragments, parsing artifacts, NSF submissions,
# affiliations leaked into authors[]. New on 2026-05-07 after audit found
# 4 phantoms that had slipped in via Semantic Scholar candidate batches.

_PHANTOM_TITLE_PREFIXES = (
    'submission ', 'page ', 'rfi', 'response to nsf',
    'request for information', 'technical report no',
    # LaTeX preamble lines that get OCR'd as titles
    'accepted by ',
)
_PHANTOM_TITLE_FRAGMENTS = (
    'preprint typeset using',
)
_GARBLED_OCR_PATTERNS = (
    re.compile(r'\bournal of\b', re.I),     # "Journal" with dropped J
    re.compile(r'\bosmology\b', re.I),       # "Cosmology" with dropped C
    re.compile(r'\bstroparticle\b', re.I),   # "Astroparticle" with dropped A
)
_AFFILIATION_MARKERS = (
    'university', 'institut', 'observator', 'laboratory', 'laboratoire',
    'national lab', 'academy of sciences', 'department of', 'school of',
    'centre for', 'center for', 'cnrs ', 'inaf ', 'max-planck', 'max planck',
)


def _validate_paper_metadata(paper_data: dict) -> tuple[bool, str]:
    """Heuristic phantom detector. Returns (ok, reason).

    Catches:
      - empty / too-short / too-long titles
      - LaTeX preamble fragments
      - "PAGE N" / NSF boilerplate
      - garbled OCR (e.g. "ournal of cosmology" — first letter dropped)
      - affiliations that leaked into the authors[] field
      - the orphan "on Matter…" fragment we found from a workshop TOC
    """
    title = (paper_data.get('title') or '').strip()
    if not title:
        return False, 'no title'
    if len(title) < 15:
        return False, f'title too short ({len(title)} chars)'
    if len(title) > 500:
        return False, 'title too long'
    tlow = title.lower()
    if any(tlow.startswith(p) for p in _PHANTOM_TITLE_PREFIXES):
        return False, 'non-paper boilerplate title prefix'
    if any(f in tlow for f in _PHANTOM_TITLE_FRAGMENTS):
        return False, 'LaTeX preamble fragment in title'
    if re.search(r'\bPAGE\s+\d+\b', title):
        return False, '"PAGE N" boilerplate in title'
    if title.startswith('on Matter'):
        return False, 'orphan workshop-TOC fragment'
    for pat in _GARBLED_OCR_PATTERNS:
        if pat.search(title):
            return False, 'garbled-OCR title (dropped leading letter)'
    # Affiliations leaked into authors[]
    authors = paper_data.get('authors') or []
    bad_author_count = 0
    for a in authors:
        if not isinstance(a, str):
            continue
        al = a.lower()
        if any(marker in al for marker in _AFFILIATION_MARKERS):
            bad_author_count += 1
        elif a.count('-') >= 4:  # excessive hyphens
            bad_author_count += 1
    # Reject if at least one bad author AND it's the majority of the list.
    # (single bad author in a long list is tolerable — CrossRef refetch can fix it)
    if bad_author_count and authors and bad_author_count >= max(1, len(authors) // 2):
        return False, 'authors[] contains institutional affiliations'
    return True, ''


def _atomic_write(path: Path, text: str, *, durable: bool = False) -> None:
    """Durable atomic write: tmp file → fsync(tmp) → rename → fsync(parent).

    D14: the rename is atomic so a reader never sees a half-written file, but
    rename alone is not crash-durable — on power loss after the rename the file
    can still be lost or truncated if neither the data nor the directory entry
    reached stable storage. So we ``fsync`` the tmp file's *data* before the
    rename, then ``fsync`` the *parent directory* after, so the new directory
    entry is durable too. Best-effort: a platform without ``O_*``/``fsync`` on
    directories (rare on the deploy target) degrades to the old behavior rather
    than raising, since the rename itself already gives the reader-atomicity we
    rely on at runtime.
    ``durable=True`` propagates both fsync failures for campaign claim barriers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            if durable:
                raise
    tmp.replace(path)
    _fsync_dir(path.parent, durable=durable)


def _fsync_dir(directory: Path, *, durable: bool = False) -> None:
    """fsync a directory so a rename into it is durable (D14). Best-effort."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        if durable:
            raise
        return
    try:
        os.fsync(fd)
    except OSError:
        if durable:
            raise
    finally:
        os.close(fd)


def _normalize_doi(doi: str) -> str:
    from .fetch import normalize_doi
    return normalize_doi(doi).strip().lower()


def _normalize_arxiv(arxiv_id: str) -> str:
    """Canonicalize an arxiv id for indexing / lookup.

    Strips BOTH the canonical ``arXiv:`` prefix AND the version suffix
    (v1, v2, ...) so the index is keyed by the bare id and every input form
    resolves to the same row:

        ``arXiv:1234.5678v3`` → ``1234.5678`` ← ``1234.5678v1`` ← ``1234.5678``

    Reuses ``fetch.normalize_arxiv`` (the same ``_ARXIV_RE`` group that powers
    the recognizer) so the recognizer and the normalizer can't drift: a form
    ``looks_like_arxiv`` accepts is a form this function canonicalizes. The
    prefix-strip is the bug fix — before it, ``arXiv:1234.5678`` (a recognized,
    canonical form) missed the bare-keyed ``_by_arxiv`` index and a held paper
    falsely reported not_found. A non-arxiv-shaped string (no regex match)
    falls back to the bare version-strip so dedup of malformed-but-present ids
    is unchanged.
    """
    if not arxiv_id:
        return ""
    from . import fetch
    return fetch.normalize_arxiv(arxiv_id)


def _title_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio on normalized titles. 0.0..1.0."""
    from difflib import SequenceMatcher
    na, nb = normalize_title(a or ""), normalize_title(b or "")
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _author_lastname_jaccard(authors_a: list[str], authors_b: list[str]) -> float:
    """Jaccard overlap on author last names. Tolerates name format
    variation between data sources (Crossref truncated, SS full)."""
    def lastnames(names: list[str]) -> set[str]:
        out = set()
        for n in names or []:
            toks = (n or "").strip().split()
            if toks:
                out.add(toks[-1].lower().strip(".,;"))
        return out
    a, b = lastnames(authors_a), lastnames(authors_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Library:
    """A directory holding library.bib + index.json + pdfs/ + extracts/."""

    def __init__(self, root: Optional[str | Path] = None, *, authority: str | None = None,
                 _handoff_token: str | None = None, read_only: bool = False):
        import threading
        import weakref
        authority = authority or os.environ.get("PAPERVAULT_CATALOG_AUTHORITY", "json")
        if authority not in {"json", "postgres"}:
            raise ValueError("PAPERVAULT_CATALOG_AUTHORITY must be json or postgres")
        self.read_only = read_only
        self._catalog_backend = None
        if authority == "postgres":
            from papervault.store.persistence import PostgresPersistence
            self._catalog_backend = PostgresPersistence.from_environment()
            self._catalog_backend.fence_id = _handoff_token
        self._merge_execution_lock = threading.RLock()
        self._merge_active: dict[str, int] = {}
        self._merge_queues = weakref.WeakSet()
        # Vault root = the library's on-disk store. Read the env LIVE at construction
        # (not the import-frozen config value) so a runtime override — e.g. the CLI's
        # --library-path, which sets PAPER_LIBRARY_PATH — is honored. config.VAULT_PATH
        # is the last-resort default (its own env capture happened at import).
        from papervault import config
        env_root = os.environ.get("PAPERVAULT_VAULT") or os.environ.get("PAPER_LIBRARY_PATH")
        # expanduser the live env value: config.VAULT_PATH is already expanded, but a raw
        # env read is not — a tilde path (the style .env.example ships) would otherwise
        # become a literal cwd-relative './~/...' dir, diverging from what doctor reports.
        self.root = Path(os.path.expanduser(str(root or env_root or config.VAULT_PATH)))
        from filelock import FileLock
        self._catalog_save_lock = FileLock(str(self.root / ".write.lock")) if self._catalog_backend else None
        from papervault.store.fencing import writer_lease
        lease = None if read_only else writer_lease(self.root, authority, handoff_token=_handoff_token)
        self._writer_finalizer = weakref.finalize(self, lease.close) if lease else None
        self._writer_closed = False
        self.bib_path = self.root / "library.bib"
        self.index_path = self.root / "index.json"
        # D14: the last known-good index, rotated in on every successful save.
        self.index_bak_path = self.root / "index.json.bak"
        self.topics_dir = self.root / "topics"
        self.pdfs_dir = self.root / "pdfs"
        self.md_dir = self.root / "extracts" / "md"
        self.txt_dir = self.root / "extracts" / "txt"
        self.manifest_path = self.root / "manifest.log"

        if not read_only:
            for d in (self.topics_dir, self.pdfs_dir, self.md_dir, self.txt_dir):
                d.mkdir(parents=True, exist_ok=True)

        self._papers: dict[str, Paper] = {}
        # Compact in-process fingerprints detect touches, including nested edits.
        # They are not persisted and never authorize a bulk legacy backfill.
        self._state_v2_fingerprints: dict[str, int] = {}
        self._by_doi: dict[str, str] = {}
        self._by_arxiv: dict[str, str] = {}
        self._by_title: dict[str, str] = {}
        self._identity_reservations: set[tuple[str, str, str, object]] = set()
        # Save-cost controls (issue #34 root cause: per-item full-index writes starved
        # the event loop during boot sweeps). Debounce is safe because the boot
        # reconcile re-registers anything a crash loses between flushes.
        self._rotation_validated = False
        self._save_min_interval = float(os.environ.get("PAPERVAULT_SAVE_MIN_INTERVAL_S", "0"))  # 0 = off (legacy semantics); deployments opt in
        self._last_index_save = 0.0
        self._index_dirty = False
        self._bib_min_interval = float(os.environ.get("PAPERVAULT_BIB_MIN_INTERVAL_S", "60"))
        self._last_bib_save = 0.0
        from .services.correction_ledger import CorrectionLedger
        self.corrections = CorrectionLedger(self.root)
        self.correction_run_lock = asyncio.Lock()
        self.correction_next_at = 0.0
        self._load()

    def close(self) -> None:
        """Release the writer fence when an embedded Library's lifetime ends."""
        if self._writer_finalizer:
            self._writer_finalizer()
        self._writer_closed = True

    def _check_writer(self):
        if self.read_only:
            raise PermissionError("Library is read-only")
        if self._writer_closed:
            from papervault.store.fencing import WriterFenced
            raise WriterFenced("Library is closed; reopen under the current catalog authority")

    # ----- load / save -----

    def _load(self) -> None:
        if self._catalog_backend is not None:
            self._papers = self._catalog_backend.load()
            for paper in self._papers.values():
                self._state_v2_fingerprints[paper.key] = hash(paper.model_dump_json())
                self._reindex(paper)
            return
        if self.read_only:
            from .state_v2 import check_index_readable
            check_index_readable(self.root)
            data = self._read_index_file(self.index_path) or self._read_index_file(self.index_bak_path)
            if data is None or not isinstance(data.get("papers"), dict):
                raise ValueError(f"no readable library index at {self.root}")
            # Inspection must refuse malformed records, never omit one silently.
            for key, raw in data["papers"].items():
                paper = Paper.model_validate(raw)
                if paper.key != key:
                    raise ValueError("library index key differs from record")
                self._papers[key] = paper
                self._reindex(paper)
            return
        if not self.index_path.exists():
            # No primary index. If a backup survived (e.g. the primary was
            # truncated to zero bytes by a crash mid-write and then the half
            # got removed), recover from it instead of starting empty (D14).
            if self.index_bak_path.exists():
                data = self._read_index_file(self.index_bak_path)
                if data is not None:
                    self._ingest_records(data)
                    return
            self._save_index()
            return

        data = self._read_index_file(self.index_path)
        if data is None:
            # Primary index is unreadable (truncated / corrupt JSON from a
            # crash mid-write). Fall back to the last known-good backup rather
            # than silently loading an EMPTY library and then overwriting the
            # backup on the next save — that would lose the whole vault (D14).
            self.log({"event": "index_load_corrupt",
                      "path": str(self.index_path),
                      "action": "falling back to index.json.bak"})
            data = self._read_index_file(self.index_bak_path)
            if data is None:
                # Neither readable. Refuse to proceed as if empty — a blank
                # library would clobber both files on the next save. Surface
                # the corruption loudly so an operator can intervene.
                raise RuntimeError(
                    f"index.json at {self.index_path} is corrupt and no usable "
                    f"index.json.bak was found; refusing to load an empty library "
                    f"(would clobber the vault on next save). Restore a backup.")
        self._ingest_records(data)

    @staticmethod
    def _read_index_file(path: Path) -> Optional[dict]:
        """Parse one index file. Returns the dict on success, ``None`` if the
        file is missing / empty / not valid JSON (D14 defensive load)."""
        try:
            text = path.read_text()
        except OSError:
            return None
        if not text.strip():
            return None
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _ingest_records(self, data: dict) -> None:
        """Build the in-memory tables from a parsed index, tolerating a single
        corrupt record without sinking the whole library (D14).

        A record that fails ``Paper(**raw)`` validation (schema drift, a
        truncated value) is logged and skipped — it must not abort loading the
        other ~4000 good records.
        """
        bad = 0
        for key, raw in (data.get("papers") or {}).items():
            try:
                paper = Paper(**raw)
            except Exception as exc:
                bad += 1
                self.log({"event": "index_record_skipped", "key": key,
                          "error": repr(exc)[:200]})
                continue
            self._papers[key] = paper
            self._state_v2_fingerprints[key] = hash(paper.model_dump_json())
            self._reindex(paper)
        if bad:
            self.log({"event": "index_load_skipped_records", "count": bad})

    def _reindex(self, p: Paper) -> None:
        key = (p.acquisition_state.alias_key or p.key) if merge_v2_enabled() or p.merge_budget else p.key
        if p.doi:
            self._by_doi[_normalize_doi(p.doi)] = key
        if p.arxiv_id:
            self._by_arxiv[_normalize_arxiv(p.arxiv_id)] = key
        nt = normalize_title(p.title)
        if nt:
            self._by_title[nt] = key

    def _save_index(self, *, durable: bool = False) -> None:
        self._check_writer()
        if self._catalog_backend is not None:
            self._save_catalog()
            return
        # Metadata/queue saves must not strip an unsupported format header,
        # even with v2 flags off. Refuse before rotating the last-good index.
        from .state_v2 import check_index_readable
        check_index_readable(self.root)
        # D14: rotate the current on-disk index into ``.bak`` BEFORE overwriting
        # it, so a known-good copy survives a crash mid-write of the new index.
        # We only rotate a copy that actually parses — never overwrite a good
        # backup with a freshly-corrupted primary.
        if self.index_path.exists():
            if not self._rotation_validated:
                # First rotation this process: parse-validate before trusting (D14).
                if self._read_index_file(self.index_path) is not None:
                    try:
                        _atomic_write(self.index_bak_path, self.index_path.read_text())
                        self._rotation_validated = True
                    except OSError:
                        pass  # backup is best-effort; the durable primary write follows
            elif self._primary_sane():
                # Later saves: the primary is a completed atomic write (ours or a peer's
                # under the same file lock). Rotate by O(1) rename instead of a full-file
                # parse+copy (the 33 MB-per-item cost behind issue #34). Crash between
                # this replace and the write below leaves only .bak; _load() recovers.
                # The cheap sanity gate keeps out-of-band corruption (truncation,
                # bit-rot) from clobbering the last known-good .bak (D14).
                try:
                    os.replace(self.index_path, self.index_bak_path)
                except OSError:
                    pass
            else:
                # Sanity gate failed: primary corrupted out-of-band. PRESERVE .bak
                # (do not rotate) and fall back to full validation on the next save.
                self._rotation_validated = False
        records, fingerprints = {}, {}
        enabled = terminal_v2_enabled() or reopen_v2_enabled() or merge_v2_enabled()
        for key, paper in self._papers.items():
            fingerprint = hash(paper.model_dump_json())
            if (enabled and paper.terminal_state is None
                    and self._state_v2_fingerprints.get(key) != fingerprint):
                # Write-through only for a row actually changed since load/save.
                paper.terminal_state = paper.unquiesced_state
                fingerprint = hash(paper.model_dump_json())
            records[key] = paper.model_dump()
            fingerprints[key] = fingerprint
        data = {"version": 1, "papers": records}
        _atomic_write(self.index_path, json.dumps(data, ensure_ascii=False, indent=2), durable=durable)
        self._state_v2_fingerprints = fingerprints

    def _save_catalog(self) -> None:
        from papervault.store.persistence import CatalogConflict
        # The PG delta preserves raw legacy payloads. Metadata edits must not
        # materialize read-derived state; ingress/status writers record actual
        # acquisition changes before this persistence boundary.
        try:
            rulings = self._catalog_backend.save(self._papers)
        except CatalogConflict as exc:
            self._index_dirty = True
            self.log({"event": "catalog_conflict", "keys": list(exc.keys)})
            raise
        from .services.operator_actions import mirror_events
        mirror_events(self, rulings)
        self._by_doi.clear()
        self._by_arxiv.clear()
        self._by_title.clear()
        for paper in self._papers.values():
            self._state_v2_fingerprints[paper.key] = hash(paper.model_dump_json())
            self._reindex(paper)

    def catalog_completion(self, token):
        from contextlib import contextmanager, nullcontext
        if not self._catalog_backend:
            return nullcontext()

        @contextmanager
        def guarded():
            # All paths take file lock -> PG lock in the same order. save()
            # reenters this exact FileLock instance during staged publication.
            with self._catalog_save_lock.acquire(timeout=30):
                with self._catalog_backend.completion(token):
                    yield
        return guarded()

    def _primary_sane(self) -> bool:
        """Cheap structural check (no full parse) that the primary index looks like a
        completed JSON object write — guards the O(1) rotation against truncation."""
        try:
            size = self.index_path.stat().st_size
            if size < 16:
                return False
            with self.index_path.open("rb") as fh:
                head = fh.read(1)
                fh.seek(-2, 2)
                tail = fh.read(2)
            return head == b"{" and tail.rstrip().endswith(b"}")
        except OSError:
            return False

    def flush_if_dirty(self, **kw) -> bool:
        """Force-persist iff a debounced save left state dirty. Returns True if it wrote."""
        if not self._index_dirty:
            return False
        self.save(force=True, **kw)
        return True

    def _save_bib(self) -> None:
        # Pass `library=self` so Paper.to_bibtex emits `file = {...}` fields
        # pointing to local PDFs / extracts for Zotero auto-attach on import.
        # Domain-quarantined papers are excluded — library.bib is a clean view
        # (no off-domain \cite keys leaking into the legal pool).
        entries = [p.to_bibtex(library=self) for p in self._papers.values()
                   if p.domain_status is None]
        _atomic_write(self.bib_path, "\n\n".join(entries) + "\n" if entries else "")

    def save(self, *, force: bool = False, lock_timeout: float = 30.0, durable: bool = False) -> None:
        """Persist index (+ BibTeX on its own cadence) under a process write lock.

        Debounced (issue #34): within ``PAPERVAULT_SAVE_MIN_INTERVAL_S`` (default 5 s)
        a non-forced save only marks the library dirty and returns — burst writers
        (the boot sweep, multi-paper ingests) do O(1) disk work per interval instead
        of per item. Callers at drain/shutdown points pass ``force=True``; anything
        a crash catches unflushed is re-registered by the next boot reconcile.
        BibTeX is a convenience view and rebuilds at most every
        ``PAPERVAULT_BIB_MIN_INTERVAL_S`` (default 60 s) or on force.

        The lock prevents two MCP server instances (or pipeline + MCP) from
        racing on index.json. Reads are unsynchronized — they always see a
        fully-written file because writes are atomic (.tmp → rename).
        Campaign callers pass ``force=True, durable=True`` to bypass debounce and
        require a stable-storage acknowledgment for the index.
        """
        self._check_writer()
        now = time.monotonic()
        if (not force and self._save_min_interval > 0
                and (now - self._last_index_save) < self._save_min_interval):
            self._index_dirty = True
            return
        from filelock import FileLock, Timeout
        lock_path = self.root / ".write.lock"
        try:
            write_lock = self._catalog_save_lock or FileLock(str(lock_path), timeout=lock_timeout)
            with write_lock.acquire(timeout=lock_timeout):
                self._save_index(durable=durable)
                self._last_index_save = time.monotonic()
                self._index_dirty = self._catalog_backend.dirty(self._papers) if self._catalog_backend else False
                if force or (now - self._last_bib_save) >= self._bib_min_interval:
                    self._save_bib()
                    self._last_bib_save = time.monotonic()
        except Timeout as exc:
            raise RuntimeError(f"library write lock timed out after {lock_timeout}s") from exc

    # ----- lookup -----

    def find(self, *, doi: str = "", arxiv_id: str = "",
             title: str = "") -> Optional[Paper]:
        """Look up current identities, retaining active verified-fill claims.

        Download workers can fill/clear identifiers on the shared Paper in
        place. A stale cache must not authorize allocating another citation key.
        """
        with self._merge_execution_lock:
            return self._find_current(doi=doi, arxiv_id=arxiv_id, title=title)

    def _find_current(self, *, doi: str, arxiv_id: str, title: str) -> Optional[Paper]:
        for field, value, index, normalize in (
            ("doi", doi, self._by_doi, _normalize_doi),
            ("arxiv_id", arxiv_id, self._by_arxiv, _normalize_arxiv),
            ("title", title, self._by_title, normalize_title),
        ):
            normalized = normalize(value)
            if not normalized:
                continue
            paper = self._papers.get(index.get(normalized, ""))
            # Verified fills reserve the index before publishing the field.
            # Distinguish those live claims from identifiers cleared by a worker.
            reserved = paper is not None and not getattr(paper, field) and any(
                claim[:3] == (field, normalized, paper.key) for claim in self._identity_reservations)
            if paper is None or (not reserved and normalize(getattr(paper, field)) != normalized):
                paper = next((p for p in self._papers.values()
                              if normalize(getattr(p, field)) == normalized), None)
                if paper is None:
                    index.pop(normalized, None)
                    continue
                index[normalized] = paper.key
            from .merge_v2 import resolve_alias
            return resolve_alias(self, paper.key) if merge_v2_enabled() or paper.merge_budget else paper
        return None

    def get(self, key: str) -> Optional[Paper]:
        return self._papers.get(key)

    def all_papers(self, *, include_quarantined: bool = False) -> list[Paper]:
        """Papers in the library. By DEFAULT excludes domain-quarantined papers
        (``domain_status`` set: off_domain / non_paper / bad_extract). This is the
        single clean-view chokepoint feeding the search snapshot, CLI, bib, and
        external consumers — so contamination stays hidden everywhere at once.
        Pass ``include_quarantined=True`` for operator/audit views (or internal
        dedup/persistence, which use ``self._papers`` directly anyway)."""
        if include_quarantined:
            return list(self._papers.values())
        return [p for p in self._papers.values() if p.domain_status is None
                and not ((merge_v2_enabled() or p.merge_budget) and p.acquisition_state.alias_key)]

    def keys(self) -> list[str]:
        return list(self._papers.keys())

    def set_domain_status(self, key: str, status: Optional[str],
                          tier: Optional[str] = None) -> bool:
        """Mark (or clear) a paper's domain membership. ``status=None`` clears the
        quarantine (back to in-domain). Append-only-safe: updates metadata only,
        never deletes. Does NOT call ``save()`` — caller batches saves."""
        self._check_writer()
        p = self._papers.get(key)
        if p is None:
            return False
        p.domain_status = status
        if tier is not None:
            p.domain_tier = tier
        self._papers[key] = p
        return True

    # ----- mutate -----

    def allocate_key(self, first_author: str, year: int | str | None) -> str:
        """Return an unused citation key for a new paper."""
        base = base_key(first_author, year)
        if base not in self._papers:
            return base
        # Conflict resolution: append a, b, c, ...
        for suffix in (chr(c) for c in range(ord("a"), ord("z") + 1)):
            candidate = base + suffix
            if candidate not in self._papers:
                return candidate
        # Extreme fallback (>26 collisions)
        i = 1
        while f"{base}_{i}" in self._papers:
            i += 1
        return f"{base}_{i}"

    def upsert(self, paper_data: dict) -> tuple[Optional[Paper], bool]:
        """Add or update a paper. Returns (paper, is_new).

        Dedupe ladder (first match wins, all checks case-insensitive):
          1. DOI exact
          2. arxiv_id exact (version-stripped)
          3. normalized title exact
          4. **Fuzzy preprint↔journal merge** (D7): when the new paper
             has DOI XOR arxiv_id and an existing paper has the other,
             check (title sim ≥ 0.85 + author Jaccard ≥ 0.7 +
             year diff ≤ 2). On match, merge: the merge always carries
             the new DOI into the existing entry (so "journal DOI wins"
             when a preprint already exists), and the cite key stays the
             existing one. Borderline matches (below the auto-merge bar)
             conservatively DON'T merge — false-positive merges are data
             loss, so a residual duplicate is the safer outcome.

        Existing entries are merged: non-empty incoming fields overwrite
        blanks; existing non-blank fields are NOT overwritten (so a
        preprint title isn't lost when the journal version arrives —
        unless callers actively choose to update it).

        New papers go through ``_validate_paper_metadata`` first; phantom-shaped
        metadata is rejected with a manifest log entry and (None, False) returned.
        Already-existing papers (merge path) skip validation — they were
        validated at first ingress (or grandfathered before validation existed).
        """
        self._check_writer()
        existing = self.find(
            doi=paper_data.get("doi", "") or "",
            arxiv_id=paper_data.get("arxiv_id", "") or "",
            title=paper_data.get("title", "") or "",
        )
        if existing:
            consolidated = self._consolidate_identity(existing, paper_data)
            if consolidated is not None:
                return consolidated, False
            updated = self._merge(existing, paper_data)
            self._papers[existing.key] = updated
            self._reindex(updated)
            return updated, False

        # Step 4: fuzzy preprint ↔ journal merge (D7).
        fuzzy_existing = self._find_preprint_journal_dup(paper_data)
        if fuzzy_existing is not None:
            consolidated = self._consolidate_identity(fuzzy_existing, paper_data, fuzzy=True)
            if consolidated is not None:
                return consolidated, False
            updated = self._merge(fuzzy_existing, paper_data)
            self._papers[fuzzy_existing.key] = updated
            self._reindex(updated)
            self.log({
                "event": "dedup_merged_preprint_journal",
                "existing_key": fuzzy_existing.key,
                "existing_doi": fuzzy_existing.doi or "",
                "existing_arxiv_id": fuzzy_existing.arxiv_id or "",
                "new_doi": paper_data.get("doi", "") or "",
                "new_arxiv_id": paper_data.get("arxiv_id", "") or "",
                "new_title": (paper_data.get("title") or "")[:120],
            })
            return updated, False

        # New paper — apply quality gate
        ok, reason = _validate_paper_metadata(paper_data)
        if not ok:
            self.log({
                "event": "phantom_rejected",
                "reason": reason,
                "title": (paper_data.get("title") or "")[:120],
                "doi": paper_data.get("doi", "") or "",
                "arxiv_id": paper_data.get("arxiv_id", "") or "",
                "source": paper_data.get("source", "") or "",
            })
            return None, False

        first_author = ""
        if paper_data.get("authors"):
            first_author = paper_data["authors"][0]
        key = self.allocate_key(first_author, paper_data.get("year"))
        paper_data["key"] = key
        paper_data.setdefault("added_at", _dt.datetime.now(_dt.timezone.utc).isoformat())
        # Coerce common None-where-string fields (citation chase / OpenAlex
        # sometimes return None for missing strings).
        for field in ("title", "venue", "volume", "pages", "issue",
                      "abstract", "doi", "arxiv_id",
                      "paper_id", "url", "source", "download_status",
                      "download_source", "added_at",
                      "txt_engine", "txt_engine_version",
                      "md_engine", "md_engine_version"):
            if paper_data.get(field) is None:
                paper_data[field] = ""
        # New papers default to "pending" so the BG worker can find queue
        # work via a single status check. Empty string came from earlier
        # versions; harmonize on entry.
        if not paper_data.get("download_status"):
            paper_data["download_status"] = DOWNLOAD_STATUS_PENDING
        paper_data.update(retained_inspire_fields(paper_data))
        paper = Paper(**{k: v for k, v in paper_data.items()
                         if k in Paper.model_fields and k not in {"reopen_claim", "merge_intent", "merge_budget"}})
        if (terminal_v2_enabled() or reopen_v2_enabled() or merge_v2_enabled()) and paper.terminal_state is None:
            # This is a new ingress observation, not a cause-unrecorded legacy row.
            paper.terminal_state = after_status_write(TerminalState(policy=Policy.CLEAR), paper)
        self._papers[key] = paper
        self._reindex(paper)
        return paper, True

    def _consolidate_identity(self, existing: Paper, incoming: dict, *, fuzzy=False):
        """Only stored twins/cross-identifier promotion take the new drain barrier."""
        if existing.merge_intent and not merge_v2_enabled():
            return existing
        if not merge_v2_enabled():
            return None
        from .merge_v2 import MergeStore, resolve_alias, write_locked
        if existing.merge_intent:
            intent = existing.merge_intent
            return write_locked(lambda: MergeStore(self)._merge_records(
                intent.canonical_key, intent.absorbed_key, metadata=incoming)) or existing
        twin = None
        for name in ("doi", "arxiv_id"):
            value = incoming.get(name)
            if value:
                other = self.find(**{name: value})
                if other and other.key != existing.key and not other.acquisition_state.alias_key:
                    twin = other
                    break
        source_key = incoming.get("key")
        if twin is None and source_key and source_key != existing.key:
            other = resolve_alias(self, source_key)
            if other and not other.acquisition_state.alias_key:
                twin = other
        promoting = fuzzy or any(incoming.get(name) and not getattr(existing, name)
                                 for name in ("doi", "arxiv_id"))
        if twin is None and not promoting:
            return None

        def consolidate():
            nonlocal twin
            merger = MergeStore(self)
            if existing.merge_intent:
                intent = existing.merge_intent
                return merger._merge_records(intent.canonical_key, intent.absorbed_key, metadata=incoming) or existing
            if merger._can_quiesce(existing.key) is None:
                return existing
            if twin is not None and (merger._can_quiesce(twin.key) is None or twin.merge_intent is not None):
                return existing
            if twin is None and all(not incoming.get(name) or normalize(incoming[name]) == normalize(getattr(existing, name))
                                    for name, normalize in (("doi", _normalize_doi), ("arxiv_id", _normalize_arxiv))):
                self._merge(existing, incoming)
                self._reindex(existing)
                return existing
            if twin is None:
                # Persist the second observation before quiescing, so deferred
                # promotions survive restart and have two auditable source keys.
                payload = {k: v for k, v in incoming.items() if k in Paper.model_fields and v is not None
                           and k not in {"key", "reopen_claim", "merge_intent", "merge_budget"}}
                payload.setdefault("title", existing.title)
                payload.setdefault("authors", existing.authors)
                authors = payload.get("authors") or existing.authors
                key = self.allocate_key(authors[0] if authors else "", payload.get("year", existing.year))
                twin = Paper(key=key, **payload)
                self._papers[key] = twin
                self._reindex(twin)
            return merger._merge_records(existing.key, twin.key, metadata=incoming) or existing
        return write_locked(consolidate)

    def purge(self, key: str, reason: str = "") -> dict:
        """Remove a paper from the library. Breaks the append-only invariant —
        use only to clean up phantom / corrupt entries.

        Removes:
          - in-memory paper + indexes (doi/arxiv/title)
          - PDF + md + txt files on disk (if present)

        Does NOT call save() — caller is responsible for batching saves.
        Logs ``{"event": "purged", "key": ..., "reason": ...}`` to manifest
        so the action is observable + auditable.

        Returns a dict describing what was removed.
        """
        self._check_writer()
        if self._catalog_backend is not None:
            from papervault.store.repository import DeletionRefused
            raise DeletionRefused("PG purge is refused in P2; retention/tombstones await P5")
        existing = self._papers.get(key)
        if existing is None:
            return {"key": key, "removed": False, "reason": "not in library"}

        # Drop from indexes
        self._papers.pop(key, None)
        if existing.doi:
            self._by_doi.pop(_normalize_doi(existing.doi), None)
        if existing.arxiv_id:
            self._by_arxiv.pop(_normalize_arxiv(existing.arxiv_id), None)
        nt = normalize_title(existing.title)
        if nt:
            self._by_title.pop(nt, None)

        # Drop on-disk files
        files_removed = []
        for path in (self.pdf_path(key), self.md_path(key), self.txt_path(key)):
            if path.is_file():
                try:
                    path.unlink()
                    files_removed.append(str(path.relative_to(self.root)))
                except Exception:
                    pass

        self.log({"event": "purged", "key": key, "reason": reason or "manual",
                  "files_removed": files_removed,
                  "doi": existing.doi or "", "arxiv_id": existing.arxiv_id or ""})
        return {"key": key, "removed": True, "files_removed": files_removed}

    # Locator fields the enrich queue is allowed to fill (NEVER title/authors/
    # year/doi — those already passed ingest). Fill-blanks only.
    _ENRICH_FIELDS = ("venue", "volume", "pages", "issue")

    def enrich(self, key: str, fields: dict, *, enriched_at: str, provenance: dict | None = None) -> bool:
        """Targeted same-key metadata fill for the enrich queue (NOT upsert).

        Fill-blanks-merges ONLY ``venue/volume/pages/issue`` onto the existing
        paper by direct mutation — never re-runs find / dedup / validate, never
        touches title/authors/year/doi (the row already passed ingest with those),
        so it cannot mis-merge into another row. ALWAYS stamps ``enriched_at``
        (worker-owned guard) on EVERY attempt — success OR clean-miss — so an
        un-completable DOI is probed once, not re-enqueued every sweep. Returns
        True iff a locator blank was filled. Does NOT save() — caller batches the
        save under ``lib_write_lock``/the file lock.
        """
        self._check_writer()
        p = self._papers.get(key)
        if p is None:
            return False
        self.corrections.stage_fields(p, {f: fields[f] for f in self._ENRICH_FIELDS if f in fields},
                                      operation="enrich", provenance=provenance)
        changed = False
        for f in self._ENRICH_FIELDS:
            val = fields.get(f)
            val = val.strip() if isinstance(val, str) else val
            if val and not (getattr(p, f, "") or ""):
                setattr(p, f, val)
                changed = True
        p.enriched_at = enriched_at        # unconditional stamp (the termination guard)
        return changed

    def set_resolved_doi(self, key: str, doi: str, *,
                         provenance: str = "doi_resolved",
                         resolved_at: Optional[str] = None) -> tuple[str, str]:
        """Stamp a by-title-RESOLVED DOI onto an existing no-DOI stub, keeping
        the ``_by_doi`` index consistent. Returns ``(outcome, detail)`` where
        outcome is one of:

          * ``"set"``       — the DOI was empty, no other row holds this DOI;
                              wrote ``p.doi``, re-indexed ``_by_doi[doi]=key``,
                              appended ``provenance`` to ``p.source`` and
                              stamped ``p.resolved_doi_at``.
          * ``"collision"`` — the DOI already maps to ANOTHER row (a twin). We
                              do NOT stamp a duplicate identity (that would
                              corrupt ``_by_doi``); we route the stub through
                              the existing preprint↔journal merge semantics:
                              the DOI-holder absorbs the stub's blank-fill
                              fields and retains the stub as an alias. ``detail`` is the
                              surviving DOI-holder's key.
          * ``"has_doi"``   — the row already carries a non-empty DOI; NEVER
                              overwritten. ``detail`` is the existing DOI.
          * ``"no_paper"``  — no such key.

        With PAPERVAULT_MERGE_V2 enabled, collisions use the persisted drain
        barrier and retain the stub as an alias. ``"merge_waiting"`` means no
        identity was written; ``"alias"`` returns an already absorbed key's
        canonical destination. Persisted intents remain fenced with the flag off.

        Does NOT call ``save()`` — the caller batches the save under the write
        lock. The ``_by_doi`` write here is the load-bearing difference from
        the ``_try_arxiv_by_title`` precedent's raw ``paper.arxiv_id = ...``
        assignment (which left its index stale).
        """
        self._check_writer()
        p = self._papers.get(key)
        if p is None:
            return ("no_paper", "")
        if p.merge_budget and p.acquisition_state.alias_key:
            from .merge_v2 import resolve_alias
            canonical = resolve_alias(self, key)
            return ("alias", canonical.key if canonical else "")
        if p.merge_intent:
            intent = p.merge_intent
            holder = self.find(doi=(doi or "").strip())
            if (merge_v2_enabled() and holder is not None
                    and holder.key in {intent.canonical_key, intent.absorbed_key}):
                from .merge_v2 import MergeStore, write_locked
                merged = write_locked(lambda: MergeStore(self)._merge_records(
                    intent.canonical_key, intent.absorbed_key))
                return ("collision" if merged is not None else "merge_waiting", intent.canonical_key)
            return ("merge_waiting", intent.canonical_key)
        if (p.doi or "").strip():
            return ("has_doi", p.doi)  # never overwrite a real DOI
        doi = (doi or "").strip()
        if not doi:
            return ("no_paper", "empty doi")

        holder_key = self._by_doi.get(_normalize_doi(doi))
        if holder_key is not None and (merge_v2_enabled() or self._papers[holder_key].merge_budget):
            from .merge_v2 import resolve_alias
            holder = resolve_alias(self, holder_key)
            if holder is None:
                return ("merge_waiting", holder_key)
            holder_key = holder.key
        if holder_key is not None and holder_key != key:
            holder = self._papers.get(holder_key)
            if not merge_v2_enabled() or (holder is not None and holder.merge_intent):
                return ("merge_waiting", holder_key)
            from .merge_v2 import MergeStore, write_locked
            merged = write_locked(lambda: MergeStore(self)._merge_records(holder_key, key))
            return ("collision" if merged is not None else "merge_waiting", holder_key)

        # No collision — claim the DOI and re-index.
        p.doi = doi
        if provenance:
            p.source = f"{p.source}+{provenance}" if (p.source or "").strip() else provenance
        p.resolved_doi_at = resolved_at or _dt.datetime.now(_dt.timezone.utc).isoformat()
        self._papers[key] = p
        self._reindex(p)
        self.log({"event": "doi_resolved", "key": key, "doi": doi,
                  "source": p.source})
        return ("set", doi)

    def set_recovered_identity(self, key: str, recovery: IdentityRecovery) -> str:
        """Claim a verified identity on an identifierless row, without merging twins.

        The backfill caller owns verification and the normal save boundary.
        Collisions never purge another record or broaden the acquisition set.
        """
        self._check_writer()
        paper = self.get(key)
        if paper is None:
            return "no_paper"
        if paper.doi or paper.arxiv_id:
            self.corrections.stage_fields(paper, {"doi": recovery.doi, "arxiv_id": recovery.arxiv_id},
                                          operation="identity_backfill", provenance={"recovery": recovery.model_dump()})
            return "has_identifier"
        if not (recovery.doi or recovery.arxiv_id):
            return "empty_identity"
        for value, index, normalize in ((recovery.doi, self._by_doi, _normalize_doi),
                                        (recovery.arxiv_id, self._by_arxiv, _normalize_arxiv)):
            if value and index.get(normalize(value), key) != key:
                holder = self.get(index[normalize(value)])
                self.corrections.stage_identity(paper, recovery.model_dump(), holder,
                    provenance={"operation": "identity_backfill", "recovery": recovery.model_dump()})
                return "collision"
        applied = self.fill_verified_identifiers(key, doi=recovery.doi, arxiv_id=recovery.arxiv_id,
            provenance={"recovery": recovery.model_dump(), "verification": recovery.route,
                        "source_url": recovery.source_url})
        if not applied:
            return "collision"
        paper.identity_recovery = recovery.model_copy(update={
            "doi": applied.get("doi", ""), "arxiv_id": applied.get("arxiv_id", "")})
        self.log({"event": "identity_recovered", "key": key,
                  "recovery": paper.identity_recovery.model_dump()})
        return "set"

    def fill_verified_identifiers(self, key: str, *, doi: str = "",
                                  arxiv_id: str = "", provenance: dict | None = None) -> dict[str, str]:
        """Fill verified blanks on this row; never redirect a merge to a twin.

        Locked adoption arbitrates concurrent download workers. Unlike
        upsert/set_resolved_doi this never
        selects or purges another row. The normal caller still batches save().
        A merge-intent or absorbed alias refuses all fills.
        """
        self._check_writer()
        paper = self.get(key)
        if paper is None:
            return {}
        from .merge_v2 import blocked
        if blocked(paper):
            return {}
        return self._fill_verified_identifiers(paper, doi=doi, arxiv_id=arxiv_id, provenance=provenance)

    @staticmethod
    def _identifier_token(name: str, value: str, source: str = "") -> str:
        if name == "doi":
            return _normalize_doi(value)
        if name == "arxiv_id":
            return _normalize_arxiv(value)
        if name == "inspire_record_id":
            from .sources.inspire import inspire_record_id
            return inspire_record_id(value)
        if name == "paper_id":
            # Source IDs belong to a namespace, not a global numeric-ID space.
            from .sources.identity import PRIMARY_NAMESPACES
            sources = re.split(r"[\s,;+|]+", source.strip().lower())
            namespace = next((PRIMARY_NAMESPACES[s] for s in sources if s in PRIMARY_NAMESPACES), source)
            return f"{namespace}:{value.strip()}" if value.strip() and namespace else ""
        raise ValueError(f"unsupported identifier: {name}")

    def _identifier_owner(self, name: str, token: str) -> Optional[Paper]:
        """Current adopted fields own identities; stale caches cannot grant one."""
        for paper in self._papers.values():
            if self._identifier_token(name, getattr(paper, name), paper.source) == token:
                from .merge_v2 import resolve_alias
                return (resolve_alias(self, paper.key) or paper) if merge_v2_enabled() or paper.merge_budget else paper
        return None

    def adopt_identifiers(self, key: str, *, provenance: dict | None = None, completion=None,
                          **identifiers: str) -> dict[str, str]:
        """Arbitrate verified blank fills and reindex under the library write lock.

        Never overwrite an adopted identity, merge another row, or claim a
        registration reservation. The caller still owns its normal save boundary.
        Each supplied axis returns an adoption or an explicit refusal reason.
        """
        self._check_writer()
        from .claims_v2 import same_attempt
        from .merge_v2 import blocked
        from .state_v2 import Merge
        from .telemetry_v2 import acknowledge_identity_adoption, adoption_snapshot

        # Validate before any side effect, including correction evidence.
        for name, value in identifiers.items():
            self._identifier_token(name, value)
        with self._merge_execution_lock:
            paper = self.get(key)
            current = paper.reopen_claim if paper is not None else None
            # An already-fenced dispatched completion may settle during merge
            # drain. Its caller still validates lease, policy and budget; no
            # ordinary fill or absorbed alias inherits that authorization.
            completing = (completion is not None and current is not None
                          and current.key == key and current.phase == "dispatched"
                          and same_attempt(current, completion)
                          and paper.acquisition_state.merge == Merge.CANONICAL
                          and not paper.acquisition_state.alias_key)
            if paper is None or (blocked(paper) and not completing):
                return {name: "refused-missing" if paper is None else "refused-blocked"
                        for name in identifiers}
            self._check_writer()
            self.corrections.stage_fields(paper, identifiers, operation="verified_identifiers",
                                          provenance=provenance)
            before = adoption_snapshot(paper)
            outcomes, applied = {}, {}
            for name, value in identifiers.items():
                token = self._identifier_token(name, value, paper.source)
                current = self._identifier_token(name, getattr(paper, name), paper.source)
                reserved = any(claim[:2] == (name, token) and claim[2] != key
                               for claim in self._identity_reservations)
                owner = self._identifier_owner(name, token) if token else None
                if not token:
                    outcome = "refused-empty"
                elif reserved:
                    outcome = "refused-reserved"
                elif owner is not None and owner.key != key:
                    outcome = "refused-held-by-other"
                    self.corrections.stage_identity(paper, {name: value}, owner,
                        provenance={"operation": "verified_identifiers", **(provenance or {})})
                elif current == token:
                    outcome = "already-mine"
                elif getattr(paper, name):
                    outcome = "refused-existing"
                else:
                    applied[name] = value
                    outcome = "adopted"
                outcomes[name] = outcome
            # Finish fallible correction recording before changing any field.
            for name, value in applied.items():
                setattr(paper, name, value)
            self._reindex(paper)
            acknowledge_identity_adoption(paper, before, applied)
            return outcomes

    def _fill_verified_identifiers(self, paper: Paper, *, doi: str = "",
                                   arxiv_id: str = "", provenance: dict | None = None,
                                   completion=None) -> dict[str, str]:
        """Compatibility result for existing verified-PDF and reopen callers."""
        incoming = {name: value for name, value in (("doi", doi), ("arxiv_id", arxiv_id)) if value}
        outcomes = self.adopt_identifiers(paper.key, provenance=provenance, completion=completion, **incoming)
        return {name: incoming[name] for name, result in outcomes.items() if result == "adopted"}

    def export_bibtex(self, keys: Optional[list[str]] = None) -> tuple[str, list[str]]:
        """Render BibTeX for a subset (or the whole library if keys is None).

        Returns (bib_text, missing_keys)."""
        if keys is None:
            clean = [p for p in self._papers.values() if p.domain_status is None]
            return ("\n\n".join(p.to_bibtex() for p in clean) + ("\n" if clean else ""), [])
        ordered: list[Paper] = []
        missing: list[str] = []
        for k in keys:
            p = self._papers.get(k)
            if p is None:
                missing.append(k)
            else:
                ordered.append(p)
        text = "\n\n".join(p.to_bibtex() for p in ordered) + ("\n" if ordered else "")
        return text, missing

    def _merge(self, existing: Paper, incoming: dict, *, provenance: dict | None = None) -> Paper:
        """Fill-blanks merge of ``incoming`` INTO ``existing``, IN PLACE — the
        SAME object identity is returned, never a fresh Paper.

        Why in place: a download/extract worker may be holding this exact Paper
        in a to_thread() call, about to write pdf_path / download_status / md_path
        onto it. The old code rebuilt a new Paper(**merged) and the callers did
        ``_papers[key] = new``; the worker then mutated the now-orphaned OLD
        object and its later save() persisted the NEW object, silently dropping
        the worker's writes (lost-update race, verified drill finding). Mutating
        in place keeps the worker's object and the stored object identical, so
        both sides' field writes land. Fill-blanks semantics are unchanged:
        non-empty incoming only fills a blank existing field (numeric upgrade for
        citation_count, latch is_review)."""
        self._check_writer()
        self.corrections.stage_fields(existing, incoming, provenance=provenance)
        for field, value in merge_inspire_fields(existing.model_dump(), incoming).items():
            setattr(existing, field, value)
        observations = incoming.get("source_identities")
        if observations:
            # Metadata-only fills must not overwrite a concurrent admission's observations.
            existing.source_identities = merge_source_identities(
                existing.source_identities, observations)
        state_observation = incoming.get("terminal_state")
        if state_observation:
            current_state = existing.unquiesced_state
            if isinstance(state_observation, dict):
                # Partial observations do not erase unobserved fields.
                fields = asdict(current_state)
                grant_observation = state_observation.get("override")
                if isinstance(grant_observation, dict) and current_state.override is not None:
                    state_observation = {**state_observation,
                                         "override": {**asdict(current_state.override), **grant_observation}}
                state_observation = {**fields, **state_observation}
            observed = Paper(key=existing.key, terminal_state=state_observation).terminal_state
            existing.terminal_state = retain_observed_restrictions(current_state, observed)
        for field, value in incoming.items():
            if field in {"inspire_record_id", "inspire_document_urls", "source_identities", "terminal_state",
                         "reopen_claim", "merge_intent", "merge_budget", "extraction_completion_id"}:
                continue  # source identity and links are merged together above
            if field == "key" or field not in Paper.model_fields:
                continue                       # never reassign the cite key
            if value in (None, "", [], 0):
                continue
            current = getattr(existing, field, None)
            if field == "citation_count" and isinstance(value, int):
                if value > int(current or 0):
                    existing.citation_count = value
            elif field == "is_review" and value:
                existing.is_review = True
            elif not current:
                setattr(existing, field, value)
        return existing

    def _find_preprint_journal_dup(self, paper_data: dict) -> Optional[Paper]:
        """D7: find an existing paper that is the preprint↔journal twin of
        ``paper_data``. Returns the existing Paper to merge into, or None.

        Criteria:
        - The new paper has a DOI XOR an arxiv_id (exactly one), AND the
          candidate existing paper has the *other* (so it's a real
          cross-identifier check, not a same-type retry).
        - Title similarity ≥ 0.85 AND author last-name Jaccard ≥ 0.7 AND
          |year diff| ≤ 2 → auto-merge.
        - Anything below that bar (borderline title sim / Jaccard) is
          conservatively SKIPPED: false-positive merges are data loss, so a
          residual duplicate is the safer outcome. (An earlier LLM-judged
          borderline path was removed — it was only ever reached via an
          ``llm=`` arg no caller passes, and ran a blocking sync LLM call on
          the event loop under the library write lock.)

        The lookup is O(N) over the library. For 850 papers this is fine
        (a few ms); if the library grows past 10K we'd want an index.
        """
        new_doi = (paper_data.get("doi") or "").strip()
        new_arxiv = _normalize_arxiv(paper_data.get("arxiv_id") or "")
        new_title = paper_data.get("title") or ""
        new_authors = paper_data.get("authors") or []
        new_year = paper_data.get("year")
        if not new_title or not new_authors:
            return None
        # Need exactly one of (DOI, arxiv_id) to make the cross-identifier check
        # meaningful — same-type fuzzy match would be caught by step-3 exact-title.
        has_doi = bool(new_doi)
        has_arxiv = bool(new_arxiv)
        if has_doi == has_arxiv:  # both or neither
            return None

        for cand in self._papers.values():
            if (merge_v2_enabled() or cand.merge_budget) and cand.acquisition_state.alias_key:
                continue
            cand_doi = (cand.doi or "").strip()
            cand_arxiv = _normalize_arxiv(cand.arxiv_id or "")
            # Candidate must have the OPPOSITE identifier from the new paper
            if has_doi and not cand_arxiv:
                continue
            if has_doi and cand_doi:  # candidate already has DOI; same-DOI was step 1
                continue
            if has_arxiv and not cand_doi:
                continue
            if has_arxiv and cand_arxiv:  # candidate already has arxiv; same-arxiv was step 2
                continue

            # Year sanity (preprint → publication can lag 2 years)
            if new_year and cand.year:
                if abs(int(new_year) - int(cand.year)) > 2:
                    continue

            title_sim = _title_similarity(new_title, cand.title or "")
            author_jac = _author_lastname_jaccard(new_authors, cand.authors or [])

            if title_sim >= 0.85 and author_jac >= 0.7:
                return cand  # high-confidence auto-merge
            # else: below the auto-merge bar → conservatively skip (a
            # false-positive merge is data loss; a residual duplicate is safer).
        return None

    # ----- topics -----

    def write_topic(self, slug: str, payload: dict) -> Path:
        """Persist a topic record (which keys belong to a search)."""
        self._check_writer()
        path = self.topics_dir / f"{slug}.json"
        _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
        return path

    # ----- manifest log -----

    def log(self, event: dict) -> None:
        """Append a manifest event as JSONL.

        PDF tier outcomes carry ``key`` and ``source``: ``download_miss``
        means attempted without a PDF; ``download_skip`` adds ``reason``
        for an unmet prerequisite, without a network attempt. Reconcile
        counts paper routes from status/disk facts, not these tier events.
        """
        self._check_writer()
        from papervault.config import stoprules_v2_enabled
        if stoprules_v2_enabled() and hasattr(self, "_telemetry_live"):
            # Preserve staged observations in the live manifest, with an
            # explicit stage marker; they are not live publication evidence.
            self._telemetry_live.log({**enrich(event, self), "staged": True})
            return
        line = json.dumps(enrich(event, self), ensure_ascii=False)
        with manifest_lock, self.manifest_path.open("a") as f:
            f.write(line + "\n")
        from .telemetry_v2 import observe_publication
        observe_publication(event, self)

    # ----- file paths -----

    def pdf_path(self, key: str) -> Path:
        return self._artifact_path(key, "pdf", self.pdfs_dir / f"{key}.pdf")

    def md_path(self, key: str) -> Path:
        return self._artifact_path(key, "md", self.md_dir / f"{key}.md")

    def txt_path(self, key: str) -> Path:
        return self._artifact_path(key, "txt", self.txt_dir / f"{key}.txt")

    def _artifact_path(self, key: str, kind: str, default: Path) -> Path:
        """Merged rows may retain an absorbed artifact within this same vault."""
        paper = self.get(key)
        if paper is None or paper.merge_budget is None or _is_nonempty_file(default):
            return default
        stored = getattr(paper, kind + "_path")
        if stored:
            candidate = self.root / stored
            if candidate.resolve().is_relative_to(self.root.resolve()) and _is_nonempty_file(candidate):
                return candidate
        return default

    def has_pdf(self, key: str) -> bool:
        return _is_nonempty_file(self.pdf_path(key))

    def has_extract(self, key: str, fmt: str) -> bool:
        path = self.md_path(key) if fmt == "md" else self.txt_path(key)
        return _is_nonempty_file(path)

    def md_source(self, key: str) -> Optional[str]:
        """Return the YAML frontmatter ``source:`` value of {key}.md, or
        None if the file is missing or has no frontmatter. Used to detect
        firecrawl-sourced (text-only) extracts so the marker upgrade path
        and idempotent re-entry work correctly."""
        p = self.md_path(key)
        if not p.is_file():
            return None
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:500]
        except Exception:
            return None
        if not head.startswith("---\n"):
            return None
        end = head.find("\n---\n", 4)
        if end < 0:
            return None
        for line in head[4:end].splitlines():
            s = line.strip()
            if s.startswith("source:"):
                return s.split(":", 1)[1].strip()
        return None
