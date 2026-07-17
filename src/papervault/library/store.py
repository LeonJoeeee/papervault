"""On-disk paper library: load / save / dedupe / key allocation."""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional

from .models import DOWNLOAD_STATUS_PENDING, Paper, base_key, normalize_title


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


def _atomic_write(path: Path, text: str) -> None:
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
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    tmp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename into it is durable (D14). Best-effort."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


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

    def __init__(self, root: Optional[str | Path] = None):
        # Vault root = the library's on-disk store. Resolved from papervault.config
        # (PAPERVAULT_VAULT, legacy PAPER_LIBRARY_PATH honored) unless passed explicitly.
        from papervault import config
        self.root = Path(root or config.VAULT_PATH)
        self.bib_path = self.root / "library.bib"
        self.index_path = self.root / "index.json"
        # D14: the last known-good index, rotated in on every successful save.
        self.index_bak_path = self.root / "index.json.bak"
        self.topics_dir = self.root / "topics"
        self.pdfs_dir = self.root / "pdfs"
        self.md_dir = self.root / "extracts" / "md"
        self.txt_dir = self.root / "extracts" / "txt"
        self.manifest_path = self.root / "manifest.log"

        for d in (self.topics_dir, self.pdfs_dir, self.md_dir, self.txt_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._papers: dict[str, Paper] = {}
        self._by_doi: dict[str, str] = {}
        self._by_arxiv: dict[str, str] = {}
        self._by_title: dict[str, str] = {}
        self._load()

    # ----- load / save -----

    def _load(self) -> None:
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
            self._reindex(paper)
        if bad:
            self.log({"event": "index_load_skipped_records", "count": bad})

    def _reindex(self, p: Paper) -> None:
        if p.doi:
            self._by_doi[p.doi.lower()] = p.key
        if p.arxiv_id:
            self._by_arxiv[_normalize_arxiv(p.arxiv_id)] = p.key
        nt = normalize_title(p.title)
        if nt:
            self._by_title[nt] = p.key

    def _save_index(self) -> None:
        # D14: rotate the current on-disk index into ``.bak`` BEFORE overwriting
        # it, so a known-good copy survives a crash mid-write of the new index.
        # We only rotate a copy that actually parses — never overwrite a good
        # backup with a freshly-corrupted primary.
        if self.index_path.exists() and self._read_index_file(self.index_path) is not None:
            try:
                _atomic_write(self.index_bak_path, self.index_path.read_text())
            except OSError:
                pass  # backup is best-effort; the durable primary write follows
        data = {"version": 1, "papers": {k: p.model_dump() for k, p in self._papers.items()}}
        _atomic_write(self.index_path, json.dumps(data, ensure_ascii=False, indent=2))

    def _save_bib(self) -> None:
        # Pass `library=self` so Paper.to_bibtex emits `file = {...}` fields
        # pointing to local PDFs / extracts for Zotero auto-attach on import.
        # Domain-quarantined papers are excluded — library.bib is a clean view
        # (no off-domain \cite keys leaking into the legal pool).
        entries = [p.to_bibtex(library=self) for p in self._papers.values()
                   if p.domain_status is None]
        _atomic_write(self.bib_path, "\n\n".join(entries) + "\n" if entries else "")

    def save(self, *, lock_timeout: float = 30.0) -> None:
        """Persist index + BibTeX to disk under a process-level write lock.

        The lock prevents two MCP server instances (or pipeline + MCP) from
        racing on index.json. Reads are unsynchronized — they always see a
        fully-written file because writes are atomic (.tmp → rename).
        """
        from filelock import FileLock, Timeout
        lock_path = self.root / ".write.lock"
        try:
            with FileLock(str(lock_path), timeout=lock_timeout):
                self._save_index()
                self._save_bib()
        except Timeout as exc:
            raise RuntimeError(f"library write lock timed out after {lock_timeout}s") from exc

    # ----- lookup -----

    def find(self, *, doi: str = "", arxiv_id: str = "",
             title: str = "") -> Optional[Paper]:
        """Look a paper up by any identifier. Case-insensitive."""
        if doi:
            k = self._by_doi.get(doi.lower())
            if k:
                return self._papers[k]
        if arxiv_id:
            k = self._by_arxiv.get(_normalize_arxiv(arxiv_id))
            if k:
                return self._papers[k]
        if title:
            k = self._by_title.get(normalize_title(title))
            if k:
                return self._papers[k]
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
        return [p for p in self._papers.values() if p.domain_status is None]

    def keys(self) -> list[str]:
        return list(self._papers.keys())

    def set_domain_status(self, key: str, status: Optional[str],
                          tier: Optional[str] = None) -> bool:
        """Mark (or clear) a paper's domain membership. ``status=None`` clears the
        quarantine (back to in-domain). Append-only-safe: updates metadata only,
        never deletes. Does NOT call ``save()`` — caller batches saves."""
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
        existing = self.find(
            doi=paper_data.get("doi", "") or "",
            arxiv_id=paper_data.get("arxiv_id", "") or "",
            title=paper_data.get("title", "") or "",
        )
        if existing:
            updated = self._merge(existing, paper_data)
            self._papers[existing.key] = updated
            self._reindex(updated)
            return updated, False

        # Step 4: fuzzy preprint ↔ journal merge (D7).
        fuzzy_existing = self._find_preprint_journal_dup(paper_data)
        if fuzzy_existing is not None:
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
        paper = Paper(**{k: v for k, v in paper_data.items()
                         if k in Paper.model_fields})
        self._papers[key] = paper
        self._reindex(paper)
        return paper, True

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
        existing = self._papers.get(key)
        if existing is None:
            return {"key": key, "removed": False, "reason": "not in library"}

        # Drop from indexes
        self._papers.pop(key, None)
        if existing.doi:
            self._by_doi.pop(existing.doi.lower(), None)
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

    def enrich(self, key: str, fields: dict, *, enriched_at: str) -> bool:
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
        p = self._papers.get(key)
        if p is None:
            return False
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
                              fields and the stub is purged. ``detail`` is the
                              surviving DOI-holder's key.
          * ``"has_doi"``   — the row already carries a non-empty DOI; NEVER
                              overwritten. ``detail`` is the existing DOI.
          * ``"no_paper"``  — no such key.

        Does NOT call ``save()`` — the caller batches the save under the write
        lock. The ``_by_doi`` write here is the load-bearing difference from
        the ``_try_arxiv_by_title`` precedent's raw ``paper.arxiv_id = ...``
        assignment (which left its index stale).
        """
        p = self._papers.get(key)
        if p is None:
            return ("no_paper", "")
        if (p.doi or "").strip():
            return ("has_doi", p.doi)  # never overwrite a real DOI
        doi = (doi or "").strip()
        if not doi:
            return ("no_paper", "empty doi")

        holder_key = self._by_doi.get(doi.lower())
        if holder_key is not None and holder_key != key:
            # A twin already owns this DOI. Merge the stub's blank-fill fields
            # into the holder (journal-DOI side wins identity), then purge the
            # stub. We DO NOT write the duplicate DOI onto the stub — that would
            # leave _by_doi[doi] still pointing at the holder while a second row
            # silently carries the same DOI.
            holder = self._papers.get(holder_key)
            if holder is not None:
                merged = self._merge(holder, p.model_dump())
                self._papers[holder_key] = merged
                self._reindex(merged)
            self.log({
                "event": "doi_resolve_collision_merged",
                "stub_key": key,
                "holder_key": holder_key,
                "doi": doi,
            })
            self.purge(key, reason="doi_resolve_twin_merged")
            return ("collision", holder_key)

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

    @staticmethod
    def _merge(existing: Paper, incoming: dict) -> Paper:
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
        for field, value in incoming.items():
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
        path = self.topics_dir / f"{slug}.json"
        _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
        return path

    # ----- manifest log -----

    def log(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self.manifest_path.open("a") as f:
            f.write(line + "\n")

    # ----- file paths -----

    def pdf_path(self, key: str) -> Path:
        return self.pdfs_dir / f"{key}.pdf"

    def md_path(self, key: str) -> Path:
        return self.md_dir / f"{key}.md"

    def txt_path(self, key: str) -> Path:
        return self.txt_dir / f"{key}.txt"

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
