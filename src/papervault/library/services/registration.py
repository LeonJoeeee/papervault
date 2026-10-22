"""Separately admitted registration commands; search only submits bounded requests.

PG commits metadata, per-key audit and pending acquisition/export work together.
The in-process queue is deliberately lossy on overload/shutdown: every such loss is
traced, and it never changes a search response. Committed pending records remain
recoverable by the existing download queue/reconcile scans after a process exit.
The acquisition budget bounds this command's enqueue decisions; a deferred pending
record may be admitted later by those existing recovery paths.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from papervault.library.models import Paper, normalize_title
from papervault.library.sources.identity import normalized_source_identities, merge_source_identities
from papervault.library.sources.inspire import merge_inspire_fields, retained_inspire_fields
from papervault.library.state_v2 import Policy, Retrieval, TerminalState, admission, effective_policy
from papervault.library.store import _validate_paper_metadata
from papervault.library.services.bm25_search import paper_to_dict
from papervault.store.persistence import _overlay
from papervault.store.repository import AuditEvent, PendingJob, Record, payload_hash
from psycopg import sql
from . import concurrency
from .judge import INGEST_TIERS

logger = logging.getLogger(__name__)
MAX_CANDIDATES = 120  # external shortlist + externally vouched library observations
MAX_REQUEST_BYTES = 2 * 1024 * 1024
_METADATA = {"title", "authors", "year", "venue", "volume", "pages", "issue", "abstract",
             "doi", "arxiv_id", "paper_id", "url", "source", "citation_count", "is_review",
             "source_identities", "inspire_record_id", "inspire_document_urls", "_inspire_metadata",
             "publication_types"}
_PAPER_TYPES = {
    "journalarticle", "journal-article", "article", "review", "conference", "conferencepaper",
    "proceedings-article", "preprint", "posted-content", "letter", "lettersandcomments",
    "casereport", "clinicaltrial", "metaanalysis", "study", "dissertation", "thesis",
    "book-chapter", "booksection",
}
_NON_PAPER_TYPES = {
    "dataset", "software", "book", "monograph", "erratum", "retraction", "editorial", "news",
    "peer-review", "peerreview", "grant", "component", "paratext", "supplementary-materials",
    "supplementary-material",
}
_IDENTIFIERS = {"doi", "arxiv_id", "paper_id", "inspire_record_id"}


def metadata_is_paper(publication_types) -> bool | None:
    types = {str(t).strip().lower() for t in publication_types or []}
    if types & _NON_PAPER_TYPES:
        return False
    return True if types & _PAPER_TYPES else None


def registration_rejection(candidate: dict, *, known_identity: bool = False) -> str:
    """Pure eligibility, shared by serving's existing gate and command admission."""
    if candidate.get("_registration_known_key"):
        return ""  # rechecked against current identity/policy inside the command
    judgment = candidate.get("_registration") or {}
    if not judgment.get("ingest_ok") or judgment.get("tier") not in INGEST_TIERS:
        return "domain_unjudged_or_rejected"
    doi = (candidate.get("doi") or "").lower()
    if doi.startswith("10.5194/egusphere-egu") and doi[21:22].isdigit():
        return "conference_abstract"
    meta = metadata_is_paper(candidate.get("publication_types"))
    if not (meta if meta is not None else judgment.get("llm_is_paper")):
        return "non_paper"
    if not (candidate.get("doi") or candidate.get("arxiv_id")
            or retained_inspire_fields(candidate)["inspire_record_id"]
            or (candidate.get("abstract") or "").strip()):
        return "contentless"
    if known_identity:
        return ""  # legacy quality validation applies only to a newly allocated record
    ok, reason = _validate_paper_metadata(candidate)
    return "" if ok else reason


def candidate_identity(candidate: dict) -> str:
    """A stable external identity; never allocates an internal citation key."""
    doi = (candidate.get("doi") or "").strip().lower()
    if doi:
        return "doi:" + doi
    arxiv = (candidate.get("arxiv_id") or "").strip()
    if arxiv:
        return "arxiv:" + arxiv
    observations = normalized_source_identities(candidate, candidate.get("source") or "")
    for observation in observations:
        for namespace, values in observation["source_ids"].items():
            if values:
                return namespace + ":" + values[0]
    payload = [candidate.get("source") or "unknown", normalize_title(candidate.get("title") or "")]
    return "external:" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()


class ServingTrace:
    """Bounded operational events in the existing manifest, independent of catalog IO."""

    def __init__(self, library, trace_id: str | None = None):
        self.library = library
        self.trace_id = trace_id or uuid4().hex

    def record(self, stage: str, *, candidate: dict | None = None, **fields):
        event = {"event": "serving_trace", "trace_id": self.trace_id, "stage": stage,
                 "at": datetime.now(timezone.utc).isoformat()}
        if candidate is not None:
            event["candidate"] = candidate_identity(candidate)[:256]
        event.update({k: v[:256] if isinstance(v, str) else v for k, v in fields.items()
                      if v is None or isinstance(v, (str, bool, int, float))})
        try:
            self.library.log(event)
        except Exception:
            logger.warning("serving trace unavailable (%s)", stage)

    def record_intent(self, observation):
        self.record("intent", **{name: getattr(observation, name) for name in (
            "model", "provider", "latency_ms", "degraded", "code",
        )})

    def record_backend(self, outcome):
        self.record("retrieval", arm="external", **{
            name: getattr(outcome, name) for name in (
                "backend", "term_idx", "status", "hit_count", "latency_ms", "code",
                "scope", "retryability", "retry_after_s",
            )
        })


def _view(library, papers):
    """Reuse #267's current-field lookup without sharing mutable paper/index state."""
    view = copy.copy(library)
    view._papers = {key: paper.model_copy(deep=True) for key, paper in papers.items()}
    view._by_doi, view._by_arxiv, view._by_title = {}, {}, {}
    view._merge_execution_lock = threading.RLock()
    view._identity_reservations = set()
    for paper in view._papers.values():
        view._reindex(paper)
    return view


def serving_excluded(paper: Paper) -> bool:
    """Current quarantine and exclusion always fence serving, including old files."""
    state = paper.acquisition_state
    return bool(paper.domain_status is not None or state.quarantine or state.policy == Policy.EXCLUDED)


def _compatible(paper, candidate):
    from ..store import _normalize_arxiv, _normalize_doi

    if paper is None:
        return None
    left, right = _normalize_doi(paper.doi), _normalize_doi(candidate.get("doi") or "")
    if left and right:
        return paper if left == right else None
    left, right = _normalize_arxiv(paper.arxiv_id), _normalize_arxiv(candidate.get("arxiv_id") or "")
    if left and right and left != right:
        return None
    left_version = re.search(r"v\d+$", paper.arxiv_id, re.I)
    right_version = re.search(r"v\d+$", candidate.get("arxiv_id") or "", re.I)
    if (left_version and right_version
            and left_version.group().lower() != right_version.group().lower()):
        return None
    return paper


def _find(library, candidate):
    """Share preview/registration identity compatibility, including explicit aliases.

    find() may already return a canonical row whose DOI differs from an alias's
    observation. Validate that observation against the alias before following it.
    """
    from ..merge_v2 import resolve_alias
    from ..store import _normalize_arxiv, _normalize_doi

    aliases = [paper for paper in library._papers.values() if paper.acquisition_state.alias_key]
    for name, normalize in (("doi", _normalize_doi), ("arxiv_id", _normalize_arxiv),
                            ("title", normalize_title)):
        value = candidate.get(name) or ""
        if not value:
            continue
        if name != "title":
            for alias in aliases:
                if (normalize(value) == normalize(getattr(alias, name))
                        and _compatible(alias, candidate) is not None):
                    # An exact alias identifier can legitimately lead to a
                    # different canonical DOI, retaining DOI-first precedence.
                    return resolve_alias(library, alias.key)
        paper = _compatible(library.find(**{name: value}), candidate)
        if paper is not None:
            if paper.acquisition_state.alias_key:
                paper = _compatible(resolve_alias(library, paper.key), candidate)
            if paper is not None:
                return paper
    paper = _compatible(library._find_preprint_journal_dup(candidate), candidate)
    if paper is not None and paper.acquisition_state.alias_key:
        return _compatible(resolve_alias(library, paper.key), candidate)
    return paper


def _fill(paper: Paper, incoming: dict, *, identities=True):
    """Fill bibliographic blanks on an isolated object, preserving authoritative state.

    Conflicting observations remain in the registration audit; registration cannot
    perform a correction, promotion/merge campaign, or copy source-supplied policy.
    """
    inspire = merge_inspire_fields(paper.model_dump(), incoming)
    if not identities and inspire.get("inspire_record_id") != paper.inspire_record_id:
        inspire = {}
    for field, value in inspire.items():
        if identities or field not in _IDENTIFIERS:
            setattr(paper, field, value)
    paper.source_identities = merge_source_identities(paper.source_identities,
                                                      incoming.get("source_identities"))
    for name in _METADATA - {"source_identities", "inspire_record_id", "inspire_document_urls",
                             "_inspire_metadata"} - (set() if identities else _IDENTIFIERS):
        value = incoming.get(name)
        if value in (None, "", [], 0):
            continue
        current = getattr(paper, name)
        if name == "citation_count":
            paper.citation_count = max(current or 0, value)
        elif name == "is_review":
            paper.is_review = current or bool(value)
        elif not current:
            setattr(paper, name, copy.deepcopy(value))


def preview_candidates(library, candidates, *, canonical_keys=None):
    """Apply version-aware lookup/fill to isolated serving objects only.

    The first occurrence keeps its scoring position/snapshot; later aliases may
    enrich its response projection without taking another result slot. Fusion's
    request-local canonical mapping also applies to matches found by preview.
    """
    def preview_find(owner, candidate):
        # Title and preprint lookups may find a different explicit version.
        # Serving projections cannot turn that bibliographic match into a merge.
        paper = _find(owner, candidate)
        if paper is not None and canonical_keys and paper.key in canonical_keys:
            canonical_key = canonical_keys[paper.key]
            if canonical_key != paper.key:
                # Bibliographic deduplication is weaker than an explicit alias
                # identifier and cannot bridge a conflicting DOI or version.
                paper = _compatible(owner.get(canonical_key), candidate)
        return paper if paper is not None and not serving_excluded(paper) else None

    view, snapshots = _view(library, {}), {}
    for candidate in candidates:
        live = preview_find(library, candidate)
        paper = view.get(live.key) if live else None
        paper = paper or preview_find(view, candidate)
        incoming = {k: copy.deepcopy(v) for k, v in candidate.items() if k in _METADATA}
        incoming["source_identities"] = normalized_source_identities(candidate, candidate.get("source") or "")
        if paper is None:
            if live:
                paper = live.model_copy(deep=True)
            else:
                data = {k: v for k, v in incoming.items() if k in Paper.model_fields and v is not None}
                data.update(retained_inspire_fields(incoming))
                paper = Paper(key=candidate_identity(candidate), **data)
            view._papers[paper.key] = paper
        _fill(paper, incoming)
        view._reindex(paper)
        if paper.key not in snapshots:
            snapshots[paper.key] = dict(paper_to_dict(paper), source=paper.source, paper_id=paper.paper_id,
                                       _source_origin="library" if library.get(paper.key) else "external")
        if "external_rrf_score" in candidate:
            # Rank evidence belongs to this serving request, never to Paper or
            # registration. Aggregate before identity enrichment changes lookup.
            snapshot = snapshots[paper.key]
            snapshot["external_rrf_score"] = snapshot.get("external_rrf_score", 0) + candidate["external_rrf_score"]
            rank = candidate["best_external_rank"]
            snapshot["best_external_rank"] = min(snapshot.get("best_external_rank", rank), rank)
            if snapshot["_source_origin"] == "library":
                snapshot["folded_vouch"] = snapshot.get("folded_vouch", 0) + candidate.get("_external_hit_count", 1)
    return list(snapshots.values()), view._papers


class _RegistrationCancelled(asyncio.CancelledError):
    """Cancellation after the executor drained still carries its actual outcome."""

    def __init__(self, outcomes):
        self.outcomes = outcomes


class _RegistrationIdentityChanged(Exception):
    """Discard a staged snapshot when a live worker has claimed its identity."""


class RegistrationService:
    def __init__(self, library, download_queue, *, max_candidates=MAX_CANDIDATES,
                 max_downloads=MAX_CANDIDATES):
        if not 0 <= max_candidates <= MAX_CANDIDATES or max_downloads < 0:
            raise ValueError("registration budgets must be bounded and nonnegative")
        self.library, self.download_queue = library, download_queue
        self.max_candidates, self.max_downloads = max_candidates, max_downloads

    async def register_candidates(self, candidates, *, actor: str, reason: str):
        if not actor.strip() or not reason.strip():
            raise ValueError("registration requires actor and reason")
        # Cancellation must not release writer ownership while the executor still
        # commits/publishes. Slow PG/disk work never runs on the serving event loop.
        async with concurrency.lib_write_lock:
            task = asyncio.create_task(asyncio.to_thread(self._register, candidates,
                                                         actor=actor, reason=reason))
            cancelled = False
            while True:
                try:
                    outcomes = await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    logger.exception("registration command failed (%s)", reason)
                    outcomes = [{"identity": candidate_identity(c), "status": "failed",
                                 "reason": "command_failed", "key": None, "download": "not_requested"}
                                for c in candidates]
                    break
            for outcome in outcomes:
                if outcome["download"] == "admitted":
                    try:
                        self.download_queue.add(outcome["key"])
                    except Exception:
                        # The committed pending row is recoverable; do not mislabel
                        # a successful registration as failed after its commit.
                        logger.exception("registration download wakeup failed for %s", outcome["key"])
            if cancelled:
                raise _RegistrationCancelled(outcomes)
            return outcomes

    def _register(self, candidates, *, actor, reason):
        lib = self.library
        lib._check_writer()
        backend = lib._catalog_backend
        with lib._catalog_save_lock.acquire(timeout=30) if backend else nullcontext():
            replace_json_snapshot = False
            while True:
                try:
                    if backend:
                        return self._register_postgres(candidates, actor=actor, reason=reason)
                    return self._register_json(candidates, actor=actor, reason=reason,
                                               replace_snapshot=replace_json_snapshot)
                except _RegistrationIdentityChanged:
                    # A verified fill won during staging IO. PG rolls back the
                    # entire stale batch; JSON replaces its isolated snapshot.
                    # Re-stage against the winner, never publish a second key.
                    replace_json_snapshot = True
                    continue

    def _check_publication(self, changed):
        """Check only identities actually adopted by the staged records."""
        for key, outcome in changed.items():
            for name, token in outcome["identifiers"].items():
                owner = self.library._identifier_owner(name, token)
                if owner is not None and owner.key != key:
                    raise _RegistrationIdentityChanged

    @contextmanager
    def _reserve_publication(self, changed):
        """Fence verified fills through PG commit without blocking search reads.

        New keys are not visible in the live working set before commit. Their
        claims therefore live in the reservation set, independently of indexes.
        """
        lib, reservations = self.library, set()
        try:
            with lib._merge_execution_lock:
                self._check_publication(changed)
                for key, outcome in changed.items():
                    for name, token in outcome["identifiers"].items():
                        claim = (name, token, key, object())
                        reservations.add(claim)
                        lib._identity_reservations.add(claim)
            yield
        finally:
            with lib._merge_execution_lock:
                lib._identity_reservations.difference_update(reservations)

    def _register_postgres(self, candidates, *, actor, reason):
        lib = self.library
        backend = lib._catalog_backend
        with backend.repository() as repo, ExitStack() as publication:
            event = AuditEvent(actor, "registration", {"command_id": reason})
            with repo._write(event):
                records = {r.key: r for r in repo.all()}
                papers = {key: Paper.model_validate(r.payload) for key, r in records.items()}
                view = _view(lib, papers)

                # Resolve routing keys, then reload their authoritative rows at
                # acceptance before applying any candidate metadata. PG's write
                # transaction serializes other catalog writers. Do IO outside
                # the identity lock so even a stalled reload cannot block reads.
                refreshed = set()
                for candidate in candidates:
                    existing = _find(view, candidate)
                    live = _find(lib, candidate)
                    existing = existing or (view.get(live.key) if live else None)
                    if existing is None or existing.key in refreshed:
                        continue
                    row = repo.get(existing.key)
                    if row is None:
                        raise _RegistrationIdentityChanged
                    records[row.key] = row
                    paper = Paper.model_validate(row.payload)
                    view._papers[row.key] = paper
                    view._reindex(paper)
                    refreshed.add(row.key)

                with lib._merge_execution_lock:
                    captured = {key: p.model_dump(mode="json") for key, p in lib._papers.items()}
                outcomes, changed = self._stage(view, candidates)
                accepted, revisions = [], {}
                recorded_observations = None
                for key, outcome in changed.items():
                    paper = view.get(key)
                    before = records.get(key)
                    normalized = Paper.model_validate(before.payload).model_dump(mode="json") if before else None
                    previous = backend.normalized.get(key)
                    if normalized is not None and previous is not None and key in captured:
                        # Retain unsaved worker metadata when authority has not
                        # changed that field since our last load/save. A remote
                        # change uses the fresh capture instead. Identities are
                        # always refreshed explicitly at publication below.
                        for name in Paper.model_fields.keys() - _IDENTIFIERS:
                            if normalized.get(name) == previous.get(name):
                                captured[key][name] = copy.deepcopy(previous.get(name))
                    payload = (_overlay(before.payload, normalized, paper.model_dump(mode="json"))
                               if before else paper.model_dump(mode="json"))
                    record = Record.from_payload(key, payload)
                    revision = before.revision if before else 0
                    jobs = backend._jobs([record], {key: revision})
                    if outcome["download"] == "admitted":
                        jobs += (PendingJob("registration-download:" + key, "download", key,
                                            detail={"command_id": reason}),)
                    audit = AuditEvent(actor, "registration", {
                        "command_id": reason, "reason": reason,
                        "outcome": outcome["status"], "download": outcome["download"],
                        "source_identities": payload.get("source_identities", []),
                        "observations": outcome["observations"],
                        "identity_conflicts": outcome.get("identity_conflicts", []),
                        "observations_sha256": payload_hash(outcome["observations"]),
                        "payload_sha256": record.payload_sha256,
                    }, pending_jobs=jobs)
                    did_change = repo.upsert(record, audit=audit, expected_revision=revision)
                    if not did_change:
                        # A no-op duplicate still has new evidence. A replay
                        # of the same command/observations cannot append twice.
                        if recorded_observations is None:
                            recorded_observations = {
                                (row["key"], row["observation"]) for row in repo.connection.execute(sql.SQL(
                                    "SELECT key, detail->>'observations_sha256' AS observation FROM {} "
                                    "WHERE actor=%s AND event='registration' AND detail->>'command_id'=%s"
                                ).format(repo._table("audit")), (actor, reason))}
                        observation = (key, audit.detail["observations_sha256"])
                        if observation not in recorded_observations:
                            repo._finish(key, replace(audit, pending_jobs=()))
                            recorded_observations.add(observation)
                    revisions[key] = revision + int(did_change)
                    accepted.append(record)
                # A collision rolls back records, audit and jobs together.
                # Keep reservations through commit and live publication, while
                # releasing the identity lock before commit IO begins.
                publication.enter_context(self._reserve_publication(changed))
            # Refresh identities from the committed row. Metadata acknowledgment
            # preserves worker edits made since capture, but it must never bless
            # an unsaved identifier by advancing its authoritative baseline.
            publication_conflicts = []
            with lib._merge_execution_lock:
                # Adopted identities are fenced through this publication. No
                # postcommit re-staging: the batch is already authoritative.
                for record in accepted:
                    current = lib.get(record.key)
                    if current is None:
                        current = Paper.model_validate(record.payload)
                        lib._papers[record.key] = current
                        captured[record.key] = current.model_dump(mode="json")
                    saved = Paper.model_validate(record.payload)
                    conflicts = self._local_identity_conflicts(view, saved, current)
                    if conflicts:
                        for outcome in outcomes:
                            if outcome["key"] == record.key:
                                outcome["reason"] = "identity_conflict"
                                known = outcome.setdefault("identity_conflicts", [])
                                known.extend(c for c in conflicts if c not in known)
                        # Includes identifier edits made during commit, after
                        # transactional audit was assembled; keep their evidence.
                        publication_conflicts.append((record.key, conflicts))
                    if (view._identifier_token("inspire_record_id", current.inspire_record_id)
                            != view._identifier_token("inspire_record_id", saved.inspire_record_id)):
                        current.inspire_document_urls = copy.deepcopy(saved.inspire_document_urls)
                    if (view._identifier_token("paper_id", saved.paper_id, current.source)
                            != view._identifier_token("paper_id", saved.paper_id, saved.source)):
                        current.source = saved.source
                    for name in _IDENTIFIERS:
                        setattr(current, name, copy.deepcopy(getattr(saved, name)))
                backend._accept(accepted, revisions, lib._papers, captured)
                for record in accepted:
                    lib._reindex(lib.get(record.key))
            for key, conflicts in publication_conflicts:
                try:
                    lib.log({"event": "registration_identity_conflict", "key": key,
                             "command_id": reason, "identity_conflicts": conflicts})
                except Exception:
                    logger.warning("registration conflict log unavailable for %s", key)
            return outcomes

    def _register_json(self, candidates, *, actor, reason, replace_snapshot=False):
        lib = self.library
        # JSON compatibility remains behind the same command, never serving.
        # Its legacy file format does not provide PG's transactional outbox.
        with lib._merge_execution_lock:
            view = _view(lib, lib._papers)
            captured = {key: p.model_dump(mode="json") for key, p in lib._papers.items()}
        outcomes, changed = self._stage(view, candidates)
        if changed or replace_snapshot:
            # A discarded snapshot may already be on disk. Even an empty retry
            # must replace it with the current live records and restrictions.
            # Write an isolated working set. No serving identity lock is held
            # across disk IO, and failed saves never publish staged objects.
            view.save(force=True)
        with lib._merge_execution_lock:
            self._check_publication(changed)
            for key in changed:
                current, staged = lib.get(key), view.get(key)
                if current is None:
                    lib._papers[key] = staged
                else:
                    now = current.model_dump(mode="json")
                    saved = staged.model_dump(mode="json")
                    for name in Paper.model_fields:
                        if (saved.get(name) != captured[key].get(name)
                                and now.get(name) == captured[key].get(name)):
                            setattr(current, name, copy.deepcopy(getattr(staged, name)))
                lib._reindex(lib.get(key))
        for key, outcome in changed.items():
            try:
                lib.log({"event": "registration", "actor": actor, "reason": reason,
                         "key": key, "outcome": outcome["status"], "download": outcome["download"],
                         "observations": outcome["observations"]})
            except Exception:
                logger.warning("registration log unavailable for %s", key)
        return outcomes

    @staticmethod
    def _local_identity_conflicts(view, paper, live):
        if live is None:
            return []
        return [{"field": name, "authority": getattr(paper, name),
                 "observed": getattr(live, name), "origin": "local"}
                for name in sorted(_IDENTIFIERS) if getattr(live, name)
                and view._identifier_token(name, getattr(live, name), live.source)
                != view._identifier_token(name, getattr(paper, name), paper.source)]

    def _adopt(self, view, paper, incoming, live, live_view, *, existing):
        """Apply staged identities only through the common adoption arbiter.

        Its working set is authoritative; live owners/reservations are additional
        conservative claims, never a replacement for the authoritative row.
        """
        claims = set(live_view._identity_reservations)
        for current in live_view._papers.values():
            for name in _IDENTIFIERS:
                if token := view._identifier_token(name, getattr(current, name), current.source):
                    claims.add((name, token, current.key, None))
        view._identity_reservations = claims
        identifiers, conflicts = {}, []
        if not paper.source:
            paper.source = incoming.get("source") or ""
        for name in sorted(_IDENTIFIERS):
            value = (retained_inspire_fields(incoming)[name]
                     if name == "inspire_record_id" else incoming.get(name))
            if not value:
                continue
            token = view._identifier_token(name, value, incoming.get("source") or paper.source)
            if name == "paper_id" and token != view._identifier_token(name, value, paper.source):
                continue  # another source's scalar ID remains an observation
            if existing and name in {"doi", "arxiv_id"} and not getattr(paper, name):
                if live is None or view._identifier_token(name, getattr(live, name)) != token:
                    continue  # observations alone cannot promote a cross-identifier
            if name == "inspire_record_id" and existing and not paper.inspire_record_id and not merge_inspire_fields(
                    paper.model_dump(), incoming):
                continue
            identifiers[name] = value
        decisions = view.adopt_identifiers(paper.key, provenance={"operation": "registration"},
                                            **identifiers) if identifiers else {}
        for name, decision in decisions.items():
            if decision not in {"adopted", "already-mine"}:
                conflicts.append({"field": name, "authority": getattr(paper, name),
                                  "observed": identifiers[name], "origin": "candidate",
                                  "decision": decision})
        return conflicts + self._local_identity_conflicts(view, paper, live)

    def _stage(self, view, candidates):
        # Admission/adoption uses an isolated live snapshot. In particular the
        # shared adoption API may record correction evidence on disk; it must
        # never perform that IO under the lock search uses for identity reads.
        # Final reservation checks arbitrate intervening live worker adoptions.
        with self.library._merge_execution_lock:
            live_view = _view(self.library, self.library._papers)
            live_view._identity_reservations = set(self.library._identity_reservations)
        outcomes, changed, downloads = [], {}, 0
        for index, candidate in enumerate(candidates):
            outcome = {"identity": candidate_identity(candidate), "status": "rejected",
                       "reason": "", "key": None, "download": "not_requested"}
            outcomes.append(outcome)
            existing = _find(view, candidate)
            # Current fields + verified-fill reservations (#267), including
            # metadata that workers have not yet saved to the catalog.
            live = _find(live_view, candidate)
            rejection = registration_rejection(candidate, known_identity=bool(existing or live))
            if rejection or index >= self.max_candidates:
                outcome["reason"] = rejection or "registration_budget"
                continue
            if live and existing and live.key != existing.key:
                outcome["reason"] = "identity_conflict"
                continue
            existing = existing or (view.get(live.key) if live else None)
            if existing is not None:
                # Authority fixes the record key even when a divergent local
                # identifier made candidate-based lookup miss its counterpart.
                live = live_view.get(existing.key)
            if live and existing is None:
                outcome.update(status="duplicate", key=live.key, reason="uncommitted_local_identity")
                continue
            if candidate.get("_registration_known_key") and existing is None:
                outcome["reason"] = "identity_no_longer_resolves"
                continue
            if existing and any(
                    p and (p.acquisition_state.quarantine or p.domain_status
                           or not effective_policy(p.acquisition_state, key=p.key, action="register"))
                    for p in (existing, live)):
                outcome.update(status="duplicate", key=existing.key, reason="registration_policy")
                continue
            # Detect conflicting exact axes instead of silently merging records.
            matches = {p.key for lookup in (view, live_view)
                       for name in ("doi", "arxiv_id") if candidate.get(name)
                       if (p := lookup.find(**{name: candidate[name]})) is not None}
            if len(matches) > 1:
                outcome["reason"] = "identity_conflict"
                continue
            if existing is None and matches:
                # The shared arbiter still reserves version-stripped arXiv IDs.
                # Refuse an incompatible version explicitly; never publish an
                # identifier-less stub or call the other version a duplicate.
                outcome["reason"] = "identity_conflict"
                continue
            if existing and any(
                    token and (owner := live_view._identifier_owner(name, token)) is not None
                    and owner.key != existing.key
                    for name in _IDENTIFIERS
                    for token in [view._identifier_token(name, getattr(existing, name), existing.source)]):
                # An authoritative owner and a different unsaved local owner
                # cannot be reconciled by retrying the same snapshot forever.
                outcome.update(key=existing.key, reason="identity_conflict")
                continue
            incoming = {k: copy.deepcopy(v) for k, v in candidate.items() if k in _METADATA}
            incoming["source_identities"] = normalized_source_identities(
                candidate, candidate.get("source") or "")
            observed = copy.deepcopy(incoming)
            observed["judgment"] = copy.deepcopy(candidate.get("_registration"))
            if existing:
                paper = existing
            else:
                authors = incoming.get("authors") or []
                # Reserve names present in either the authoritative snapshot or
                # the live working set, including another uncommitted ingress.
                reserved = set(view._papers) | set(live_view._papers)
                key_view = copy.copy(view)
                key_view._papers = dict.fromkeys(reserved)
                key = key_view.allocate_key(authors[0] if authors else "", incoming.get("year"))
                data = {k: v for k, v in incoming.items() if k in Paper.model_fields
                        and k not in _IDENTIFIERS | {"inspire_document_urls"} and v is not None}
                paper = Paper(key=key, **data, download_status="pending",
                              terminal_state=TerminalState(retrieval=Retrieval.ELIGIBLE, policy=Policy.CLEAR),
                              added_at=datetime.now(timezone.utc).isoformat())
                view._papers[key] = paper
            conflicts = self._adopt(view, paper, incoming, live, live_view, existing=bool(existing))
            _fill(paper, incoming, identities=False)
            if conflicts:
                observed["identity_conflicts"] = conflicts
                outcome["identity_conflicts"] = conflicts
            tier = (candidate.get("_registration") or {}).get("tier")
            if paper.domain_tier is None and tier in INGEST_TIERS:
                paper.domain_tier = tier
            view._reindex(paper)
            outcome.update(status="duplicate" if existing else "registered", key=paper.key,
                           reason="identity_conflict" if conflicts else "identity_exists" if existing else "admitted")
            if not existing and admission(paper.acquisition_state, identity_resolved=True,
                                           key=paper.key):
                if downloads < self.max_downloads:
                    outcome["download"] = "admitted"
                    downloads += 1
                else:
                    outcome["download"] = "budget_rejected"
            if paper.key not in changed:
                outcome["observations"] = []
                changed[paper.key] = outcome
            changed[paper.key]["observations"].append(observed)
            if changed[paper.key] is not outcome and conflicts:
                changed[paper.key].setdefault("identity_conflicts", []).extend(conflicts)
        for key, outcome in changed.items():
            paper = view.get(key)
            outcome["identifiers"] = {
                name: token for name in _IDENTIFIERS
                if (token := view._identifier_token(name, getattr(paper, name), paper.source))}
        return outcomes, changed


class RegistrationQueue:
    def __init__(self, service: RegistrationService, *, capacity: int = 32):
        if capacity < 0:
            raise ValueError("registration queue capacity must be nonnegative")
        self.service, self.capacity = service, capacity
        self._queue = asyncio.Queue(maxsize=max(1, capacity))
        self._workers = []
        self._started = False

    async def start(self):
        if not self._started:
            self._started = True
            self._workers = [asyncio.create_task(self._worker(), name="paper-registration")]

    def submit(self, candidates, *, actor, reason, trace: ServingTrace):
        if not candidates:
            return
        rejection = ""
        if not self._started:
            rejection = "registration_unavailable"
        elif not self.capacity or self._queue.full():
            rejection = "registration_queue_full"
        elif len(candidates) > MAX_CANDIDATES:
            rejection = "registration_request_limit"
        elif len(json.dumps(candidates).encode()) > MAX_REQUEST_BYTES:
            rejection = "registration_request_bytes"
        if rejection:
            for candidate in candidates[:MAX_CANDIDATES]:
                trace.record("registration", candidate=candidate, decision="rejected", reason=rejection)
            return
        snapshot = copy.deepcopy(candidates)
        self._queue.put_nowait((snapshot, actor, reason, trace))
        for candidate in snapshot:
            trace.record("registration", candidate=candidate, decision="submitted")

    async def _worker(self):
        while True:
            candidates, actor, reason, trace = await self._queue.get()
            try:
                for candidate in candidates:
                    trace.record("registration", candidate=candidate, decision="accepted")
                outcomes = await self.service.register_candidates(candidates, actor=actor, reason=reason)
                self._record_outcomes(trace, candidates, outcomes)
            except _RegistrationCancelled as exc:
                self._record_outcomes(trace, candidates, exc.outcomes)
                raise
            except asyncio.CancelledError:
                for candidate in candidates:
                    trace.record("registration", candidate=candidate, decision="failed", reason="shutdown")
                raise
            except Exception:
                logger.exception("registration command failed (%s)", trace.trace_id)
                for candidate in candidates:
                    trace.record("registration", candidate=candidate, decision="failed", reason="command_failed")
            finally:
                self._queue.task_done()

    @staticmethod
    def _record_outcomes(trace, candidates, outcomes):
        for candidate, outcome in zip(candidates, outcomes):
            trace.record("registration", candidate=candidate, decision=outcome["status"],
                         reason=outcome["reason"], key=outcome["key"], download=outcome["download"],
                         identity_conflicts=len(outcome.get("identity_conflicts", [])))
        trace.record("registration_summary", **{
            status: sum(o["status"] == status for o in outcomes)
            for status in ("registered", "duplicate", "rejected", "failed")})

    async def stop(self):
        self._started = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        while not self._queue.empty():
            candidates, _, _, trace = self._queue.get_nowait()
            for candidate in candidates:
                trace.record("registration", candidate=candidate, decision="rejected", reason="shutdown")
            self._queue.task_done()
