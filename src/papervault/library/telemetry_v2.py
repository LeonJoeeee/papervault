"""Additive A5 correlation and observed-only measurement.

Nulls identify unavailable evidence, not zero latency or a guessed attempt.
Identity version names the adopted v2 definition, independently of generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from papervault.config import stoprules_v2_enabled

from .claims_v2 import record_generation

IDENTITY_VERSION = 2
log = logging.getLogger(__name__)
_attempt = ContextVar("pipeline_attempt", default=None)
_scope = ContextVar("pipeline_request_scope", default=None)


@contextmanager
def attempt(claim):
    token = _attempt.set(claim)
    try:
        yield
    finally:
        _attempt.reset(token)


def scope_id(scope) -> str:
    # Opaque full-scope identity; never emit credentials, IPs, or endpoint URLs.
    values = (scope.service, scope.credential, scope.ip, scope.endpoint)
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


@contextmanager
def request_scope(scope):
    token = _scope.set(scope_id(scope))
    try:
        yield
    finally:
        _scope.reset(token)


def enrich_event(event, library):
    if not stoprules_v2_enabled():
        return event
    paper = library.get(event.get("key", ""))
    claim = _attempt.get()
    if claim is None or claim.key != event.get("key"):
        claim = paper.reopen_claim if paper else None
    fields = {"identity_version": IDENTITY_VERSION,
              "attempt_generation": claim.generation if claim else record_generation(paper) if paper else None,
              "lease_id": claim.lease_id if claim else None, "request_scope_id": _scope.get()}
    return {**fields, **event}


def record_censored(library, key: str, *, reason: str, wait_s=None, service_s=None,
                    drain_s=None, clock=time.monotonic):
    """Append immutable incomplete evidence; never estimate a terminal outcome."""
    if stoprules_v2_enabled():
        try:
            library.log({"event": "download_telemetry", "phase": "attempt", "mark": "censored",
                         "key": key, "censored": True, "reason": reason,
                         "observed_at": clock(), "wait_s": wait_s, "service_s": service_s,
                         "drain_s": drain_s})
        except Exception:
            log.warning("censored attempt telemetry unavailable")


def admission_denied(library, claim, reason):
    if stoprules_v2_enabled():
        try:
            with attempt(claim):
                library.log({"event": "download_telemetry", "phase": "admission-denied",
                             "mark": "point", "key": claim.key, "reason": reason})
        except Exception:
            log.warning("admission denial telemetry unavailable")


def duration_fields(phase, elapsed, *, incomplete=False):
    if not stoprules_v2_enabled():
        return {}
    kind = ("wait_s" if phase.endswith("wait") else "drain_s" if phase.endswith("drain")
            else "service_s" if phase in {"executor", "member", "request", "browser_call", "verification"}
            else None)
    fields = {kind: None if incomplete else elapsed} if kind else {}
    if incomplete:
        fields.update(censored=True, observed_elapsed_s=elapsed)
    return fields


def identity_snapshot(paper):
    from .identity_v2 import canonical_doi
    return {"doi": canonical_doi(paper.doi), "arxiv_id": paper.arxiv_id,
            "paper_id": paper.paper_id, "inspire_record_id": paper.inspire_record_id,
            "url": paper.url}


def publication_fields(before, paper, *, verified=False):
    if not stoprules_v2_enabled():
        return {}
    after = identity_snapshot(paper)
    return {"identity_matches": all(not value or after[name] == value for name, value in before.items()),
            "identity_verified": verified}


def observe_publication(event, library):
    """Retain only actionable evidence; rounds never scan growing manifest logs."""
    if not stoprules_v2_enabled():
        return
    if event.get("staged"):
        return
    if event.get("event") not in {"downloaded", "reopen_complete", "merge_committed"}:
        return
    paper = library.get(event.get("key", ""))
    claim = _attempt.get()
    if claim is None or claim.key != event.get("key"):
        claim = paper.reopen_claim if paper else None
    if paper is None:
        return
    if event.get("identity_verified") is True and event.get("identity_matches") is True:
        # A detached campaign stage is not a live published success.
        if event["event"] == "reopen_complete" or claim is None:
            try:
                library.log({"event": "download_telemetry", "key": paper.key,
                             "phase": "verified-success", "mark": "point"})
            except Exception:
                log.warning("verified success telemetry unavailable")
    if event.get("identity_matches") is False:
        from .stop_rules import Publication
        from .download_telemetry import manifest_lock
        with manifest_lock:
            pending = getattr(library, "_stoprule_publications", None)
            if pending is None:
                pending = library._stoprule_publications = {}
            pending[claim.lease_id if claim else paper.key] = Publication(
                claim, identity_matches=False, key=paper.key, generation=record_generation(paper))


def track_execution(work):
    """Count actual ordinary executor lifetime independently of the merge flag."""
    @wraps(work)
    def tracked(paper, library, *args, **kwargs):
        owner = getattr(library, "_telemetry_live", library)
        counted = stoprules_v2_enabled() and paper.reopen_claim is None
        if counted:
            with owner._merge_execution_lock:
                active = getattr(owner, "_stoprule_active", None)
                if active is None:
                    active = owner._stoprule_active = {}
                active[paper.key] = active.get(paper.key, 0) + 1
        try:
            return work(paper, library, *args, **kwargs)
        finally:
            if counted:
                with owner._merge_execution_lock:
                    remaining = active[paper.key] - 1
                    if remaining:
                        active[paper.key] = remaining
                    else:
                        active.pop(paper.key)
    return tracked
