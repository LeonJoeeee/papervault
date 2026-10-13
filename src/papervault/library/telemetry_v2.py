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
_identity_adoptions = ContextVar("pipeline_identity_adoptions", default=None)


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


def _identity_fields(paper):
    return {name: getattr(paper, name) for name in
            ("doi", "arxiv_id", "inspire_record_id", "paper_id", "source", "url")}


def identity_snapshot(paper, *, library=None):
    """Snapshot the adopted canonical identity, excluding secondary evidence."""
    from .identity_v2 import select_canonical
    from .state_v2 import Merge
    if library is not None and (paper.acquisition_state.merge == Merge.ALIAS
                                or paper.acquisition_state.alias_key):
        from .merge_v2 import resolve_alias
        paper = resolve_alias(library, paper.key)
        if paper is None:
            return None
    # Source observations can contain unadopted/conflicting evidence. Use only
    # the record's adopted fields, as the in-flight identity detector does.
    return select_canonical(_identity_fields(paper))


def begin_identity_measurement(paper, *, library):
    before = identity_snapshot(paper, library=library)
    adoptions = _identity_adoptions.get()
    if adoptions is not None:
        adoptions[paper.key] = {"initial": before, "recognized": {before}, "fields": _identity_fields(paper)}
    return before


def adoption_snapshot(paper):
    return identity_snapshot(paper) if _identity_adoptions.get() is not None else None


def acknowledge_identity_adoption(paper, before, fields):
    """Project acknowledged blank fills; never bless an intervening mismatch."""
    adoptions = _identity_adoptions.get()
    if adoptions is None or not fields:
        return
    from .download_telemetry import manifest_lock
    from .identity_v2 import select_canonical
    with manifest_lock:
        state = adoptions.get(paper.key)
        if state is None or before not in state["recognized"]:
            return
        expected = state["fields"]
        for name, value in fields.items():
            if name in expected and not expected[name]:
                expected[name] = value
        state["recognized"].add(select_canonical(expected))


def publication_fields(before, paper, *, verified=False, library=None):
    if not stoprules_v2_enabled():
        return {}
    adoptions = _identity_adoptions.get()
    state = adoptions.get(paper.key) if adoptions is not None else None
    if state is not None and before == state["initial"]:
        from .identity_v2 import select_canonical
        before = select_canonical(state["fields"])
    after = identity_snapshot(paper, library=library)
    return {"identity_matches": before is None or before == after,
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
        measurement_token = _identity_adoptions.set({}) if stoprules_v2_enabled() else None
        try:
            return work(paper, library, *args, **kwargs)
        finally:
            if measurement_token is not None:
                _identity_adoptions.reset(measurement_token)
            if counted:
                with owner._merge_execution_lock:
                    remaining = active[paper.key] - 1
                    if remaining:
                        active[paper.key] = remaining
                    else:
                        active.pop(paper.key)
    return tracked
