"""Flag-gated API send boundary for the first politeness-v2 migration.

Service-wide scopes deliberately combine credentials, endpoints and this
process's egress IPs. Publisher content/browser sends are migrated separately.
Configuration is pinned at first enabled send; changing it cannot reset debits.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

import requests
from urllib3.exceptions import ReadTimeoutError

from papervault.config import polite_v2_enabled, stoprules_v2_enabled
from .download_telemetry import emit, span
from .telemetry_v2 import request_scope
from .politeness_v2 import AdmissionDenied, Envelope, Ledger, Policy, Scope, Window


# Wiley's canonical parallel windows are pinned by issue #221 and accepted
# spec D-D. The old wiley.py docstring names a limit without numeric pacing.
# S2: services/batch.py module comment, "free-tier traffic at 1 r/s".
# arXiv: sources/arxiv.py comment, "1 request per 3 seconds per IP".
DOCUMENTED = {
    "wiley": (Window(3, 1), Window(60, 600)),
    "semantic_scholar": (Window(1, 1),),
    "arxiv": (Window(1, 3),),
}
DEFAULTS = {
    Scope("wiley"): Policy((Window(1, 1), Window(30, 600))),
    Scope("semantic_scholar"): Policy((Window(1, 2),)),
    Scope("arxiv"): Policy(DOCUMENTED["arxiv"]),
}
_HOSTS = {
    "api.wiley.com": "wiley", "api.crossref.org": "crossref",
    "api.unpaywall.org": "unpaywall", "api.semanticscholar.org": "semantic_scholar",
    "api.openalex.org": "openalex", "export.arxiv.org": "arxiv",
}
_UNDOCUMENTED = {"crossref", "unpaywall", "openalex"}
_lock = threading.RLock()
_ledger: Ledger | None = None
_configuration: str | None = None
_unavailable: set[Scope] = set()


class PolitenessDenied(requests.ConnectionError):
    """No request emitted: scope disabled, full, or cooling down."""


class PolitenessUnavailable(requests.ConnectionError):
    """Accounting/configuration failed; the scope must not send."""


def get_ledger() -> Ledger:
    global _ledger, _configuration
    with _lock:
        configured = os.environ.get("PAPERVAULT_POLITE_ENVELOPES", "{}")
        if _ledger is not None:
            if configured != _configuration:
                raise ValueError("live envelope changes require a process restart")
            return _ledger
        values = json.loads(configured)
        if not isinstance(values, dict):
            raise ValueError("envelopes must be a service-keyed object")
        policies = dict(DEFAULTS)
        for service, value in values.items():
            if service not in _UNDOCUMENTED or not isinstance(value, dict):
                raise ValueError("envelopes configure undocumented API services only")
            if value.keys() - {"analogue", "initial", "hard_cap", "authorization"}:
                raise ValueError("unknown envelope configuration field")
            analogue = DOCUMENTED.get(value.get("analogue"), ())
            initial = tuple(Window(*w) for w in value.get("initial", []))
            hard_cap = tuple(Window(*w) for w in value.get("hard_cap", []))
            policies[Scope(service)] = Envelope(analogue, initial, hard_cap,
                                                value.get("authorization", ""))
        _ledger = Ledger(policies)
        _configuration = configured
        return _ledger


def _scope(target):
    url = target if isinstance(target, str) else target.url
    parsed = urlsplit(url)
    service = _HOSTS.get(parsed.hostname)
    # arXiv page/PDF traffic and its redirects belong to a later migration.
    if service == "arxiv" and not parsed.path.startswith("/api/"):
        return None
    return Scope(service) if service else None


@dataclass
class _Result:
    status: int | None = None
    timed_out: bool = False
    completed: bool = False
    capture: Callable | None = None


def _is_timeout(exc):
    # urllib3 can wrap a socket/read timeout in requests.ConnectionError.
    if isinstance(exc, (requests.Timeout, TimeoutError, ReadTimeoutError)):
        return True
    return any(_is_timeout(arg) for arg in exc.args if isinstance(arg, BaseException))


@contextmanager
def _account(scope):
    try:
        with _lock:
            if scope in _unavailable:
                raise ValueError("scope accounting is unavailable")
            ledger = get_ledger()
            ticket = ledger.note_sent(scope)
    except AdmissionDenied as exc:
        if stoprules_v2_enabled():
            emit("admission-denied", reason=exc.decision.reason)
        raise PolitenessDenied(str(exc)) from exc
    except Exception as exc:
        if stoprules_v2_enabled():
            emit("admission-denied", reason="accounting_unavailable")
        raise PolitenessUnavailable("politeness accounting unavailable; no request emitted") from exc
    result = _Result()
    def capture(response, **kwargs):
        result.status = response.status_code
        with _lock:
            try:
                ledger.note_response(ticket, status_code=result.status)
            except Exception as exc:
                _unavailable.add(scope)
                raise PolitenessUnavailable("politeness response accounting unavailable") from exc
        return response
    result.capture = capture
    try:
        yield result
    except BaseException as exc:
        result.timed_out = _is_timeout(exc)
        raise
    finally:
        with _lock:
            try:
                ledger.note_result(ticket, status_code=result.status,
                                   timed_out=result.timed_out, completed=result.completed)
            except Exception as exc:
                _unavailable.add(scope)
                raise PolitenessUnavailable("politeness result accounting unavailable") from exc


def polite_get(transport, url, **kwargs):
    """Flag off preserves transport arguments; flag on debits before GET."""
    if not polite_v2_enabled():
        return transport(url, **kwargs)
    scope = _scope(url)
    if scope is None:
        return transport(url, **kwargs)
    if kwargs.get("stream"):
        raise PolitenessUnavailable("streaming sends require polite_stream")
    with request_scope(scope), (span("request") if stoprules_v2_enabled() else nullcontext()), _account(scope) as result:
        hooks = dict(kwargs.get("hooks") or {})
        previous = hooks.get("response")
        if not previous:
            session = getattr(transport, "__self__", None)
            previous = getattr(session, "hooks", {}).get("response", [])
        previous = [previous] if callable(previous) else list(previous)
        hooks["response"] = [result.capture, *previous]
        kwargs = {**kwargs, "hooks": hooks}
        response = transport(url, **kwargs)
        if result.status is None:  # also support transports without response hooks
            result.capture(response)
        result.completed = True
        return response


@contextmanager
def polite_stream(transport, target, **kwargs):
    """Hold the scope permit through body consumption and response close."""
    scope = _scope(target) if polite_v2_enabled() else None
    if scope is None:
        with transport(target, **kwargs) as response:
            yield response
        return
    with request_scope(scope), (span("request") if stoprules_v2_enabled() else nullcontext()), _account(scope) as result:
        with transport(target, **kwargs) as response:
            result.capture(response)
            yield response
            # A closed raw stream can still be unread. Unknown-length raw
            # streams earn no ramp evidence unless requests consumed them.
            result.completed = (response._content_consumed
                                or response.raw is not None and response.raw.length_remaining == 0)
