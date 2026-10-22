"""Safe, structured signalling for source failures and source observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
import math
from typing import Literal

import requests

Scope = Literal["item", "service"]
Retryability = Literal["permanent", "transient_with_delay", "transient_no_delay"]
DEFAULT_RETRY_AFTER_S = 5.0


class BackendDegraded(Exception):
    """An unsuccessful source fetch, distinct from a genuine zero-hit result.

    Existing positional-message callers remain valid. A missing/invalid delay
    for ``transient_with_delay`` means a five-second delay; permanent and
    transient-no-delay failures carry no delay. This is advice for a subsequent
    attempt, not permission for an adapter to retry without its existing bound.
    Messages are for local diagnosis only and are never copied into receipts.
    """

    def __init__(self, message: str, *, code: str = "backend_degraded",
                 scope: Scope = "service",
                 retryability: Retryability = "transient_with_delay",
                 retry_after_s: float | None = None):
        super().__init__(message)
        if scope not in ("item", "service"):
            raise ValueError("invalid source error scope")
        if retryability not in ("permanent", "transient_with_delay", "transient_no_delay"):
            raise ValueError("invalid source retryability")
        self.code = code
        self.scope = scope
        self.retryability = retryability
        self.retry_after_s = (
            _delay(retry_after_s) if retryability == "transient_with_delay" else None
        )


@dataclass(frozen=True)
class SourceGateOutcome:
    """One backend/term observation; contains no provider body or error text."""

    backend: str
    term: str
    status: Literal["success", "degraded", "skipped"]
    hit_count: int
    latency_ms: float
    term_idx: int = 0
    code: str | None = None
    scope: Scope | None = None
    retryability: Retryability | None = None
    retry_after_s: float | None = None


def _delay(value) -> float:
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return DEFAULT_RETRY_AFTER_S
    return value if math.isfinite(value) and value >= 0 else DEFAULT_RETRY_AFTER_S


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float:
    """Read Retry-After seconds or HTTP-date, using the default when absent/bad."""
    if value is None:
        return DEFAULT_RETRY_AFTER_S
    try:
        return _delay(float(value))
    except (ValueError, TypeError, OverflowError):
        pass
    try:
        deadline = parsedate_to_datetime(value)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return max(0.0, (deadline - (now or datetime.now(timezone.utc))).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return DEFAULT_RETRY_AFTER_S


def backend_degraded(error: Exception | None = None, *, response=None,
                     now: datetime | None = None) -> BackendDegraded:
    """Classify actual boundary failures without retaining response/error text."""
    if isinstance(error, BackendDegraded):
        return error
    if isinstance(error, (requests.Timeout, TimeoutError)):
        return BackendDegraded("Source timeout", code="timeout",
                               retryability="transient_no_delay")
    if isinstance(error, requests.ConnectionError):
        return BackendDegraded("Source transport failure", code="transport_error",
                               retryability="transient_no_delay")
    if response is None and isinstance(error, requests.HTTPError):
        response = error.response
    if response is not None:
        status = response.status_code
        if status == 429 or 500 <= status < 600:
            return BackendDegraded(
                "Source rate limited" if status == 429 else "Source unavailable",
                code="rate_limited" if status == 429 else "provider_unavailable",
                retry_after_s=retry_after_seconds(response.headers.get("Retry-After"), now=now),
            )
        if status in (401, 403):
            return BackendDegraded("Source authentication failed", code="authentication_failed",
                                   retryability="permanent")
        if status == 409:
            return BackendDegraded("Source conflict", code="request_conflict", scope="item",
                                   retryability="transient_no_delay")
        if 400 <= status < 500:
            return BackendDegraded("Source request rejected", code="invalid_request", scope="item",
                                   retryability="permanent")
    if isinstance(error, (ValueError, TypeError, AttributeError, KeyError, IndexError)):
        return BackendDegraded("Invalid source payload", code="provider_invalid_payload",
                               retryability="permanent")
    if isinstance(error, (requests.RequestException, OSError)):
        return BackendDegraded("Source transport failure", code="transport_error",
                               retryability="transient_no_delay")
    return BackendDegraded("Source backend failed", code="backend_error", retryability="permanent")


def source_errors(fn):
    """Normalize network and malformed-record errors at a source boundary."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BackendDegraded:
            raise
        except (requests.RequestException, ValueError, TypeError, AttributeError,
                KeyError, IndexError) as error:
            raise backend_degraded(error) from error
    return wrapped


def source_results(data, *path: str) -> list[dict]:
    """Validate a provider's result envelope, including the genuine empty list."""
    try:
        for key in path:
            data = data[key]
    except (KeyError, TypeError) as error:
        raise backend_degraded(ValueError("missing result envelope")) from error
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise backend_degraded(ValueError("invalid result records"))
    return data
