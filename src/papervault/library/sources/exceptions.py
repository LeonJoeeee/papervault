"""Source-backend signalling exceptions (§3).

A leaf module (sibling of the per-backend wrappers) so that from WITHIN a
``sources/*.py`` module the import is ``from .exceptions import BackendDegraded``
(sibling) and from ``search.py`` (one level up) it is
``from .sources.exceptions import BackendDegraded``.
"""

from __future__ import annotations


class BackendDegraded(Exception):
    """Raised by a source on exhausted-retry / WAF / bounded-fail / auth-rejected
    (distinct from returning ``[]`` for a genuine 0-hit query).

    Deliberately derives DIRECTLY from ``Exception`` — NOT a subclass of
    ``requests.HTTPError`` — so a source loop's ``except requests.exceptions
    .HTTPError: return []`` does NOT swallow a ``BackendDegraded`` raised earlier
    in the same ``try``; it escapes intact to ``_fetch_one_backend``'s typed
    handler, which records the (term, backend) pair as DEGRADED.
    """
