"""Persistent paper library: BibTeX + index + PDFs/extracts on disk.

Idempotent: re-adding a paper that's already known is a noop. Designed to
be shared across multiple downstream projects via PAPER_LIBRARY_PATH.
"""

from .models import (
    Paper,
    base_key,
    normalize_title,
    slugify_lastname,
)
from .store import Library

__all__ = [
    "Library",
    "Paper",
    "base_key",
    "normalize_title",
    "slugify_lastname",
]
