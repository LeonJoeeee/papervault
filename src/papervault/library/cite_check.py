"""Validate that all \\cite-style keys in a LaTeX document resolve in the library."""

from __future__ import annotations

import re
from typing import Iterable

from .store import Library


_CITE_RE = re.compile(r"\\cite[a-z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}")


def extract_cite_keys(latex: str) -> list[str]:
    """Pull every key referenced by any \\cite-family command."""
    out: list[str] = []
    for match in _CITE_RE.findall(latex or ""):
        for raw in match.split(","):
            k = raw.strip()
            if k:
                out.append(k)
    return out


def check(latex: str, library: Library, *, allowed: Iterable[str] | None = None) -> dict:
    """Return a report: ok flag, list of dangling keys, and counts.

    `allowed` lets the pipeline restrict to a topic-specific subset; if None,
    the entire library is fair game.
    """
    cited = extract_cite_keys(latex)
    legal = set(allowed) if allowed is not None else set(library.keys())
    dangling = [k for k in cited if k not in legal]
    return {
        "ok": not dangling,
        "cited_count": len(cited),
        "unique_keys": sorted(set(cited)),
        "dangling": sorted(set(dangling)),
        "legal_count": len(legal),
    }
