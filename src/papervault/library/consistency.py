"""metadata <-> abstract <-> extract consistency predicate.

A record is CONSISTENT-AND-USABLE iff it has a usable abstract AND its stored
full-text extract (if any) actually corresponds to the same paper as the
metadata. This single predicate is reused by:

  * the existing-stock audit (papervault.library.cli / a sweep) — to find records to
    rescue or purge, and
  * the ingest gate (mcp/server.py Stage 3) — to refuse admitting an
    inconsistent record going forward.

The extract<->metadata check is DETERMINISTIC and fast, returning one of
MATCH / MISMATCH / UNCERTAIN. The discriminating signal is the FIRST-AUTHOR
surname: a wrong-but-same-topic full text (e.g. a stub titled "Galactic winds
driven by cosmic rays" whose stored text is a *different* same-topic paper by
other authors) has a high title-word overlap yet the metadata's first author is
absent — that lands in UNCERTAIN and is escalated to an LLM check by the caller,
never silently passed. A multi-article journal scan (the metadata paper IS in
the file, just not at offset 0) keeps both signals and passes as MATCH.
"""
from __future__ import annotations

from .fetch import _name_to_surname
from .models import normalize_title

# Statuses for the extract<->metadata relation.
NO_EXTRACT = "no_extract"   # the record has no stored full text at all
MATCH = "match"             # the extract is this paper (high confidence)
MISMATCH = "mismatch"       # the extract is a DIFFERENT paper (high confidence)
UNCERTAIN = "uncertain"     # signals conflict -> caller should escalate to LLM

_ABSTRACT_FLOOR = 40        # chars; shorter than this is not a usable abstract
_TITLE_STRONG = 0.60        # title-token overlap that counts as "topic present"
_MIN_SURNAME = 3            # surnames shorter than this are too common to trust
_TOP_CHARS = 1500           # the extract's OWN title/author block lives here


def has_usable_abstract(abstract: str | None, *, floor: int = _ABSTRACT_FLOOR) -> bool:
    """True iff the abstract is present and at least ``floor`` chars."""
    return len((abstract or "").strip()) >= floor


def _significant_title_tokens(title: str) -> set[str]:
    return {t for t in normalize_title(title or "").split() if len(t) >= 4}


def title_overlap(meta_title: str, head: str) -> float:
    """Fraction of the metadata title's significant tokens present in ``head``
    (order-insensitive). 0.0 when the title has no significant tokens."""
    toks = _significant_title_tokens(meta_title)
    if not toks:
        return 0.0
    htoks = set(normalize_title(head or "").split())
    return sum(1 for t in toks if t in htoks) / len(toks)


def first_author_in_head(meta_authors: list[str] | None, head: str) -> bool:
    """True iff the metadata first-author surname appears in ``head``. Returns
    False (i.e. "no signal") for missing / too-short surnames rather than
    guessing."""
    if not meta_authors:
        return False
    surname = _name_to_surname(meta_authors[0])
    if len(surname) < _MIN_SURNAME:
        return False
    return surname in normalize_title(head or "")


def extract_status(meta_title: str, meta_authors: list[str] | None,
                   extract_head: str) -> tuple[str, str]:
    """Deterministic verdict on whether the extract is THIS paper. Returns
    (status, reason).

    Only the TOP window (the paper's own title/author block) is inspected, NOT
    the whole head: a wrong same-topic paper often CITES the metadata's author
    deeper down (e.g. a 2021 paper citing the 1975 seminal work whose title this
    stub carries), and matching anywhere in the body would false-pass it. So a
    MATCH requires the title AND first author to be present AT THE TOP. Anything
    else is UNCERTAIN -> the caller escalates to an LLM, which is the only thing
    that declares a confirmed MISMATCH. We never declare MISMATCH deterministically
    because a real paper can sit below a multi-article journal scan's first page.
    """
    if not (extract_head or "").strip():
        return UNCERTAIN, "empty extract head"
    top = extract_head[:_TOP_CHARS]
    ov = title_overlap(meta_title, top)
    au = first_author_in_head(meta_authors, top)
    if ov >= _TITLE_STRONG and au:
        return MATCH, f"clean top match (title_overlap={ov:.2f}, first-author present)"
    return UNCERTAIN, f"not a clean top match (overlap={ov:.2f}, author_at_top={au}) -> LLM"


def record_verdict(*, abstract: str | None, title: str,
                   authors: list[str] | None,
                   extract_head: str | None) -> dict:
    """Top-level per-record consistency verdict.

    ``extract_head`` is None when the record has no stored full text (a
    metadata-only record is allowed, provided it has an abstract). Otherwise it
    is the first few-thousand chars of the stored extract.

    Returns a dict: has_abstract, extract_status, ok, needs_llm, reason.
    ``ok`` is the bottom line: a usable abstract AND (no extract OR a matching
    extract). ``needs_llm`` flags the UNCERTAIN extract cases the caller should
    resolve with an LLM before treating the record as ok or junk.
    """
    ha = has_usable_abstract(abstract)
    if extract_head is None:
        es, reason = NO_EXTRACT, "metadata-only (no stored extract)"
    else:
        es, reason = extract_status(title, authors, extract_head)
    ok = ha and es in (NO_EXTRACT, MATCH)
    return {
        "has_abstract": ha,
        "extract_status": es,
        "ok": ok,
        "needs_llm": es == UNCERTAIN,
        "reason": ("no usable abstract; " if not ha else "") + reason,
    }
