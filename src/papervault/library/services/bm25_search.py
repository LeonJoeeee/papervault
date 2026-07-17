"""Per-term BM25 ranking over the year-eligible library corpus + the shared
fair-share round-robin shortlister (search_papers Stage 1 + Stage 2.5, §2/§4).

V6 (2026-05-31): the old single-pool ``bm25_topk`` + ``dedupe_pool`` are GONE.
The library arm now builds ``BM25Okapi`` ONCE over the year-eligible corpus and
queries it once per search term (``bm25_per_term_ranklists``), emitting one
ranked node-ref list per term — byte-identical in shape to the external arm's
per-term ranklists. ONE ``round_robin`` (per-arm key-fn) fair-shares either arm's
``list[list[node]]`` down to a cap. ``_node_key`` is the inline identity helper
for EXTERNAL nodes (and the cross-arm fold collision test); LIBRARY nodes key on
their stable ``Paper.key``.

RRF consensus (2026-06-01): ``rrf_fuse`` + ``reorder_ranklists_by_rrf`` add true
Reciprocal Rank Fusion (``score = Σ 1/(RRF_K + rank0)``) as a calibration-free
cross-term / cross-backend CONSENSUS signal. RRF re-orders each per-term list
(consensus to the head) — the WITHIN-term half of the locked composition — while
``round_robin`` stays the BETWEEN-term diversity FLOOR (every term keeps a slot,
term-0 the A1 anchor leading each cycle). RRF reorders; round-robin guarantees.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable

from ..models import normalize_title

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokens. Ignores punctuation / unicode."""
    return _TOKEN_RE.findall((text or "").lower())


def _paper_text(p: dict) -> str:
    """Concatenate paper fields into a single string for the BM25 corpus."""
    authors = p.get("authors") or []
    if not isinstance(authors, list):
        authors = []
    return " ".join(filter(None, [
        p.get("title") or "",
        p.get("venue") or "",
        (p.get("abstract") or "")[:2000],   # cap abstract to 2K chars (named recall ceiling, §10)
        " ".join(str(a) for a in authors),
        str(p.get("year") or ""),
    ]))


def _node_key(node: dict) -> str:
    """Inline identity key for an EXTERNAL node (and the cross-arm fold collision
    test). Precedence: normalized DOI > version-stripped arXiv id
    (case-insensitive) > normalized title > a per-OBJECT ``_uid:<id>`` sentinel.

    The ``_uid`` fallback is per-object (NOT copy-tolerant): it gives an
    identity-LESS node a guaranteed-unique key so two all-empty-identity works
    stay DISTINCT (they ARE distinct works, must not collide). Computed at the
    POINT OF USE — never stamped onto a node.
    """
    doi = (node.get("doi") or "").strip().lower()         # _norm_doi
    if doi:
        return f"doi:{doi}"
    ax = node.get("arxiv_id") or ""
    ax = re.sub(r"[vV]\d+$", "", ax)                       # version-strip, case-insensitive
    if ax:
        return f"arx:{ax}"
    nt = normalize_title(node.get("title") or "")
    if nt:
        return f"ttl:{nt}"
    return f"_uid:{id(node)}"                              # all three empty → unique sentinel


def bm25_per_term_ranklists(pool: list[dict], terms: list[str]) -> list[list[dict]]:
    """Build ``BM25Okapi`` over ``pool`` ONCE; return one ranked node-ref list
    per term.

    Each list = the ``pool`` nodes with BM25 score > 0 for that term, sorted by
    ``(-score, node["key"])`` (DESC score, then ascending stable ``Paper.key`` —
    the shared relevance-NEUTRAL tiebreak). Positional rank is implicit in list
    order; the round-robin consumes node refs directly.

    Empty-token term (CJK / degenerate → ``_tokenize`` → ``[]``) → an EMPTY list
    (NEVER ``pool[:k]``). ``pool`` is the YEAR-ELIGIBLE library subset (==
    ``lib_corpus``). AUTHORITATIVE EMPTY-CORPUS GUARD: ``if not pool: return
    [[] for _ in terms]`` — never construct ``BM25Okapi([])``. This helper
    stamps NO field on any node; the returned lists hold ``pool`` node refs.
    """
    if not pool:
        return [[] for _ in terms]

    # Lazy import: rank_bm25 not in critical paper-library paths.
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(_paper_text(p)) for p in pool]
    # AUTHORITATIVE EMPTY-VOCAB GUARD: the ``if not pool`` guard above catches an
    # empty corpus, but a NON-empty pool whose EVERY doc tokenizes to ``[]`` (an
    # all-CJK / punctuation-only library with no years — None-year papers are
    # always year-eligible, so they survive into lib_corpus) yields an empty
    # global vocabulary. ``BM25Okapi`` then divides by ``len(self.idf)==0`` in
    # ``_calc_idf`` (``average_idf = idf_sum / 0``) → ``ZeroDivisionError``, which
    # is UNCAUGHT in the orchestrator and crashes the whole search_papers tool
    # (a raw tool error, NOT the fail-closed ``{status:error}``). Degrade exactly
    # like the empty-pool case: no votes, never a crash.
    if not any(corpus):
        return [[] for _ in terms]
    bm25 = BM25Okapi(corpus)

    ranklists: list[list[dict]] = []
    for term in terms:
        tokens = _tokenize(term)
        if not tokens:
            ranklists.append([])              # empty-token term → no votes
            continue
        scores = bm25.get_scores(tokens)
        scored = [(scores[i], pool[i]) for i in range(len(pool)) if scores[i] > 0]
        # (-score, key): DESC score, then ascending stable Paper.key tiebreak.
        scored.sort(key=lambda sn: (-sn[0], sn[1].get("key") or ""))
        ranklists.append([n for _s, n in scored])

    logger.info(
        "bm25_per_term_ranklists: corpus=%d, terms=%d, per-term hits=%s",
        len(pool), len(terms), [len(rl) for rl in ranklists],
    )
    return ranklists


# Standard Cormack RRF damping constant. score(node) = Σ 1/(RRF_K + rank0).
# k=60 is the canonical value: it makes the rank-1 vs rank-2 contribution ratio
# gentle (1/60 : 1/61 ≈ 1.016), so a paper many lists agree on accrues a high
# consensus score WITHOUT any one list's head dominating — calibration-free,
# which is the whole point for the 6 incomparable-native-score backends.
RRF_K = 60


def rrf_fuse(
    ranklists: list[list[dict]],
    key_fn: Callable[[dict], str],
    *,
    k: int = RRF_K,
) -> dict[str, float]:
    """True Reciprocal Rank Fusion (search_papers §2.5 consensus signal).

    ``score(node) = Σ`` over the per-term ranklists the node appears in of
    ``1 / (k + rank0)``, where ``rank0`` is the node's 0-BASED position IN THAT
    list. A node ranked high in MANY lists (many terms / many backends agreeing)
    accrues a HIGH score = cross-term / cross-backend CONSENSUS, using ONLY
    within-list ORDER — never the incomparable native scores of the 6 backends
    (calibration-free, the reason RRF is the right fuser here). A node alone in
    ONE list still earns its single ``1/(k+rank0)``.

    PURE: returns a ``{key_fn(node): score}`` dict and stamps NOTHING on any
    node. Empty ranklists contribute nothing. Keyed by the per-arm ``key_fn``
    (``_node_key`` external, ``lambda n: n["key"]`` library) so a node surfacing
    under several terms fuses onto ONE identity.
    """
    scores: dict[str, float] = {}
    for rl in ranklists:
        for rank0, node in enumerate(rl):
            key = key_fn(node)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank0)
    return scores


def reorder_ranklists_by_rrf(
    ranklists: list[list[dict]],
    rrf: dict[str, float],
    key_fn: Callable[[dict], str],
) -> list[list[dict]]:
    """Re-sort each per-term ranklist by RRF score DESC (search_papers §2.5).

    Consensus papers (high ``rrf``) rise to each term's HEAD; the node's ORIGINAL
    position (native rank / BM25 order) is the relevance-NEUTRAL stable tiebreak,
    so a tie in RRF preserves the per-term native order. RRF reorders WITHIN a
    term only — a term holding a single niche entry is UNCHANGED (its lone head
    stays its head), so the niche-survives floor that the round-robin guarantees
    is never touched. PURE: returns NEW lists holding the SAME node refs.

    This is the within-term half of the locked RRF ⊕ fair-share composition: RRF
    decides the per-term ORDER, ``round_robin`` keeps the between-term FLOOR.
    """
    out: list[list[dict]] = []
    for rl in ranklists:
        # (-rrf, i): RRF DESC primary; original index i the stable tiebreak.
        reordered = [n for _i, n in sorted(
            enumerate(rl),
            key=lambda pair: (-rrf.get(key_fn(pair[1]), 0.0), pair[0]),
        )]
        out.append(reordered)
    return out


def round_robin(ranklists: list[list[dict]], cap: int, key_fn: Callable[[dict], str]) -> list[dict]:
    """Per-term fair-share round-robin (mechanism 1) — ONE function, called once
    per arm with a per-arm KEY FUNCTION. The BETWEEN-term diversity FLOOR of the
    RRF ⊕ fair-share composition: RRF reorders WITHIN each term (so each list's
    head is now the consensus-best); this floor guarantees every non-empty term
    still contributes that head in round order before any term contributes a
    second — so a niche-but-relevant paper surfaced by only ONE term is never
    dropped by the RRF consensus score.

    Walks the per-term ranklists in cycles, taking the next not-yet-emitted node
    from each non-exhausted term per pass, until ``cap`` is reached or every term
    is exhausted. Term order is the ranklist order, so ``ranklists[0]`` (the A1
    anchor = the verbatim core concept) leads every cycle and is emitted first. A
    node appearing under several terms is emitted exactly ONCE (global ``emitted``
    set keyed on ``key_fn(node)``) but is NOT exiled from its niche term.
    ``key_fn`` is ``_node_key`` (external; non-empty via the ``_uid`` sentinel) or
    ``lambda n: n["key"]`` (library; a real stable ``Paper.key``).
    """
    emitted: set[str] = set()                 # keyed on key_fn(node)
    pool: list[dict] = []
    T_active = [t for t in range(len(ranklists)) if ranklists[t]]
    cursors = {t: 0 for t in T_active}
    while len(pool) < cap and T_active:
        for t in list(T_active):
            rl = ranklists[t]
            while cursors[t] < len(rl):
                if key_fn(rl[cursors[t]]) in emitted:
                    cursors[t] += 1
                    continue
                break
            if cursors[t] >= len(rl):
                T_active.remove(t)
                continue
            node = rl[cursors[t]]
            cursors[t] += 1
            emitted.add(key_fn(node))
            pool.append(node)
            if len(pool) >= cap:
                break
    return pool


def paper_to_dict(paper: Any) -> dict:
    """Normalize a ``Paper`` model object (or dict) into a plain dict shape.

    Used to merge ``library.all_papers()`` (Paper objects) with external
    search results (dicts) into one homogeneous pool. Adds a ``_source_origin``
    marker so the §4a partition can tell library vs external nodes apart.
    """
    if isinstance(paper, dict):
        # Already a dict (external candidate). Add origin marker if missing.
        out = dict(paper)
        out.setdefault("_source_origin", "external")
        return out

    # Paper model object (papervault.library.models.Paper)
    return {
        "key":            getattr(paper, "key", None),
        "title":          getattr(paper, "title", "") or "",
        "authors":        list(getattr(paper, "authors", []) or []),
        "year":           getattr(paper, "year", None),
        "venue":          getattr(paper, "venue", "") or "",
        "abstract":       getattr(paper, "abstract", "") or "",
        "doi":            getattr(paper, "doi", "") or "",
        "arxiv_id":       getattr(paper, "arxiv_id", "") or "",
        "citation_count": getattr(paper, "citation_count", 0) or 0,
        "is_review":      bool(getattr(paper, "is_review", False)),
        "publication_types": list(getattr(paper, "publication_types", []) or []),
        "_source_origin": "library",
    }
