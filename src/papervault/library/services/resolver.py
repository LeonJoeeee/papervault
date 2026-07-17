"""Resolve fuzzy text references ('Potgieter's living review') to library candidates.

Resolver only looks at what's already in the library — it does not reach out
to external APIs. AddService handles "go fetch from outside" when resolver
returns nothing.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..models import Paper, normalize_title
from ..store import Library


_MAX_LLM_CANDIDATES = 30


def _keyword_score(query: str, paper: Paper) -> float:
    """Cheap textual relevance: token overlap with title + venue + first 200 abs chars."""
    q = re.findall(r"[a-zA-Z]+", query.lower())
    if not q:
        return 0.0
    blob = " ".join([
        paper.title.lower(),
        paper.venue.lower(),
        (paper.abstract or "")[:300].lower(),
        " ".join(paper.authors).lower(),
        str(paper.year or ""),
    ])
    tokens = set(re.findall(r"[a-zA-Z]+", blob))
    overlap = sum(1 for t in q if t in tokens) / max(len(q), 1)
    return overlap


# Fallback admission count for the all-zero case. The s>0 keyword pre-filter
# used to be a HARD gate, so a token-disjoint query (only digits, or words
# absent from title+venue+abstract-head+authors) made an in-library paper
# unreachable by fuzzy text even when the LLM would have recognised it. So when
# NO candidate has positive overlap we still hand the LLM the top-N by score
# (here all 0) to rescue a recognisable-but-token-disjoint target; the keyword
# score is still the ranker, just no longer a hard admission veto. (When some
# candidates DO score >0, only those positives are admitted — see resolve().)
_LLM_ADMIT_FLOOR = 5


def _candidate_dict(paper: Paper, score: float, library: Library,
                    *, score_kind: str = "llm") -> dict:
    """Resolver-INTERNAL triage dict (NOT caller-facing — the MCP layer
    re-projects every candidate through ``_paper_dict`` before it reaches the
    executor; see SDD §5 I-PROJ).

    ``score_kind`` records the score's provenance so the server's auto-resolve
    gate can branch: ``"llm"`` = LLM confidence (the LLM-rerank path),
    ``"keyword"`` = alpha-token-overlap fraction (every degraded fallback path —
    LLM down / no-JSON / bad-JSON). A keyword overlap of 1.0 is NOT an LLM
    "this is the paper" — the server holds keyword-derived scores to a stricter
    auto-resolve floor so a token-saturated WRONG paper can't auto-``found`` on
    overlap alone during an LLM outage.
    """
    return {
        "key": paper.key,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "abstract": paper.abstract,
        "citation_count": paper.citation_count,
        "is_review": paper.is_review,
        "in_library": True,
        "has_extract_md": library.has_extract(paper.key, "md"),
        "has_pdf": library.has_pdf(paper.key),
        "score": round(float(score), 4),
        "score_kind": score_kind,
    }


class ResolverService:
    """LLM-augmented fuzzy lookup of papers already in the library."""

    def __init__(self, library: Library, llm=None):
        self.library = library
        self._llm = llm  # injected; created lazily if None

    def _ensure_llm(self):
        if self._llm is None:
            from ..llm import get_llm
            self._llm = get_llm()
        return self._llm

    def resolve(self, query: str, *, top_k: int = 5,
                use_llm: bool = True, min_confidence: float = 0.7) -> list[dict]:
        """Return up to top_k candidates for the fuzzy query.

        Pipeline:
          1. token-overlap pre-filter to top _MAX_LLM_CANDIDATES
          2. (optional) LLM re-rank for confidence + ordering
          3. drop anything below min_confidence (after LLM); pass-through if no LLM
        """
        papers = self.library.all_papers()
        if not papers:
            return []

        scored = sorted(
            ((_keyword_score(query, p), p) for p in papers),
            key=lambda x: x[0], reverse=True,
        )
        # Admission to the LLM rerank: prefer the keyword-positive candidates;
        # but when NONE score positive, still admit the top-N by score (all 0)
        # so a token-disjoint but recognisable target (pure year/volume query,
        # or words not in the first 300 abstract chars) stays reachable — the
        # LLM recognises it from the full metadata. The keyword score still
        # RANKS; it is no longer a hard >0 admission veto. (Conservative: when
        # some candidates DO score >0, only those positives are admitted.)
        positive = [(s, p) for s, p in scored if s > 0]
        if positive:
            prelim = positive[:_MAX_LLM_CANDIDATES]
        else:
            # No keyword overlap at all — still let the LLM look at the top-N
            # (it sees full metadata, not just the alpha-token blob).
            prelim = scored[:_LLM_ADMIT_FLOOR]
        if not prelim:
            return []

        if not use_llm:
            # Degraded (no-LLM) path: only the keyword-positive candidates are
            # trustworthy without an LLM to judge — never surface a 0-overlap
            # candidate as a keyword match.
            kw = [(s, p) for s, p in prelim if s > 0]
            return [_candidate_dict(p, s, self.library, score_kind="keyword")
                    for s, p in kw[:top_k]]

        return self._llm_rerank(query, prelim, top_k=top_k, min_confidence=min_confidence)

    def _llm_rerank(self, query: str, prelim: list[tuple[float, Paper]],
                    *, top_k: int, min_confidence: float) -> list[dict]:
        items = []
        for i, (_, p) in enumerate(prelim, 1):
            items.append({
                "i": i,
                "key": p.key,
                "title": p.title,
                "authors": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
                "year": p.year,
                "venue": p.venue,
                "abstract_head": (p.abstract or "")[:300],
            })

        prompt_system = (
            "You map a vague paper reference to specific entries in a personal "
            "library. Return ONLY a JSON object on one line: "
            '{"matches": [{"i": <int>, "confidence": 0..1, "reason": "<short>"}, ...]}. '
            "Order by confidence descending. Skip entries you have no signal for. "
            "Confidence > 0.7 means strong signal."
        )
        prompt_user = (
            f"Query: {query}\n\n"
            f"Candidates (JSON):\n{json.dumps(items, ensure_ascii=False)}"
        )

        try:
            llm = self._ensure_llm()
            raw = llm.call([{"role": "system", "content": prompt_system},
                            {"role": "user", "content": prompt_user}])
        except Exception:
            # LLM down → fall back to keyword score (tagged so the server's
            # auto-resolve gate holds it to the stricter keyword floor).
            return self._keyword_fallback(prelim, top_k)

        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            return self._keyword_fallback(prelim, top_k)
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return self._keyword_fallback(prelim, top_k)

        out = []
        for match in parsed.get("matches", [])[:top_k]:
            try:
                idx = int(match["i"]) - 1
                conf = float(match.get("confidence", 0))
            except (KeyError, ValueError, TypeError):
                continue
            if not (0 <= idx < len(prelim)):
                continue
            if conf < min_confidence:
                continue
            paper = prelim[idx][1]
            out.append(_candidate_dict(paper, conf, self.library, score_kind="llm"))
        return out

    def _keyword_fallback(self, prelim: list[tuple[float, Paper]],
                          top_k: int) -> list[dict]:
        """Degraded-path result (LLM down / unusable response): keyword-overlap
        scored candidates, tagged ``score_kind="keyword"`` so the server applies
        the stricter auto-resolve floor. Only positive-overlap candidates are
        surfaced — a 0-overlap admit (token-disjoint, only reachable via the LLM)
        is NOT a keyword match and must not auto-resolve on a degraded path.
        """
        kw = [(s, p) for s, p in prelim if s > 0]
        return [_candidate_dict(p, s, self.library, score_kind="keyword")
                for s, p in kw[:top_k]]
