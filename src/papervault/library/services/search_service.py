"""In-library search with optional LLM rerank.

Distinct from `library.search` (which searches external sources): this only
returns papers already in the library.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..models import Paper
from ..store import Library
from .resolver import _keyword_score, _candidate_dict


class SearchService:

    def __init__(self, library: Library, llm=None):
        self.library = library
        self._llm = llm

    def _ensure_llm(self):
        if self._llm is None:
            from ..llm import get_llm
            self._llm = get_llm()
        return self._llm

    def search(self, query: str, *, limit: int = 10,
               year_min: Optional[int] = None,
               year_max: Optional[int] = None,
               is_review: Optional[bool] = None,
               has_extract: Optional[bool] = None,
               citation_min: int = 0,
               rerank: bool = True) -> dict:
        papers = self.library.all_papers()

        # Apply filters first to keep the LLM rerank set small.
        def keep(p: Paper) -> bool:
            if year_min is not None and (p.year is None or p.year < year_min):
                return False
            if year_max is not None and (p.year is None or p.year > year_max):
                return False
            if is_review is not None and bool(p.is_review) != is_review:
                return False
            if has_extract is not None:
                got = self.library.has_extract(p.key, "md") or self.library.has_extract(p.key, "txt")
                if got != has_extract:
                    return False
            if citation_min > 0 and (p.citation_count or 0) < citation_min:
                return False
            return True

        filtered = [p for p in papers if keep(p)]

        scored = sorted(
            ((_keyword_score(query, p), p) for p in filtered),
            key=lambda x: x[0], reverse=True,
        )
        prelim = [(s, p) for s, p in scored if s > 0][:max(limit * 3, 20)]

        if not prelim:
            return {"results": [], "query": query,
                    "total_in_library": len(papers)}

        if not rerank:
            return {
                "results": [_candidate_dict(p, s, self.library) for s, p in prelim[:limit]],
                "query": query,
                "total_in_library": len(papers),
            }

        # LLM rerank for relevance
        items = [{
            "i": i + 1,
            "key": p.key,
            "title": p.title,
            "authors": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
            "year": p.year,
            "venue": p.venue,
            "abstract_head": (p.abstract or "")[:300],
        } for i, (_, p) in enumerate(prelim)]

        prompt_system = (
            "Re-rank the candidate papers below by relevance to the query. "
            'Return ONLY a JSON object: {"order": [<i>, <i>, ...]} '
            "listing the indices in best→worst order. "
            "Drop indices that are not relevant at all."
        )
        prompt_user = f"Query: {query}\n\nCandidates:\n{json.dumps(items, ensure_ascii=False)}"

        try:
            raw = self._ensure_llm().call([
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ])
        except Exception:
            return {
                "results": [_candidate_dict(p, s, self.library) for s, p in prelim[:limit]],
                "query": query,
                "total_in_library": len(papers),
            }

        m = re.search(r"\{.*\}", raw or "", re.S)
        order_idx: list[int] = []
        if m:
            try:
                order_idx = [int(i) - 1 for i in (json.loads(m.group(0)).get("order") or [])
                             if isinstance(i, int) or (isinstance(i, str) and i.isdigit())]
            except json.JSONDecodeError:
                pass

        seen: set[int] = set()
        ordered: list[tuple[float, Paper]] = []
        for i in order_idx:
            if 0 <= i < len(prelim) and i not in seen:
                seen.add(i)
                ordered.append(prelim[i])
        # append anything LLM dropped, lower priority
        for i, sp in enumerate(prelim):
            if i not in seen:
                ordered.append(sp)

        return {
            "results": [_candidate_dict(p, s, self.library) for s, p in ordered[:limit]],
            "query": query,
            "total_in_library": len(papers),
        }
