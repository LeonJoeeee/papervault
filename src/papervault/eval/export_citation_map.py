"""Export {paper_key: citation_count} from paper-library → the static authority map KS reads.

Path 1 (recall-ceiling root-cause, 2026-06-18) feeds this into multiquery.py's citation rerank.
The map is REGENERABLE derived data (citation counts drift slowly) — not committed; refresh
periodically. The knowledge side reads it from ``PAPERVAULT_CITATION_MAP`` (default
``citation_map.json`` under the papervault data dir; the legacy ``KS_CITATION_MAP`` is still
honored).

Run:
    python -m papervault.eval.export_citation_map
"""
import json
import os

from papervault import config
from papervault.library import Library

OUT = os.getenv("KS_CITATION_MAP") or str(config.CITATION_MAP)


def main() -> None:
    lib = Library()
    m = {}
    for p in lib.all_papers():
        k = getattr(p, "key", None)
        if k:
            m[k] = int(getattr(p, "citation_count", 0) or 0)
    with open(OUT, "w") as f:
        json.dump(m, f)
    nz = sum(1 for v in m.values() if v > 0)
    print(f"wrote {len(m)} papers ({nz} with citation_count>0) → {OUT}")


if __name__ == "__main__":
    main()
