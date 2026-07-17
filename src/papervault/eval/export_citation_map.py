"""Export {paper_key: citation_count} from paper-library → the static authority map KS reads.

Path 1 (recall-ceiling root-cause, 2026-06-18) feeds this into multiquery.py's citation rerank.
The map is REGENERABLE derived data (citation counts drift slowly) — not committed; refresh
periodically. KS loads it from KS_CITATION_MAP (default research/citation_map.json), mirroring
the shared llm_keys.json pattern.

Run in paper-library's env (it imports paper_library):
    uv run --project ../../../paper-library python experiments/eval/export_citation_map.py
or from the pl service dir:
    cd services/paper-library && uv run python <path>/export_citation_map.py
"""
import json
import os

from paper_library import Library

OUT = os.getenv("KS_CITATION_MAP", "~/projects/dev/research/citation_map.json")


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
