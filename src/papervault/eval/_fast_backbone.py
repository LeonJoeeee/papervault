"""FAST-tier deterministic backbone (no LLM judge) — for quick retrieval-lever iteration.

Reads results/<tag>.jsonl (a run_eval dump) + a gold file, computes the DETERMINISTIC metrics that
need no judge: paper_recall_at_served_distinct (@served, the headline's recall half), gold_citation_
recall, hallucinated_rate. These come from the retrieval (data.chunks[].paper_key + cited_papers) vs
gold, so they're instant + judge-free. Use to FILTER retrieval levers (fcap/v3) on gold_fast.jsonl;
the full tier (judge) is still required to PROMOTE (trap/nugget/cit_sp/faith).
When answer text is available, also report every bracket token that does not resolve to
cited_papers or an operator-source label, with the same total broken out by class (paper
stand-in, operator-shape, numeric/other) so a paper-stand-in regression stays visible (#122).
Answers absent in --no-synth dumps are unscored, not clean. This check measures token
resolution, not whether a source supports a claim.

Usage:  python -m papervault.eval._fast_backbone <tag> [gold_fast.jsonl]
        # compare two: ... _fast_backbone.py fast_fcapv2 ; ... _fast_backbone.py fast_fsdef
"""
import json, statistics, sys
import re
from pathlib import Path

EVAL = Path(__file__).resolve().parent

# Classes of a non-resolving bracket token (#122), in report order. paper_stand_in is the
# residual: a token lands there unless it is clearly operator-shaped or mathematical.
UNRESOLVED_CLASSES = ("paper_stand_in", "operator_shape", "numeric_other")
# An operator source class split off by a separator: "Baumjohann2012, textbook",
# "Textbook: Parks2004", "Textbook:Parks2004" (only the lowercase prefix resolves).
_OPERATOR_SHAPE = re.compile(
    r"(?:textbook|notebook|web)\s*[:,;].*\S|.*\S\s*[:,;]\s*(?:textbook|notebook|web)", re.I)
# Arithmetic, or a slash opening a parenthesised term: "9.74/(4.69+VBs)", "B^2 = ...".
# A bare slash is not enough ("Smith 2019/2020" is a stand-in).
_MATH = re.compile(r"[=+*^<>≈±×−·√]|/\s*\(")
_LETTER = re.compile(r"[^\W\d_]")


def _load(p):
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}


def unresolved_citations(answer: str, cited_papers: list[str]) -> list[str]:
    """Return non-resolving bracket tokens in order, retaining repeated occurrences.

    Deliberately include descriptive stand-ins and numeric tokens: a paper-key-shaped
    regex would hide the defect this check measures. Operator colon labels are allowed.
    """
    cited = set(cited_papers)
    return [token for token in re.findall(r"\[([^\[\]]+)\]", answer)
            if token not in cited and not token.startswith(("textbook:", "notebook:", "web:"))]


def classify_unresolved(token: str) -> str:
    """Class of one non-resolving bracket token; see UNRESOLVED_CLASSES."""
    token = token.strip()
    if _OPERATOR_SHAPE.fullmatch(token):
        return "operator_shape"
    if not _LETTER.search(token) or _MATH.search(token):
        return "numeric_other"
    return "paper_stand_in"


def unresolved_citation_breakdown(answer: str, cited_papers: list[str]) -> dict[str, int]:
    """Per-class counts of unresolved_citations(answer, cited_papers), plus their total.

    Nothing is dropped: bracketed mathematics still counts, under numeric_other.
    """
    counts = dict.fromkeys(UNRESOLVED_CLASSES, 0)
    tokens = unresolved_citations(answer, cited_papers)
    for token in tokens:
        counts[classify_unresolved(token)] += 1
    return {**counts, "total": len(tokens)}


def main():
    tag = sys.argv[1]
    gold_name = sys.argv[2] if len(sys.argv) > 2 else "gold_fast.jsonl"
    gold = _load(EVAL / gold_name)
    recs = _load(EVAL / "results" / f"{tag}.jsonl")
    served_recall, gcr, hallu = [], [], []
    n_ans = 0
    prose_checked = 0
    prose_unavailable = []
    unresolved = {}
    unresolved_classes = dict.fromkeys((*UNRESOLVED_CLASSES, "total"), 0)
    for qid, g in gold.items():
        r = recs.get(qid)
        if not r:
            continue
        answer = r.get("answer")
        if isinstance(answer, str):
            prose_checked += 1
            bad_tokens = unresolved_citations(answer, r.get("cited_papers") or [])
            if bad_tokens:
                unresolved[qid] = bad_tokens
            for cls, n in unresolved_citation_breakdown(answer, r.get("cited_papers") or []).items():
                unresolved_classes[cls] += n
        else:
            prose_unavailable.append(qid)
        gk = set(g.get("gold_keys") or [])
        if not gk:
            continue  # trap — @served undefined (judge handles trap refusal in the full tier)
        n_ans += 1
        served = {c.get("paper_key") for c in (r.get("data", {}).get("chunks") or []) if c.get("paper_key")}
        cited = set(r.get("cited_papers") or [])
        served_recall.append(len(gk & served) / len(gk))
        gcr.append(len(gk & cited) / len(gk))
        hallu.append(len(cited - served) / max(1, len(cited)))
    m = lambda xs: round(statistics.mean(xs), 4) if xs else None
    print(json.dumps({
        "tag": tag, "gold": gold_name, "answerable_scored": n_ans,
        "paper_recall_at_served_distinct": m(served_recall),
        "gold_citation_recall": m(gcr),
        "hallucinated_rate": m(hallu),
        "prose_citation_answers_checked": prose_checked,
        "prose_citation_answers_unavailable": prose_unavailable,
        "unresolved_prose_citations": unresolved,
        "unresolved_prose_citation_classes": unresolved_classes,
    }, indent=2))


if __name__ == "__main__":
    main()
