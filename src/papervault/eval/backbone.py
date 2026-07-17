"""Deterministic backbone metrics for the KS downstream eval (the fixed 'ruler').

PURE PYTHON, ZERO judge noise. Given ONE saved per-question result (the `aquery_data`
output + KS out-feed fields {cited_papers, kb_coverage, answer}) and the matching gold
entry, compute the deterministic backbone of the 6-metric vector — set-math only, no DB,
no LLM, no network. The Claude-judge overlay (metrics 3/4/5) is a SEPARATE module; this
file owns the noise-floor-free part.

Metrics implemented here (numbering follows the settled KS metric design):
  (1) retrieval recall:
        paper-recall@12  — retrieved paper set = strip 'paper/' from the TOP-12 chunks
                           (the chunk_top_k=12 that reach synth). recall = |retrieved ∩ gold|/
                           |gold|. The @12 cut is enforced IN THIS MODULE (top_n=_CHUNK_TOP_K),
                           not delegated to LightRAG: a saved dump with >12 chunks (or a variant
                           that changed chunk_top_k) is still scored over the top-12 only.
                           '@12' is an upper-bound slice, NOT a fixed denominator — the denom is
                           always |gold|, and token-truncation upstream can leave <12 chunks, so
                           it really means "chunks reaching synth (<=12)".
        hit@12           — 1 if retrieved ∩ gold is non-empty else 0.
        paper-recall@5   — same over only the TOP-5 chunks (rank order preserved by KS).
        NOTE: there is deliberately NO references-recall / rerank-cut-gap metric. The original
        design wanted data.references[] as a "pre-rerank-cut recall ceiling" and the
        references − chunks gap to localize a rerank-cut vs a retrieval-breadth failure. In
        LightRAG 1.4.16 mix/kg_query that is structurally impossible: operate.py:4143 builds
        truncated_chunks (rerank + chunk_top_k cap + min-score + token-truncation) and :4152
        feeds that SAME truncated_chunks to generate_reference_list_from_chunks, so data.chunks
        AND data.references are both the deduped file_paths of the already-cut chunks (it only
        drops 'unknown_source' → references is a SUBSET of chunks, never wider). references-recall
        ≤ chunks-recall always, the gap is ≤0 (never positive), so it can never attribute loss
        to the rerank cut — a no-op that looks meaningful. Measuring a real rerank-cut ceiling
        would need a SEPARATE wide pass at eval time (enable_rerank=False / large chunk_top_k);
        that is deferred (a user decision), not built here. See SDD §6.10 A.(1).
  (2) citation integrity (parse [key] tags out of the prose):
        hallucinated_rate — |prose_keys NOT in retrieved paper set| / |prose_keys|
                            (synth invented a citation to a paper it never saw = worst poison).
        phantom_rate      — |cited_papers NOT in prose_keys| / |cited_papers|
                            (out-feed advertises a citation the prose never actually used).
  (6) gold-citation-recall:
        |cited_papers ∩ gold| / |gold| — did the advertised citations land on gold papers.

  kb_coverage GUARDRAIL (a guardrail, NOT a ranking score):
        per-question: trap_violation — for a trap (gold empty) kb_coverage MUST be 'empty';
                      overconfident   — kb_coverage in the 'strong' tier AND paper-recall@12==0.
        corpus-level:  over_confidence_rate = mean(overconfident) over the run.
        ⚠️ DEAD ON THE FULL CORPUS (2026-06-08 drill D1): this block is corpus-size-dependent —
        at 227k entities _assess_coverage reports 'strong' for EVERY query (all 9 off-domain
        traps included), so trap_violation_rate pins at 1.0 and over_confidence_rate degenerates
        to exactly 1−hit@12. Both remain in the vector as DASHBOARD values only — never gate or
        optimize them (the only way to "improve" them is recalibrating the coverage bins = signal
        laundering; over_confidence_rate was removed from headline.ABSOLUTE_GATES). Refusal
        honesty is measured by the judge's trap_correct_refusal_rate. Revisit only after a
        relevance-bearing kb_coverage redesign.

ALL recall denominators use the GOLD set. Trap questions have |gold|==0; recall is reported
as None for traps (no gold to recall) — they are scored only by the guardrail + (downstream)
the judge's correct-empty-answer credit. Aggregation over the fixed 25 averages each scalar
over the NON-trap questions for the recall/citation metrics and over ALL questions for the
guardrail rates, and is reported as a VECTOR (never collapsed to one number).

Saved result dict shape (one per question), exactly the KS out-feed + the raw aquery_data:
  {
    "qid": "...",                         # matches a gold entry
    "data": {                             # the aquery_data 'data' block
        "chunks":     [{"file_path": "paper/Key", "content": "..."}, ...],  # rank-ordered
        "references": [{"file_path": "paper/Key"}, ...],  # unused by the backbone (see NOTE above)
        "entities": [...], "relationships": [...],        # unused by the backbone
    },
    "metadata": {"processing_info": {"total_entities_found": N}},   # optional
    "cited_papers": ["Key", ...],         # aquery._cited_papers output (already stripped)
    "kb_coverage":  "empty"|"thin"|"strong",
    "answer":       "<synth prose with [Key] tags>"|"(KB 无相关知识)"|"(synthesis LLM failed...)",
  }
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# --- sentinels (kept in lockstep with query/aquery.py + query/synth.py) ------
# Hardcoded (not imported) so the backbone stays a zero-dependency pure module that runs
# on a saved JSON dump with no KS package on the path. If either sentinel changes in the
# query layer, update here too (tests pin the exact strings).
EMPTY_ANSWER_SENTINEL = "(KB 无相关知识)"           # query.aquery._EMPTY["answer"]
SYNTH_FAILED_PREFIX = "(synthesis LLM failed"       # query.synth.SYNTH_FAILED_PREFIX

# kb_coverage tiers, ordered. 'strong' is the over-confidence tier (the design says
# ">=strong"; with the 3-tier scale that is exactly {'strong'}).
_COVERAGE_ORDER = {"empty": 0, "thin": 1, "strong": 2}
_STRONG_TIER = 2

_PAPER_PREFIX = "paper/"

# The @12 chunk cut, enforced IN THIS MODULE (not delegated to LightRAG). MUST mirror
# query/aquery.py::_CHUNK_TOP_K — that is the chunk_top_k the live query path pins, so the
# retrieved paper set we score is exactly "the chunks that reach synth". We deliberately do
# NOT import aquery (the backbone is a zero-dependency pure module that runs on a saved JSON
# dump with no KS package on the path); a guarded cross-check test asserts the two stay equal.
# NOTE: '@12' is an UPPER-BOUND slice ("the gold paper appeared among the top-12 chunks"), not
# a fixed denominator of 12 — recall's denominator is always |gold_keys|, and LightRAG's
# token-truncation step can leave FEWER than 12 chunks, so the real meaning is "chunks reaching
# synth (<=12)". See SDD §6.10 A.(1) (drill 2026-06-02e).
_CHUNK_TOP_K = 12

# A citation key as written inline in the prose: [SomeKey2021], [MKachelriess2019],
# [de2020]. Keys are pl citation keys — alnum, may start lower/upper, may contain digits.
# We do NOT allow whitespace, commas, or path separators inside one tag (those are prose,
# not a key). MINIMUM 2 chars: every real pl key is author-surname+year (>=2 chars), so the
# {2,} floor harmlessly rejects single-letter enumeration markers like "[a]"/"[i]" that the
# synth LLM might emit as list bullets. A tag that is PURELY digits (e.g. "[12]", a numeric
# footnote) is also NOT a paper key and is excluded below.
_CITE_TAG = re.compile(r"\[([A-Za-z0-9_+-]{2,})\]")


def strip_paper_key(file_path: str | None) -> str | None:
    """'paper/Corti2018' -> 'Corti2018'; non-paper or blank -> None.

    Mirrors aquery._cited_papers' membership rule: only 'paper/<key>' file_paths are
    citeable; textbook/web/unknown sources are not paper keys and are dropped.
    """
    if not file_path or not file_path.startswith(_PAPER_PREFIX):
        return None
    key = file_path[len(_PAPER_PREFIX):]
    return key or None


def _paper_set(items: list[dict] | None) -> list[str]:
    """Ordered, deduped paper keys from a list of {file_path:...} (rank order preserved)."""
    out: list[str] = []
    seen: set[str] = set()
    for it in items or []:
        key = strip_paper_key((it or {}).get("file_path"))
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def retrieved_papers_from_chunks(data: dict, top_n: int | None = _CHUNK_TOP_K) -> list[str]:
    """Paper set from data.chunks[] (the chunks that reach synth), rank order preserved.

    top_n caps to the first N CHUNKS (not the first N distinct papers) before deduping, so
    paper-recall@12 reflects 'the gold paper appeared among the top-12 (chunk_top_k) retrieved
    chunks' and paper-recall@5 the same over the top-5. DEFAULT is the @12 cut (_CHUNK_TOP_K),
    enforced HERE rather than relying on LightRAG having already capped data.chunks upstream —
    so a dump with >12 chunks (or a variant with a different chunk_top_k) is still scored @12.
    Pass top_n=None to score over ALL present chunks (rarely wanted; the @12/@5 cuts are canon).
    """
    chunks = data.get("chunks") or []
    if top_n is not None:
        chunks = chunks[:top_n]
    return _paper_set(chunks)


def distinct_papers_from_chunks(data: dict, top_n: int = _CHUNK_TOP_K) -> list[str]:
    """First `top_n` DISTINCT papers in the reranked chunk order (a denominator-fair @12 set).

    DIFFERENCE from retrieved_papers_from_chunks(data, top_n=12): that one cuts the first 12
    CHUNKS then dedups, so a dup-dominated query (the same paper filling several of the 12 chunk
    slots) often surfaces only 5-7 DISTINCT papers in the 12 chunk slots — which lets a variant
    that merely de-dups / diversifies chunks fill the 12 distinct-paper budget and MECHANICALLY
    inflate recall without truly retrieving more gold (drill 2026-06-02 H4). This walks the FULL
    reranked chunk order (top_n=None → no chunk cut), dedups to distinct papers, and keeps the
    first `top_n` DISTINCT papers — so the denominator is a fair "12 distinct papers" budget.
    The HEADLINE uses recall over THIS set (paper_recall_at_12_distinct); the chunk-based
    paper_recall_at_12 stays as a secondary reported value. See SDD §6.10 A.(1)/G (H4).
    """
    return retrieved_papers_from_chunks(data, top_n=None)[:top_n]


def is_real_prose(answer: str | None) -> bool:
    """True iff the answer is real synthesized prose (not the empty sentinel / synth-fail).

    Citation parsing (metric 2) must SKIP the empty sentinel and any SYNTH_FAILED answer —
    those carry no genuine inline citations and would pollute hallucinated/phantom rates.
    """
    if not answer:
        return False
    a = answer.strip()
    if a == EMPTY_ANSWER_SENTINEL:
        return False
    if a.startswith(SYNTH_FAILED_PREFIX):
        return False
    return True


def parse_prose_citations(answer: str | None) -> list[str]:
    """Distinct [key] tags in the prose, in first-appearance order.

    Excludes purely-numeric tags (e.g. '[12]' numeric footnotes) — those are not paper keys.
    Returns [] for the empty sentinel / synth-fail (is_real_prose gate).
    """
    if not is_real_prose(answer):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _CITE_TAG.finditer(answer):
        key = m.group(1)
        if key.isdigit():
            continue
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _recall(retrieved: set[str], gold: set[str]) -> float | None:
    """|retrieved ∩ gold| / |gold|; None when there is no gold (trap)."""
    if not gold:
        return None
    return len(retrieved & gold) / len(gold)


def _rate(numerator_set: set[str], denom: list[str]) -> float | None:
    """|numerator_set| / |denom|; None when denom is empty (rate undefined)."""
    if not denom:
        return None
    return len(numerator_set) / len(denom)


@dataclass
class QuestionMetrics:
    """The deterministic backbone for ONE question. Floats in [0,1]; None where undefined.

    None semantics (so aggregation can drop-not-zero, never silently bias the mean):
      - recall fields are None for trap questions (no gold).
      - hallucinated_rate / citation_recall_in_prose are None when the prose has no
        inline citations (rate over zero tags is undefined).
      - phantom_rate / gold_citation_recall denominators: phantom is None when there are
        no cited_papers; gold_citation_recall is None for traps.
    """

    qid: str
    is_trap: bool

    # (1) retrieval recall
    paper_recall_at_12: float | None          # over the first 12 CHUNKS (dup-dominated; secondary)
    paper_recall_at_12_distinct: float | None  # over the first 12 DISTINCT papers (legacy headline; H4)
    paper_recall_at_served_distinct: float | None  # over ALL distinct papers served to synth (long-ctx headline)
    hit_at_12: int | None
    paper_recall_at_5: float | None

    # (2) citation integrity (over the prose)
    has_real_prose: bool
    n_prose_citations: int
    hallucinated_rate: float | None
    phantom_rate: float | None

    # (6) gold citation recall (over cited_papers)
    gold_citation_recall: float | None

    # kb_coverage guardrail
    kb_coverage: str
    trap_violation: bool       # trap but kb_coverage != 'empty'
    overconfident: bool        # kb_coverage in strong tier AND paper_recall_at_12 == 0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_question_metrics(result: dict, gold: dict) -> QuestionMetrics:
    """The whole deterministic backbone for one (result, gold) pair. Pure, no I/O.

    `result` and `gold` must share the same qid (caller's responsibility; asserted).
    """
    qid = result.get("qid")
    if qid is not None and gold.get("qid") is not None and qid != gold.get("qid"):
        raise ValueError(f"qid mismatch: result {qid!r} vs gold {gold.get('qid')!r}")
    qid = qid or gold.get("qid") or "?"

    gold_keys = set(gold.get("gold_keys") or [])
    is_trap = len(gold_keys) == 0

    data = result.get("data") or {}

    # (1) retrieval recall — both cuts enforced in-module (top_n defaults to _CHUNK_TOP_K=12).
    chunk_papers_12 = set(retrieved_papers_from_chunks(data))            # top-12 chunks (<=12)
    chunk_papers_5 = set(retrieved_papers_from_chunks(data, top_n=5))    # top-5 chunks
    # H4: first 12 DISTINCT papers in reranked order (denominator-fair) — the HEADLINE's recall.
    distinct_papers_12 = set(distinct_papers_from_chunks(data))

    recall_12 = _recall(chunk_papers_12, gold_keys)
    recall_12_distinct = _recall(distinct_papers_12, gold_keys)
    recall_5 = _recall(chunk_papers_5, gold_keys)
    # @served: recall over ALL distinct papers actually handed to synth (top_n=None). In the
    # long-context regime (serving 60 chunks) this is the meaningful COVERAGE term — @12 is slack
    # there. == recall_12_distinct whenever served<=12 (so short-context numbers are unchanged).
    # H4-safe: distinct-paper counting + the retrieval ceiling cap it; over-serving past the
    # unimodal peak is penalised by the trap/faithfulness gates, not rewarded here.
    recall_served_distinct = _recall(set(distinct_papers_from_chunks(data, top_n=None)), gold_keys)
    hit_12 = None if is_trap else int(bool(chunk_papers_12 & gold_keys))

    # (2) citation integrity
    prose = result.get("answer")
    has_prose = is_real_prose(prose)
    prose_keys = parse_prose_citations(prose)
    # hallucinated = the synth cited a paper it was NEVER retrieved/shown (a fabrication). The
    # "allowed" set must therefore be the FULL served chunk list (what synth actually saw), NOT the
    # first-12-chunk cut: a variant that serves chunk_top_k>12 (e.g. the #5 cap-raise variants) hands
    # synth 18-20 chunks, so legitimately citing a paper from served chunk #15 is NOT a hallucination.
    # Drill 2026-06-14: using chunk_papers_12 here mislabeled ~100 real cites/run as hallucinated on
    # the cap-raise variants (worse_by 0.20 was a pure artifact). No-op for the 12-chunk baseline/
    # V-MQ+V-SR runs (served==12 → top_n=None == the first-12). recall_12/hit_12 still use the @12 cut.
    allowed = set(retrieved_papers_from_chunks(data, top_n=None))  # ALL papers handed to synth
    hallucinated = {k for k in prose_keys if k not in allowed}
    hallucinated_rate = _rate(hallucinated, prose_keys)

    cited_papers = list(result.get("cited_papers") or [])
    prose_key_set = set(prose_keys)
    phantom = {k for k in cited_papers if k not in prose_key_set}
    # phantom_rate is only meaningful when the prose is real (else there are no prose_keys
    # to be a phantom against). With synth-fail/empty prose, leave it None.
    phantom_rate = _rate(phantom, cited_papers) if has_prose else None

    # (6) gold citation recall
    gold_citation_recall = (
        None if is_trap else len(set(cited_papers) & gold_keys) / len(gold_keys)
    )

    # kb_coverage guardrail
    kb = result.get("kb_coverage") or "empty"
    trap_violation = is_trap and kb != "empty"
    overconfident = (
        _COVERAGE_ORDER.get(kb, 0) >= _STRONG_TIER
        and recall_12 is not None
        and recall_12 == 0.0
    )

    return QuestionMetrics(
        qid=qid,
        is_trap=is_trap,
        paper_recall_at_12=recall_12,
        paper_recall_at_12_distinct=recall_12_distinct,
        paper_recall_at_served_distinct=recall_served_distinct,
        hit_at_12=hit_12,
        paper_recall_at_5=recall_5,
        has_real_prose=has_prose,
        n_prose_citations=len(prose_keys),
        hallucinated_rate=hallucinated_rate,
        phantom_rate=phantom_rate,
        gold_citation_recall=gold_citation_recall,
        kb_coverage=kb,
        trap_violation=trap_violation,
        overconfident=overconfident,
    )


def _mean(values: list[float | None]) -> float | None:
    """Mean over the non-None values; None when every value is None (nothing to average)."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def aggregate(per_question: list[QuestionMetrics]) -> dict:
    """Aggregate the per-question backbone into the corpus-level metric VECTOR.

    Reported as a vector (never one number). Recall/citation scalars average over the
    questions where they are DEFINED (None dropped, not zero-filled) — so trap questions
    don't drag recall to 0, and questions with no inline citations don't define a 0/0 rate.
    The guardrail rates average over the relevant population:
      trap_violation_rate over the traps; over_confidence_rate over the non-trap questions
      (a trap can't be over-confident — it has no gold to miss).
    """
    nonzero = [q for q in per_question]
    traps = [q for q in nonzero if q.is_trap]
    answerable = [q for q in nonzero if not q.is_trap]

    return {
        "n_questions": len(nonzero),
        "n_traps": len(traps),
        "n_answerable": len(answerable),
        # (1) retrieval
        "paper_recall_at_12": _mean([q.paper_recall_at_12 for q in answerable]),
        "paper_recall_at_12_distinct": _mean([q.paper_recall_at_12_distinct for q in answerable]),
        "hit_at_12": _mean([float(q.hit_at_12) for q in answerable if q.hit_at_12 is not None]),
        "paper_recall_at_5": _mean([q.paper_recall_at_5 for q in answerable]),
        # (2) citation integrity
        "hallucinated_rate": _mean([q.hallucinated_rate for q in answerable]),
        "phantom_rate": _mean([q.phantom_rate for q in answerable]),
        # (6) gold citation recall
        "gold_citation_recall": _mean([q.gold_citation_recall for q in answerable]),
        # guardrail
        "trap_violation_rate": _mean([float(q.trap_violation) for q in traps]),
        "over_confidence_rate": _mean([float(q.overconfident) for q in answerable]),
    }


def evaluate_run(results: list[dict], gold_by_qid: dict[str, dict]) -> dict:
    """End-to-end: per-question backbone + corpus aggregate for one full run.

    `results` is the list of saved per-question result dicts; `gold_by_qid` maps qid -> gold
    entry (e.g. {g['qid']: g for g in load_gold()}). Every result must have a gold entry.
    Returns {'per_question': [dict,...], 'aggregate': {...}}.
    """
    per_q: list[QuestionMetrics] = []
    for r in results:
        qid = r.get("qid")
        if qid not in gold_by_qid:
            raise KeyError(f"result qid {qid!r} has no gold entry")
        per_q.append(compute_question_metrics(r, gold_by_qid[qid]))
    return {
        "per_question": [q.to_dict() for q in per_q],
        "aggregate": aggregate(per_q),
    }
