"""V-MQ: multi-query + RRF (RAG-Fusion) retrieval variants for the KS query path — EXPERIMENTAL,
reached only via `run_eval.py --variant multiquery|standard` (the shipped query() in aquery.py is
untouched).

WHY: the 49-Q baseline's weak arm is multi-paper recall (recall@12-distinct 0.816; the hard
synthesis Qs sit at 0.2-0.85 because one mix-mode retrieval can't surface all 4-6 gold papers that
a multi-aspect intent needs). RAG-Fusion decomposes the intent into focused sub-queries, retrieves
each through the mix path, and fuses the rankings with Reciprocal Rank Fusion — so a paper that's
only retrievable from one facet still reaches synth.

TWO VARIANTS share this module:

  V-MQ (retrieve_fused, --variant multiquery): the original gated RAG-Fusion. Reranker ON; fusion
  FIRES ONLY when base coverage >= KS_MQ_MIN_COVERAGE; kb_coverage from the original single-intent
  signal. ISOLATION (moves ONLY the headline, never the trap/refusal gate):
    1. kb_coverage ALWAYS comes from the ORIGINAL single-intent retrieval's graph signal — multi-
       query inflates entity counts and would push traps thin→strong.
    2. Fusion FIRES ONLY when the base single-query coverage rank >= the gate. Empty/thin questions
       fall through to the BASELINE single-query data unchanged, so their synth input — and thus the
       judged refusal behaviour — is byte-identical to baseline.

  STANDARD (retrieve_standard, --variant standard): the LOCKED "standard modern RAG" baseline.
  Anchor + orthogonal facets (the upgraded _decompose), each retrieved with the cross-encoder
  reranker OFF (enable_rerank=False) and a WIDE chunk pool, fused by RRF-within ⊕ round-robin-
  between (anchor list leads every cycle) and packed up to _STD_CHUNK_TOKEN_BUDGET=130k tokens of
  chunk content → synth. NO coverage gate:
  the standard pipeline ALWAYS runs the full multi-query+merge. kb_coverage still pinned to the
  ANCHOR (full-intent) single-query's graph signal so multi-query never inflates the entity count
  that drives it (the one isolation we keep — coverage is a property of the intent, not the merge).

RRF: score(item) = Σ_q 1/(K + rank_q(item)), K=60, over the per-sub-query lists; fuse chunks /
entities / relationships, take a sensible top-K of each, rebuild references from the fused chunks
(cited_papers reads references[].file_path).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from lightrag import QueryParam

from papervault import config as _pv_config
from papervault.domain import get_domain
from papervault.knowledge.query import aquery as aq
from papervault.knowledge.store.llm import mimo_complete

logger = logging.getLogger("ks.query.multiquery")

# env-tunable for the #5 optimization loop (defaults = the baselined V-MQ values; unset → byte-identical).
_N_SUBQ = int(os.getenv("KS_MQ_N_SUBQ", "5"))          # focused sub-queries (#5 default-promote: was 4)
_RRF_K = int(os.getenv("KS_MQ_RRF_K", "60"))           # RRF rank constant (lower = sharper consensus bias)
_SUB_CHUNK_TOP_K = int(os.getenv("KS_MQ_SUB_CHUNK_TOP_K", "60"))  # each sub-query's reranked pool (#5: was 16)
# #5 long-context unlock (2026-06-14): LightRAG's default MAX_TOTAL_TOKENS=30000 silently truncated
# the served chunks to ~6-12 regardless of chunk_top_k, leaving MiMo's 1M context unused. Pass 300000
# explicitly on the query QueryParams so the live path serves the full 60-chunk window by default.
_MAX_TOTAL_TOKENS = int(os.getenv("KS_MAX_TOTAL_TOKENS", "300000"))
# #5 long-context loop: facet rerank gate. With rerank ON, all facets push the SAME top passages
# high → RRF consensus collapses the served set to few DISTINCT chunks. Turning rerank OFF on the
# FACETS (base query stays reranked for quality) lets each facet keep its own diverse pool so the
# fused set actually reaches 50+ distinct chunks. Default 'true' = byte-identical to the baselined path.
_MQ_ENABLE_RERANK = os.getenv("KS_MQ_ENABLE_RERANK", "true").lower() == "true"
# #5 drill L2 (2026-06-14): cap chunks-per-paper in the served window so a redundant multi-chunk
# paper can't evict DISTINCT gold from the 60-chunk cap (served sets carry 44-54 distinct/60; a
# paper with 4-5 chunks directly costs distinct-paper recall). 0 = unlimited = baseline behaviour.
_MQ_MAX_CHUNKS_PER_PAPER = int(os.getenv("KS_MQ_MAX_CHUNKS_PER_PAPER", "0"))

# Path 1 (recall-ceiling root-cause 2026-06-18): citation/authority rerank. The recall misses
# concentrate on FOUNDATIONAL papers (Raissi2019=16k cites, Lagaris1997, Parker1965) that the
# semantic stage crowds out behind their many specific descendants. Blend each chunk's semantic
# rank with its paper's CITATION rank — RANK-based (RRF, not raw count: median 29 vs max 70k would
# otherwise dominate) and ONLY over papers ALREADY in the fused pool (an off-topic high-cite paper
# can never enter — it was never retrieved). Default OFF = byte-identical. λ tunes strength.
_CITATION_PRIOR = os.getenv("KS_MQ_CITATION_PRIOR", "0") == "1"
_CITATION_LAMBDA = float(os.getenv("KS_MQ_CITATION_LAMBDA", "0.5"))
_CITATION_MAP_PATH = os.getenv("KS_CITATION_MAP") or str(_pv_config.CITATION_MAP)

# STANDARD variant: the merged-chunk budget handed to synth. The standard path drops the reranker,
# so each sub-query returns a WIDE pool (_STD_SUB_CHUNK_TOP_K) and the RRF⊕round-robin merge orders
# them; we then pack the merged-ranked chunks until ~_STD_CHUNK_TOKEN_BUDGET tokens of chunk content
# is reached. A TOKEN budget (not a fixed chunk count) is robust to chunk-size changes: at 2400-token
# chunks this is ~54 chunks, at 1200 ~108. _STD_TOP_K is the safety ceiling on how many chunks the
# merge enumerates before the token budget is applied.
_STD_CHUNK_TOKEN_BUDGET = 130000  # ~tokens of chunk content packed into the synth context
_STD_TOP_K = 200             # safety ceiling on enumerated merged chunks before the token budget cut
_STD_SUB_CHUNK_TOP_K = 40   # each anchor/facet query's chunk pool (no reranker → wide recall)
_STD_TOP_K_GRAPH = 40       # merged entities / relationships kept (each independently)

# Anchor + facet cap for the STANDARD decompose: term-0 anchor + 2-5 orthogonal facets. The prompt
# constrains the LLM to this; we cap server-side as defensive truncation (a PREFIX, so it never
# displaces the anchor). Mirrors paper-library intent_parser.MAX_SEARCH_TERMS (cap 8) framing.
_STD_MAX_TERMS = 6

# Coverage gate: fuse only when the base single-query coverage is at least this rich. 'empty' NEVER
# fuses (nothing to fuse + clean refuse). Default 'thin' = fuse on thin+strong (reaches the 7/18
# hard movers that sit at 'thin' — exactly the multi-paper Qs needing recall help); the cost is that
# near-adjacent 'thin' traps also fuse, so the ruler's trap gate adjudicates the refusal risk. Set
# KS_MQ_MIN_COVERAGE=strong to restrict fusion to strong-coverage Qs (trap path then byte-identical).
# NOTE: this gate applies to V-MQ (retrieve_fused) ONLY. The STANDARD pipeline removes the gate.
_COV_RANK = {"empty": 0, "thin": 1, "strong": 2}
_MIN_COV = os.environ.get("KS_MQ_MIN_COVERAGE", "thin").strip().lower()

_DECOMPOSE_SYSTEM = """You decompose a researcher's information need into focused retrieval facets \
for a graph + vector literature knowledge base. You output ONLY a JSON array, nothing else.

EXAMPLE (FORMAT reference only — your facet count and content come from the user message; this just \
shows the required SHAPE: object 0 = the anchor/core concept, then distinct-angle facets, each with \
query + hl + ll, no synonym padding):
Information need: I'm inverting Voyager cosmic-ray spectra in the outer heliosphere with physics-informed neural networks — what's known about training stability and the transport-equation setup?
[
  {"query": "physics-informed neural networks for cosmic-ray transport inversion", "hl": ["physics-informed neural network", "cosmic-ray transport", "inverse problem"], "ll": ["PINN", "Parker transport equation", "loss balancing", "Voyager"]},
  {"query": "PINN training stability and failure modes", "hl": ["training dynamics", "gradient pathologies", "convergence"], "ll": ["loss weighting", "neural tangent kernel", "stiff PDE", "collocation sampling"]},
  {"query": "Voyager cosmic-ray measurements in the outer heliosphere and heliopause", "hl": ["outer heliosphere", "local interstellar spectrum", "heliopause crossing"], "ll": ["Voyager 1", "Voyager 2", "CRS instrument", "modulation boundary"]}
]"""
_DECOMPOSE_PROMPT = """Information need:
{intent}

Break this into {n} focused retrieval FACETS. Each facet targets a DISTINCT sub-question,
competing hypothesis, method, experiment, or quantity that must be retrieved separately to answer
the need fully — together they should span the different papers the full answer needs.

For EACH facet output an object with:
  "query": a self-contained search phrase (the topic to look up, NOT a question to the user)
  "hl":    2-4 HIGH-LEVEL / conceptual keywords (themes, mechanisms, phenomena)
  "ll":    2-6 LOW-LEVEL / specific keywords (named entities, instruments, quantities, models)

Return ONLY a JSON array of exactly {n} such objects. These keywords are used DIRECTLY for graph
retrieval (no further extraction), so make them precise and discriminating."""


# ---- STANDARD variant: anchor + orthogonal-facet decompose ------------------------------------
# Ported from paper-library's intent_parser._SYSTEM_PROMPT (services/paper-library/src/
# paper_library/services/intent_parser.py): TERM-0 ANCHOR (the verbatim core concept) + 2-5
# ORTHOGONAL facets over the phenomenon/method/system/regime taxonomy, with the OVER-SPLIT GUARD
# (no synonym shards; an intent with 2 real facets yields 2, not 5). Adapted from pl's flat
# search_terms to KS's {query, hl, ll} shape so each retrieval skips its own LLM keyword extraction.
_STD_DECOMPOSE_SYSTEM = (
    "You decompose a researcher's information need into focused retrieval facets for a graph + "
    # domain label from the active domain pack (ADR-0003, papervault.domain)
    f"vector literature knowledge base, working in **{get_domain().label}**. You output ONLY "
    "a JSON array, nothing else."
)
_STD_DECOMPOSE_PROMPT = """Information need:
{intent}

First THINK: what would a paper that is a PERFECT hit for this intent actually be ABOUT? Name its
core concept and its distinct facets. THEN emit the array. The quality of the facets depends on it.

Decompose the need into ONE ANCHOR + its DISTINCT FACETS. TWO HARD RULES govern the array:

  RULE 1 — ITEM 0 IS THE ANCHOR (the verbatim core concept). The FIRST object's "query" must be the
  single core concept of the intent stated PLAINLY — the phrase a perfectly on-target paper's title
  would literally contain. The merge gives item 0 priority so the facets (expansions) can never
  outvote the user's actual intent. Get the anchor right first; everything else expands around it.

  RULE 2 — FACETS COVER DISTINCT ANGLES, NOT REWORDINGS. Each remaining facet must cover a DIFFERENT
  angle of the intent, not a near-duplicate rewording of the same idea. The facets of a space-physics
  + AI4Science intent are usually some of:
    * phenomenon — what is being studied (e.g. cosmic ray transport, SEP events)
    * method — how (e.g. physics-informed neural network, Bayesian inversion)
    * system — which mission / instrument / object / dataset (e.g. Voyager, Parker Solar Probe)
    * regime — which range / condition / horizon (e.g. outer heliosphere, solar maximum)
  Aim for ONE facet per real angle present in the intent. Vary terminology ACROSS facets, but the
  variation must change the ANGLE, not just reword the same angle.

  OVER-SPLIT GUARD (do NOT manufacture fake facets): one atomic concept is ONE facet, never several
  synonym shards. "PINN", "physics-informed neural network", "physics-informed deep learning" are
  the SAME method facet — pick one. Prefer FEWER orthogonal facets over many near-duplicates: an
  intent with only 2 real facets should yield 2 objects (anchor + 1), NOT 5 padded ones. A very
  broad, facet-less intent (e.g. "machine learning") yields JUST the anchor.

Emit a JSON array: object 0 = the ANCHOR, then 0-4 FACET objects (TOTAL 1-{n}, typically 3-5). For
EACH object output:
  "query": a self-contained search phrase (the topic to look up, NOT a question to the user)
  "hl":    2-4 HIGH-LEVEL / conceptual keywords (themes, mechanisms, phenomena)
  "ll":    2-6 LOW-LEVEL / specific keywords (named entities, instruments, quantities, models)

Return ONLY the JSON array. These keywords are used DIRECTLY for graph retrieval (no further
extraction), so make them precise and discriminating."""


def _parse_facets(raw: str, cap: int) -> list[dict]:
    """Parse an LLM JSON-array reply into [{query, hl, ll}], capped to `cap` (a PREFIX so the
    anchor at index 0 is never displaced). Shared by both decompose variants. Returns [] on any
    parse failure → callers fall back to the single-query baseline."""
    try:
        s = raw[raw.find("["): raw.rfind("]") + 1]
        arr = json.loads(s)
    except Exception as e:  # noqa: BLE001
        logger.warning("multiquery decompose parse failed: %s | raw=%.200s", e, raw)
        return []
    facets: list[dict] = []
    for o in arr:
        if not isinstance(o, dict):
            continue
        q = str(o.get("query") or "").strip()
        hl = [str(x).strip() for x in (o.get("hl") or []) if str(x).strip()]
        ll = [str(x).strip() for x in (o.get("ll") or []) if str(x).strip()]
        if q and (hl or ll):
            facets.append({"query": q, "hl": hl, "ll": ll})
    return facets[:cap]


# Decompose (V-MQ + STANDARD) outer deadline (Phase 0, 2026-06-04 — docs/history/2026-06-04-gateway/2026-06-04-llm-gateway.md).
# The decompose is a MiMo REASONING call (no max_tokens passed → inherits llm.py's 128K default
# _DEFAULT_MAX_TOKENS=131072; an older comment said 8000, which was never set): its think-phase can
# exceed the old 120s, which raised asyncio.TimeoutError → empty → SILENT single-query fallback,
# i.e. the "standard" variant secretly degraded to single-query and the benchmark measured the
# wrong pipeline. 300s gives the reasoning decompose room; still well under the synth/transport
# layers. env-tunable.
_DECOMPOSE_TIMEOUT_S = float(os.getenv("KS_DECOMPOSE_TIMEOUT_S", "300"))


# ---- fcap v2: LLM-driven CONTINUOUS fanout budget (the "standard" + few-shots, user 2026-06-18) ----
# A separate cheap MiMo call (isolated from decompose so it can't perturb the proven facet logic)
# estimates n_sources = how many DISTINCT library papers a thorough answer needs. The served + sub-pool
# budget then scales CONTINUOUSLY with it: clamp(n_sources*SLOPE, _CHUNK_TOP_K floor, MAX cap). Broad
# synthesis Qs get a wider pool (the validated +0.023 @served lever); off-domain/trap Qs estimate ~1 ->
# floor budget -> no trap leak. KS_MQ_FANOUT default "1" (PROMOTED ON 2026-06-18, +0.0324 validated;
# set =0 for the byte-identical 60/5 baseline) — see the line-349 block, the authoritative default.
_FANOUT_SYSTEM = (
    "You size the retrieval budget for a literature query over a space-physics + AI4Science paper "
    "library. Estimate how many DISTINCT library papers a THOROUGH, COMPLETE answer would need to "
    "draw on. Output ONLY a single integer, nothing else."
)
_FANOUT_PROMPT = """Question: {intent}

How many distinct library papers would a thorough answer need to cite? Standard:
- one fact / one method / one phenomenon  -> 2-4
- a mechanism spanning a few works        -> 4-7
- broad synthesis / multi-way comparison / "review the field" / several competing hypotheses,
  particle species, or parameter regimes  -> 8-15
- a question this library almost certainly CANNOT answer (off-domain: condensed-matter, chemistry,
  pure math — outside cosmic-ray / heliophysics / solar / space-weather + AI4Science) -> 1

Examples:
  "What single parameter does the force-field approximation use?"  -> 3
  "Synthesize how solar modulation, drift, and diffusion jointly shape the GCR proton spectrum over a
   solar cycle, across competing model families."  -> 12
  "Explain the light-emission mechanism of single-bubble sonoluminescence."  -> 1

Output ONLY the integer."""


async def _estimate_n_sources(intent: str) -> int:
    """Cheap fanout sizer (fcap v2). Returns the LLM's estimate of distinct papers a thorough answer
    needs (off-domain -> ~1). 0 on any failure -> caller uses the floor budget. Isolated from
    _decompose so it cannot regress the facet logic."""
    try:
        raw = await asyncio.wait_for(
            mimo_complete(_FANOUT_PROMPT.format(intent=intent), system_prompt=_FANOUT_SYSTEM, temperature=0.0),
            timeout=_DECOMPOSE_TIMEOUT_S,
        )
        if not (raw or "").strip():
            # S16 (SDD §6.4): an empty/whitespace-200 is a SILENT LLM failure, NOT a deliberate
            # "narrow question" estimate of ~0 — surface it (mirrors the #21 synth fix) before
            # taking the floor-budget fallback, so a gateway/model hiccup isn't read as a real
            # estimate. Still returns 0 → caller uses the safe floor budget, just no longer silent.
            logger.warning("fcap n_sources got empty/whitespace 200 (S16) — using floor budget")
            return 0
        m = re.search(r"\d+", raw)
        return max(0, min(30, int(m.group()))) if m else 0  # clamp [0,30] sanity
    except Exception as e:  # noqa: BLE001
        logger.warning("fcap n_sources estimate failed: %s", e)
        return 0


async def _decompose(intent: str, n: int = _N_SUBQ) -> list[dict]:
    """V-MQ LLM-decompose the intent into n facets, each {query, hl:[...], ll:[...]}. The keywords
    are fed straight to LightRAG (hl_keywords/ll_keywords) so each sub-query retrieval SKIPS its own
    LLM keyword-extraction pass — the whole point of the rework (was ~7 MiMo calls/fused-Q → now ~3).
    Returns [] on any failure → caller falls back to the single-query baseline."""
    try:
        raw = await asyncio.wait_for(
            mimo_complete(
                _DECOMPOSE_PROMPT.format(intent=intent, n=n),
                # MiMo is a REASONING model: CoT tokens are billed from max_tokens BEFORE the
                # visible JSON. A small cap risks the think-phase eating it all -> empty reply ->
                # parse-fail -> silent single-query fallback. Use the 128K default (llm.py
                # setdefault=131072; user 2026-06-14) so CoT never starves the output; the 300s
                # decompose timeout is what bounds runaway cost. The array is short = unused free.
                system_prompt=_DECOMPOSE_SYSTEM, temperature=0.3,
            ),
            timeout=_DECOMPOSE_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("multiquery decompose failed: %s", e)
        return []
    return _parse_facets(raw, n)


async def _decompose_standard(intent: str, n: int = _STD_MAX_TERMS) -> list[dict]:
    """STANDARD decompose: ONE LLM call → [anchor, *facets], each {query, hl:[...], ll:[...]}.

    Index 0 is the ANCHOR (verbatim core concept of the intent); the rest are 0-4 ORTHOGONAL facets
    (phenomenon/method/system/regime), with the over-split guard. Capped to `n` as a PREFIX so the
    anchor is never displaced. Returns [] on any failure → caller falls back to a single-query path
    on the bare intent."""
    try:
        raw = await asyncio.wait_for(
            mimo_complete(
                _STD_DECOMPOSE_PROMPT.format(intent=intent, n=n),
                # MiMo is a REASONING model: the "First THINK … THEN emit" prompt spends a large
                # CoT budget (billed from max_tokens) BEFORE the JSON array. Use the 128K default
                # (llm.py setdefault; user 2026-06-14) so CoT can't starve the output -> empty ->
                # silent single-query fallback; the 300s decompose timeout bounds runaway cost.
                system_prompt=_STD_DECOMPOSE_SYSTEM, temperature=0.3,
            ),
            timeout=_DECOMPOSE_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("standard decompose failed: %s", e)
        return []
    return _parse_facets(raw, n)


def _rrf(ranked_lists: list[list[dict]], key_fn, top: int | None = None, k: int = _RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion over several reranked item lists. key_fn(item) gives the dedup key;
    the first-seen item dict is kept as the representative. Returns the fused list (top `top`)."""
    scores: dict[Any, float] = {}
    rep: dict[Any, dict] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            kk = key_fn(item)
            if kk is None or kk == "":
                continue
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (k + rank)
            rep.setdefault(kk, item)
    ordered = sorted(scores, key=lambda kk: -scores[kk])
    if top is not None:
        ordered = ordered[:top]
    return [rep[kk] for kk in ordered]


def _empty_data() -> dict[str, Any]:
    return {"chunks": [], "references": [], "entities": [], "relationships": []}


_citation_cache: dict | None = None


def _citation_map() -> dict:
    """Lazy-load {paper_key: citation_count} (exported from paper-library, mirrors the shared
    llm_keys.json pattern). Empty dict on any failure → the citation prior is inert (safe)."""
    global _citation_cache
    if _citation_cache is None:
        try:
            import json
            with open(_CITATION_MAP_PATH) as f:
                _citation_cache = {k: int(v) for k, v in json.load(f).items()}
            logger.info("citation map loaded: %d papers from %s", len(_citation_cache), _CITATION_MAP_PATH)
        except Exception as e:  # noqa: BLE001
            logger.warning("citation map load failed (%s) — citation prior inert", e)
            _citation_cache = {}
    return _citation_cache


def _citation_rerank(ranked_chunks: list[dict]) -> list[dict]:
    """Path-1 authority rerank (two-pass, bias-safe). Re-order the RRF-fused chunk pool by blending
    each chunk's SEMANTIC rank (its position in the fused order) with its paper's CITATION rank
    (within THIS pool only). RANK-based RRF so the raw-count skew can't dominate; pool-only so an
    off-topic high-cite paper can never enter. Lifts foundational papers the semantic stage crowds
    out. No-op (returns input) if the map is unavailable."""
    cmap = _citation_map()
    if not cmap:
        return ranked_chunks
    keyof = lambda c: (c.get("file_path") or "").split("/", 1)[-1]
    pool_keys = list(dict.fromkeys(keyof(c) for c in ranked_chunks))
    by_cit = sorted(pool_keys, key=lambda k: -cmap.get(k, 0))   # most-cited papers in the pool first
    cit_rank = {k: i for i, k in enumerate(by_cit)}
    worst = len(pool_keys)

    def _score(idx_chunk: tuple) -> float:
        idx, c = idx_chunk
        sem = 1.0 / (_RRF_K + idx)                              # semantic rank (chunk's fused position)
        cit = 1.0 / (_RRF_K + cit_rank.get(keyof(c), worst))   # paper's citation rank within the pool
        return sem + _CITATION_LAMBDA * cit

    return [c for _, c in sorted(enumerate(ranked_chunks), key=_score, reverse=True)]


async def retrieve_fused(intent: str, rag: Any) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Multi-query+RRF retrieval. Returns (data, base_metadata, fused_applied).

    data: the fused data dict when base coverage=='strong' AND decomposition succeeded, else the
          UNCHANGED baseline single-query data (so non-strong / failed = baseline path exactly).
    base_metadata: ALWAYS the original single-intent retrieval's metadata (kb_coverage source).
    fused_applied: whether multi-query fusion actually ran (for run logging)."""
    params = QueryParam(mode=aq._QUERY_MODE, top_k=aq._TOP_K, chunk_top_k=aq._CHUNK_TOP_K, enable_rerank=aq._ENABLE_RERANK, max_total_tokens=_MAX_TOTAL_TOKENS)
    base = await rag.aquery_data(intent, params)
    base_ok = isinstance(base, dict) and base.get("status") == "success" and base.get("data")
    base_data = base["data"] if base_ok else _empty_data()
    base_meta = base.get("metadata") if isinstance(base, dict) else None

    # Gate: fuse only when base coverage rank >= KS_MQ_MIN_COVERAGE (default 'thin'). 'empty' never
    # fuses → that path is byte-identical to baseline (clean refuse). The fused set always carries
    # the original intent's results as one input, so fusion can only ADD recall, never lose the
    # baseline's own top chunks.
    cov = aq._assess_coverage(base_meta, base_data)
    if _COV_RANK.get(cov, 0) < _COV_RANK.get(_MIN_COV, 1):
        return base_data, base_meta, False

    # fcap v2: LLM-driven CONTINUOUS fanout budget (KS_MQ_FANOUT, DEFAULT ON 2026-06-18 — validated
    # +0.0324 headline, passes noise floor + jackknife; set KS_MQ_FANOUT=0 to disable). The headline's
    # @served bottleneck is broad/high-fanout questions (corr(gold_size,
    # recall)=-0.61) whose fixed 60-chunk budget can't surface their 8-14 answerers past the global RRF
    # 60-cut. _estimate_n_sources (a cheap separate call, concurrent with decompose) sizes how many
    # distinct library papers a thorough answer needs; the served + sub-pool budget then scales
    # CONTINUOUSLY = clamp(n_sources*SLOPE, _CHUNK_TOP_K, MAX). Off-domain/trap Qs estimate ~1 -> floor
    # budget -> no trap leak (a per-task LLM judgment, not a word-count proxy). user 2026-06-18.
    if os.getenv("KS_MQ_FANOUT", "1") == "1":
        facets, _n_src = await asyncio.gather(_decompose(intent), _estimate_n_sources(intent))
    else:
        facets, _n_src = await _decompose(intent), 0
    if not facets:
        return base_data, base_meta, False

    if _n_src > 0:
        _b = max(aq._CHUNK_TOP_K, min(int(os.getenv("KS_MQ_FANOUT_MAX", "100")),
                                      int(round(_n_src * float(os.getenv("KS_MQ_FANOUT_SLOPE", "7"))))))
        _served_cap = _sub_ctk = _b
    else:
        _served_cap, _sub_ctk = aq._CHUNK_TOP_K, _SUB_CHUNK_TOP_K

    def _sub_param(f: dict) -> QueryParam:
        # hl/ll keywords supplied → LightRAG skips the per-sub-query LLM keyword extraction
        # (operate.py: "if pre-defined keywords are already provided ... return"). The query text
        # still drives the vector/chunk retrieval; the keywords drive the graph retrieval.
        return QueryParam(
            mode=aq._QUERY_MODE, top_k=aq._TOP_K, chunk_top_k=_sub_ctk,
            enable_rerank=_MQ_ENABLE_RERANK, hl_keywords=f["hl"], ll_keywords=f["ll"],
            max_total_tokens=_MAX_TOTAL_TOKENS,
        )

    sub_res = await asyncio.gather(
        *[rag.aquery_data(f["query"], _sub_param(f)) for f in facets], return_exceptions=True
    )

    datas: list[dict] = [base_data]
    n_failed = 0  # facet sub-queries that ERRORED or returned non-success — a REAL failure, as
    # opposed to a success-with-empty-data (a legitimately empty facet retrieval, not a failure).
    for r in sub_res:
        if isinstance(r, dict) and r.get("status") == "success":
            if r.get("data"):
                datas.append(r["data"])
            # else: success-but-empty = a legitimately empty facet retrieval, NOT counted as failed.
        else:
            n_failed += 1  # exception captured by return_exceptions, or a status != 'success' dict
    if n_failed:
        # Sibling of #29 (rerank degradation), on the RECALL arm: a partial/total facet failure
        # silently shrinks the fused recall pool — and if ALL facets fail, fusion no-ops to the
        # single-query baseline while still returning fused_applied=True, so the data layer looks
        # identical to a healthy fusion. SURFACE it LOUD for the operator (the only place this is
        # observable is right here). return_exceptions=True is kept so one bad facet never crashes
        # the query; we only add the missing signal — no behavior change.
        _exc_types = sorted({type(r).__name__ for r in sub_res if isinstance(r, BaseException)})
        logger.warning(
            "multiquery DEGRADED — %d/%d facet sub-queries failed/non-success; fused on %d facet(s)"
            "%s. exc=%s intent=%r",
            n_failed, len(facets), len(datas) - 1,
            " — COLLAPSED to single-query baseline (recall arm lost for this answer)"
            if len(datas) == 1 else " (recall reduced for this answer)",
            _exc_types, intent[:160],
        )

    _ranked_chunks = _rrf(
        [d.get("chunks") or [] for d in datas],
        key_fn=lambda c: ((c.get("content") or "")[:200]) or c.get("file_path"),
        # need the FULL fused pool (not pre-cut) when diversity-cap OR citation-prior will reorder it.
        top=None if (_MQ_MAX_CHUNKS_PER_PAPER or _CITATION_PRIOR) else _served_cap,
    )
    if _CITATION_PRIOR:
        # Path 1: authority rerank the full pool BEFORE the served cut / diversity step, so a
        # foundational paper crowded out of the top-60 by its descendants gets lifted into the window.
        _ranked_chunks = _citation_rerank(_ranked_chunks)
    if _MQ_MAX_CHUNKS_PER_PAPER:
        # L2 diversification: greedily fill chunk_top_k from the RRF order, skipping a chunk once its
        # paper already holds _MQ_MAX_CHUNKS_PER_PAPER chunks → frees served slots for distinct papers.
        chunks = []
        _per_paper: dict = {}
        for c in _ranked_chunks:
            fp = c.get("file_path") or ""
            if _per_paper.get(fp, 0) >= _MQ_MAX_CHUNKS_PER_PAPER:
                continue
            chunks.append(c)
            _per_paper[fp] = _per_paper.get(fp, 0) + 1
            if len(chunks) >= _served_cap:
                break
    else:
        chunks = _ranked_chunks[:_served_cap]  # no-op when neither flag set (_rrf already cut to cap)
    entities = _rrf([d.get("entities") or [] for d in datas], key_fn=lambda e: e.get("entity_name"), top=aq._TOP_K)
    rels = _rrf([d.get("relationships") or [] for d in datas], key_fn=lambda r: (r.get("src_id"), r.get("tgt_id")), top=aq._TOP_K)

    seen: set = set()
    refs: list[dict] = []
    for c in chunks:
        fp = c.get("file_path")
        if fp and fp not in seen:
            seen.add(fp)
            refs.append({"file_path": fp, "reference_id": c.get("reference_id")})

    fused = {"chunks": chunks, "references": refs, "entities": entities, "relationships": rels}
    return fused, base_meta, True


# ---- STANDARD variant: RRF ⊕ fair-share round-robin merge -------------------------------------
# Ported from paper-library bm25_search.py (rrf_fuse / reorder_ranklists_by_rrf / round_robin),
# adapted from Paper nodes to LightRAG chunk/entity/relationship dicts via a per-item key_fn. The
# locked composition: RRF re-orders WITHIN each query's list (consensus to the head) and round-robin
# guarantees the BETWEEN-query fair-share FLOOR, with the ANCHOR list (index 0) leading every cycle.


def _chunk_key(c: dict) -> str:
    """Identity key for a chunk. content[:200] is the stable primary (same passage retrieved under
    different queries fuses onto ONE identity); falls back to file_path, then a per-object _uid
    sentinel so an identity-less chunk stays distinct rather than colliding to ''."""
    body = (c.get("content") or "")[:200]
    if body:
        return f"c:{body}"
    fp = c.get("file_path") or ""
    if fp:
        return f"fp:{fp}"
    return f"_uid:{id(c)}"


def _entity_key(e: dict) -> str:
    name = (e.get("entity_name") or "").strip()
    return f"e:{name}" if name else f"_uid:{id(e)}"


def _rel_key(r: dict) -> str:
    src = r.get("src_id") or ""
    tgt = r.get("tgt_id") or ""
    return f"r:{src}\x1f{tgt}" if (src or tgt) else f"_uid:{id(r)}"


def rrf_fuse(ranklists: list[list[dict]], key_fn, *, k: int = _RRF_K) -> dict[str, float]:
    """True Reciprocal Rank Fusion over several ranked lists. score(item) = Σ over the lists the
    item appears in of 1/(k + rank0), rank0 = 0-based position in that list. An item ranked high in
    MANY lists (many queries agreeing) accrues a HIGH consensus score, using ONLY within-list ORDER.
    PURE: returns {key_fn(item): score}, stamps nothing. (Ported from bm25_search.rrf_fuse.)"""
    scores: dict[str, float] = {}
    for rl in ranklists:
        for rank0, item in enumerate(rl):
            key = key_fn(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank0)
    return scores


def reorder_ranklists_by_rrf(
    ranklists: list[list[dict]], rrf: dict[str, float], key_fn
) -> list[list[dict]]:
    """Re-sort each per-query list by RRF score DESC; the item's ORIGINAL position is the
    relevance-neutral stable tiebreak (a tie preserves the per-query native order). RRF reorders
    WITHIN a query only — a list with a single niche entry is unchanged, so the round-robin floor is
    never touched. PURE: returns NEW lists holding the SAME item refs. (Ported from bm25_search.)"""
    out: list[list[dict]] = []
    for rl in ranklists:
        reordered = [n for _i, n in sorted(
            enumerate(rl),
            key=lambda pair: (-rrf.get(key_fn(pair[1]), 0.0), pair[0]),
        )]
        out.append(reordered)
    return out


def round_robin(ranklists: list[list[dict]], cap: int, key_fn) -> list[dict]:
    """Per-query fair-share round-robin — the BETWEEN-query diversity FLOOR. Walks the per-query
    lists in cycles, taking the next not-yet-emitted item from each non-exhausted list per pass,
    until `cap` is reached or every list is exhausted. List order is the ranklist order, so
    ranklists[0] (the ANCHOR = verbatim core concept) leads every cycle and is emitted first. An
    item appearing under several queries is emitted exactly ONCE (global emitted set keyed on
    key_fn) but is not exiled from its niche query. (Ported from bm25_search.round_robin.)"""
    emitted: set[str] = set()
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


def _merge_rrf_round_robin(ranklists: list[list[dict]], cap: int, key_fn) -> list[dict]:
    """The locked composition: RRF reorders WITHIN each query (consensus to the head), then the
    fair-share round-robin fills `cap` slots BETWEEN queries with the anchor list (ranklists[0])
    leading every cycle. Returns up to `cap` deduped items."""
    rrf = rrf_fuse(ranklists, key_fn)
    reordered = reorder_ranklists_by_rrf(ranklists, rrf, key_fn)
    return round_robin(reordered, cap, key_fn)


_TOKENIZER = None  # lazy LightRAG tiktoken tokenizer for the STANDARD chunk-token budget


def _count_tokens(text: str) -> int:
    """Token count of `text` via the LightRAG tiktoken tokenizer, falling back to a ~4-chars/token
    estimate if the tokenizer is unreachable (never raises — a budget cut must not break retrieval)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from lightrag.utils import TiktokenTokenizer
            _TOKENIZER = TiktokenTokenizer()
        except Exception as e:  # noqa: BLE001
            logger.warning("standard chunk-budget: tokenizer unreachable, using ~4 chars/token: %s", e)
            _TOKENIZER = False  # sentinel: tokenizer permanently unavailable for this process
    if _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:  # noqa: BLE001
            pass
    return (len(text) + 3) // 4  # ~4 chars/token estimate


def _pack_chunks_to_token_budget(chunks: list[dict], budget: int) -> list[dict]:
    """Pack the merged-RANKED chunks (already ordered) from the front until ~`budget` tokens of chunk
    CONTENT is reached, then stop. Always keeps at least the first chunk so a single oversized chunk
    isn't dropped entirely. Robust to chunk-size changes (a token budget, not a fixed chunk count)."""
    packed: list[dict] = []
    used = 0
    for c in chunks:
        t = _count_tokens(c.get("content") or "")
        if packed and used + t > budget:
            break
        packed.append(c)
        used += t
    return packed


async def retrieve_standard(intent: str, rag: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """STANDARD "modern RAG" baseline retrieval. Returns (data, anchor_metadata).

    Pipeline (NO reranker, NO coverage gate — ALWAYS runs the full multi-query+merge):
      1. ONE LLM call → [anchor, *facets] (each {query, hl, ll}); the anchor is item 0.
      2. Retrieve the anchor AND each facet via rag.aquery_data(mode='mix', top_k, chunk_top_k wide,
         enable_rerank=FALSE, hl/ll pre-supplied so each retrieval skips its own keyword extraction).
      3. MERGE chunks via RRF-within ⊕ round-robin-between (anchor list leads every cycle), then
         pack the merged-ranked chunks from the front until ~_STD_CHUNK_TOKEN_BUDGET (130k) tokens
         of chunk content is reached; merge entities/relationships the same way → top
         _STD_TOP_K_GRAPH (40).
      4. Rebuild references from the merged chunks (cited_papers reads references[].file_path).

    anchor_metadata: the ANCHOR (full-intent) single-query's metadata — kb_coverage reads this, so
    multi-query never inflates the entity count that drives coverage.

    On decompose failure (returns []) the anchor falls back to the bare intent and the pipeline runs
    on the single anchor query alone — still the standard path, just un-decomposed."""
    facets = await _decompose_standard(intent)
    if not facets:
        # Decompose failed → run the standard path on the bare intent as the lone anchor query.
        facets = [{"query": intent, "hl": [], "ll": []}]

    def _param(f: dict) -> QueryParam:
        # enable_rerank=False is the defining choice of this path (drop the cross-encoder). hl/ll
        # pre-supplied → LightRAG skips the per-query LLM keyword extraction (operate.py). The query
        # text drives the vector/chunk retrieval; the keywords drive the graph retrieval.
        return QueryParam(
            mode=aq._QUERY_MODE, top_k=aq._TOP_K, chunk_top_k=_STD_SUB_CHUNK_TOP_K,
            enable_rerank=False, hl_keywords=f["hl"], ll_keywords=f["ll"],
        )

    # Anchor first (index 0) so it leads every round-robin cycle; facets follow in order.
    results = await asyncio.gather(
        *[rag.aquery_data(f["query"], _param(f)) for f in facets], return_exceptions=True
    )

    # anchor_meta = the ANCHOR query's metadata (kb_coverage source). The anchor is facets[0]; its
    # result is results[0]. This pins coverage to the full-intent signal, never the merged count.
    anchor_res = results[0]
    anchor_meta = anchor_res.get("metadata") if isinstance(anchor_res, dict) else None

    # Per-query ranked lists, anchor first (so ranklists[0] is the anchor's list).
    chunk_lists: list[list[dict]] = []
    ent_lists: list[list[dict]] = []
    rel_lists: list[list[dict]] = []
    for r in results:
        if isinstance(r, dict) and r.get("status") == "success" and r.get("data"):
            d = r["data"]
            chunk_lists.append(d.get("chunks") or [])
            ent_lists.append(d.get("entities") or [])
            rel_lists.append(d.get("relationships") or [])
        else:
            # Keep a placeholder empty list so the anchor stays at index 0 even if a facet failed.
            chunk_lists.append([])
            ent_lists.append([])
            rel_lists.append([])

    # Merge the chunks (anchor-led RRF⊕round-robin), then pack from the front up to the TOKEN
    # budget — robust to chunk-size changes (vs a fixed chunk count). entities/relationships keep
    # their fixed top-K caps.
    chunks = _merge_rrf_round_robin(chunk_lists, _STD_TOP_K, _chunk_key)
    chunks = _pack_chunks_to_token_budget(chunks, _STD_CHUNK_TOKEN_BUDGET)
    entities = _merge_rrf_round_robin(ent_lists, _STD_TOP_K_GRAPH, _entity_key)
    rels = _merge_rrf_round_robin(rel_lists, _STD_TOP_K_GRAPH, _rel_key)

    seen: set = set()
    refs: list[dict] = []
    for c in chunks:
        fp = c.get("file_path")
        if fp and fp not in seen:
            seen.add(fp)
            refs.append({"file_path": fp, "reference_id": c.get("reference_id")})

    data = {"chunks": chunks, "references": refs, "entities": entities, "relationships": rels}
    return data, anchor_meta
