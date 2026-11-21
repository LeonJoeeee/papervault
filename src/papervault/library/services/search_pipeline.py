"""Search serving stages; registration is a separate, nonblocking command.

Stage outputs are request-local snapshots. External rank evidence survives folds
into local identities, and traces describe the candidates actually served.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Callable

from papervault import config
from ..download_telemetry import network_slot
from ..identity_v2 import serving_bibliographic_key
from ..search import CAPS, observe_source_gates
from ..sources.identity import merge_source_identities, normalized_source_identities
from ..sources.inspire import merge_inspire_fields
from . import concurrency
from .bm25_search import (
    RRF_K, _node_key, _paper_text, _tokenize, bm25_per_term_ranklists, paper_to_dict,
    reorder_ranklists_by_rrf, round_robin, rrf_fuse,
)
from .intent_parser import named_entity_groups, observe_intent
from .kg_search import kg_local_search
from .registration import (
    MAX_CANDIDATES, ServingTrace, _IDENTIFIERS, _METADATA, _fill, _find, _view,
    preview_candidates, registration_rejection, serving_excluded,
)

logger = logging.getLogger(__name__)
RETURN_THRESHOLD = 0.4
EXT_CAP = 60
LIB_CAP = 45
ALSO_INCLUDE_CAP = 45
_YEAR_NEUTRAL = 10**9
SEARCH_AUTHORITY_PRIOR = config.search_authority_prior_enabled()
BACKENDS = tuple(CAPS)
UNCONFIGURED_CHECK = {
    "ads": lambda: bool(os.environ.get("ADS_API_TOKEN")),
    "core": lambda: bool(os.environ.get("CORE_API_KEY", "").strip()),
}


def _parse_authority_lambda(raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid search authority weight; using 0.25")
        return 0.25
    if not 0 <= value <= 1:
        logger.warning("Search authority weight outside [0, 1]; clamping")
        return min(1.0, max(0.0, value))
    return value


SEARCH_AUTHORITY_LAMBDA = _parse_authority_lambda(
    os.environ.get("PAPERVAULT_SEARCH_AUTHORITY_LAMBDA", "0.25"))

_SINGULAR_VENUE_WORDS = {
    "letters", "reviews", "sciences", "proceedings", "methods", "systems", "statistics", "physics",
}
_VENUE_TIERS = (
    (1.5, (
        "10.1103/physrevlett", "10.1103/physrevx", "10.1103/revmodphys", "10.1126/science",
        "10.1038/nature", "10.1038/s41567", "10.1038/s42254", "10.1007/s11214", "10.12942/lrsp",
        "10.1007/s41116", "10.1073/pnas", "10.1016/j.physrep", "10.1088/0034-4885",
        "10.1146/annurev", "10.1137/s00361445",
        "10.1038/s41928", "10.1038/s42256", "10.1109/jproc", "10.1109/comst",
        "10.1016/j.paerosci",
    ), (
        "prl", "prx", "rmp", "ssr", "lrsp", "pnas", "physical review letter", "physical review x",
        "review of modern physic", "science", "science advance", "nature", "nature physic",
        "nature review physic", "space science review", "living review in solar physic",
        "review of geophysic", "physic report", "report on progress in physic",
        "annual review of nuclear and particle science", "annual review of astronomy and astrophysic",
        "annual review of fluid mechanic", "siam review",
        "proceeding of the national academy of science",
        "nature electronics", "nature machine intelligence", "proceeding of the ieee",
        "ieee communications surveys tutorials", "progress in aerospace science",
    )),
    (1.2, (
        "10.1086/", "10.3847/", "10.1029/", "10.1051/0004-6361", "10.1093/mnras",
        "10.1103/physrevd", "10.1103/physreve", "10.1103/physrevc", "10.1088/1475-7516",
        "10.1016/j.astropartphys", "10.1016/j.nuclphysb", "10.1016/j.nima", "10.1007/s11207",
        "10.1016/j.pss", "10.1016/0032-0633", "10.1016/j.jcp", "10.1016/j.cma", "10.1007/s40304",
        "10.1109/taes", "10.1109/tns", "10.1016/j.actaastro", "10.2514/1.a", "10.2514/1.j",
        "10.1016/j.ast", "10.1016/j.cja", "10.1109/tgrs", "10.1109/jsac", "10.1109/twc",
        "10.1109/tmc", "10.1109/jiot", "10.1109/mnet", "10.1109/tc.", "10.1109/tcad",
        "10.1109/tpds",
    ), (
        "apj", "apjl", "apjs", "jgr", "grl", "aa", "mnras", "prd", "prc", "pre", "jcap", "jcp",
        "sisc", "cmame", "icml", "neurips", "nips", "iclr", "astrophysical journal", "astrophys j",
        "journal of geophysical research", "geophysical research letter", "astronomy and astrophysic",
        "astron astrophys", "monthly notice of the royal astronomical society", "mon not roy astron soc",
        "physical review d", "phys rev d", "physical review c", "physical review e",
        "journal of cosmology and astroparticle physic", "astroparticle physic", "nuclear physic b",
        "nuclear instrument and method in physic research", "solar physic", "planetary and space science",
        "journal of computational physic", "siam journal on scientific computing",
        "computer method in applied mechanic and engineering", "international conference on machine learning",
        "neural information processing system", "advance in neural information processing system",
        "international conference on learning representation", "communication in mathematic and statistic",
        # Names use _venue_name_multiplier's punctuation removal and singularization.
        "ieee transactions on aerospace and electronic system", "taes",
        "ieee transactions on nuclear science", "tns", "acta astronautica",
        "journal of spacecraft and rockets", "aiaa journal", "aerospace science and technology",
        "chinese journal of aeronautics", "ieee transactions on geoscience and remote sensing", "tgrs",
        "ieee journal on selected areas in communications", "jsac",
        "ieee transactions on wireless communications", "twc",
        "ieee transactions on mobile computing", "tmc", "ieee internet of things journal",
        "ieee network", "asplos", "isca", "micro", "hpca", "osdi", "sosp", "nsdi", "eurosys",
        "mobicom", "sensys", "dac", "fpga", "cvpr", "iccv", "eccv", "sc",
        "ieee transactions on computers",
        "ieee transactions on computeraided design of integrated circuits and system", "tcad",
        "ieee transactions on parallel and distributed system", "tpds",
    )),
)
_LOW_TIER_VENUE = re.compile(
    r"agu fall meeting|agu spring meeting|meeting abstract|conference abstract|"
    r"cospar scientific assembly|egu general assembly|aas meeting|aps division of plasma|"
    r"bulletin of the american astronomical society|bulletin of the american physical society|"
    r"^pos$|proceeding of science|international cosmic ray conference|aip conf proc|"
    r"aip conference proceeding|hal ccsd|^ccsd$")


@lru_cache(maxsize=2048)
def _venue_name_multiplier(venue: str) -> float | None:
    words = re.sub(r"[^a-z0-9\s]", "", venue.lower()).split()
    stem = " ".join(word[:-1] if word in _SINGULAR_VENUE_WORDS else word for word in words)
    if not stem:
        return None
    if _LOW_TIER_VENUE.search(stem):
        return 0.5
    # Match exact names across both tiers before fuzzy matching: Physical Review
    # C/D/E must not inherit the S tier from their similarity to Physical Review X.
    for multiplier, _, names in _VENUE_TIERS:
        if stem in names:
            return multiplier
    ratio, multiplier = max(
        (SequenceMatcher(None, stem, name).ratio(), multiplier)
        for multiplier, _, names in _VENUE_TIERS for name in names)
    return multiplier if ratio >= 0.88 else None


def _venue_tier_multiplier(cand: dict) -> float:
    """Classify venue authority; identifiers rescue unknown/abbreviated venues."""
    doi, venue = (cand.get("doi") or "").strip().lower(), cand.get("venue") or ""
    venue_multiplier = _venue_name_multiplier(venue)
    if doi.startswith("10.5194/egusphere-egu") or venue_multiplier == 0.5:
        return 0.5
    # A broad publisher DOI (e.g. 10.1029/) must not demote an explicitly
    # recognized S-tier journal such as Reviews of Geophysics.
    for multiplier, prefixes, _ in _VENUE_TIERS:
        if doi.startswith(prefixes) or venue_multiplier == multiplier:
            return multiplier
    if (venue.strip() and not (doi or cand.get("arxiv_id") or cand.get("inspire_record_id"))
            and (cand.get("citation_count") or 0) < 5):
        return 0.5
    return 1.0


_AUTHORITY_UNDATED_AGE = 10
_AUTHORITY_MAX_AGE = 30
_AUTHORITY_POOL_DEPTH = 100
_KG_TOP_PROTECT = 15


def _authority_value(cand: dict, now_year: int) -> float:
    cites, year = cand.get("citation_count") or 0, cand.get("year")
    if isinstance(year, int) and not isinstance(year, bool) and year > 0:
        age = min(_AUTHORITY_MAX_AGE, max(1, now_year - year))
    else:
        age = _AUTHORITY_UNDATED_AGE
    return (cites / age) * _venue_tier_multiplier(cand)


def _authority_reorder(scored, *, now_year=None, enabled=None, lam=None, k=60):
    """Pool-only RRF blend of dense relevance rank and age-normalized authority.

The prior can move a candidate across score bands; it cannot admit one rejected
by the gate. Explicit recency/importance hints bypass it in the return stage.
"""
    enabled = SEARCH_AUTHORITY_PRIOR if enabled is None else enabled
    lam = SEARCH_AUTHORITY_LAMBDA if lam is None else lam
    if not enabled or len(scored) < 2 or lam == 0:
        return scored
    now_year = now_year or datetime.now(timezone.utc).year
    ranks, previous, rank = [], None, -1
    for score, _ in scored:
        if score != previous:
            rank += 1
            previous = score
        ranks.append(rank)
    order = sorted(range(len(scored)), key=lambda i: -_authority_value(scored[i][1], now_year))
    authority = {i: pos for pos, i in enumerate(order)}
    return [scored[i] for i in sorted(
        range(len(scored)), key=lambda i: -(1 / (k + ranks[i]) + lam / (k + authority[i])))]


def _derive_source_health(degraded_map, n_terms):
    unconfigured = [b for b in BACKENDS if b in UNCONFIGURED_CHECK and not UNCONFIGURED_CHECK[b]()]
    degraded = [b for b in BACKENDS if b not in unconfigured and n_terms > 0
                and degraded_map.get(b, 0) == n_terms]
    return unconfigured, degraded


def YEAR_DROP(paper, year_min, year_max):
    year = paper.get("year")
    return isinstance(year, int) and (
        (year_min is not None and year < year_min) or (year_max is not None and year > year_max))


@dataclass(frozen=True)
class PlannedSearch:
    intent: dict
    terms: list[str]
    filters: dict
    limit: int
    ranking_hint: str
    query: str = ""

    @property
    def kg_concepts(self) -> list[str]:
        return self.intent.get("kg_concepts") or self.terms

    def year_eligible(self, node):
        return not YEAR_DROP(node, self.filters.get("year_min"), self.filters.get("year_max"))


@dataclass
class RetrievedSearch:
    external: list[dict]
    local: list
    degraded: dict
    kg_ranks: list[str] = field(default_factory=list)


@dataclass
class FusedSearch:
    external: list[dict]
    local: list[dict]
    vouched: list[dict]
    canonical_keys: dict[str, str]


@dataclass
class ReturnSelection:
    scored: list[tuple[float, dict]]
    projections: dict
    registration: list[dict]
    ingest_dropped: int
    return_dropped: int


def _normalized_ids(node: dict) -> tuple[str, str, str]:
    """Return ``(norm_doi, base_arxiv, version_suffix)`` for exact identity matching."""
    from ..store import _normalize_arxiv, _normalize_doi
    doi = _normalize_doi(node.get("doi") or "")
    raw_arxiv = (node.get("arxiv_id") or "").strip()
    base_arxiv = _normalize_arxiv(raw_arxiv)
    ver_match = re.search(r"v\d+$", raw_arxiv, re.I) if base_arxiv else None
    version = ver_match.group().lower() if ver_match else ""
    return doi, base_arxiv, version


def _exact_identity_axes(node: dict) -> list[str]:
    """Return all normalized exact identifier axes (DOI, versioned/base arXiv)."""
    doi, base_arxiv, version = _normalized_ids(node)
    axes: list[str] = []
    if doi:
        axes.append("doi:" + doi)
    if base_arxiv:
        if version:
            axes.append("arxiv:" + base_arxiv + version)
        axes.append("arxiv:" + base_arxiv)
    return axes


def _nodes_compatible(existing: dict, incoming: dict) -> bool:
    """True iff no populated exact identifier conflicts between the two records."""
    e_doi, e_arxiv, e_ver = _normalized_ids(existing)
    i_doi, i_arxiv, i_ver = _normalized_ids(incoming)
    if e_doi and i_doi and e_doi != i_doi:
        return False
    if e_arxiv and i_arxiv:
        if e_arxiv != i_arxiv:
            return False
        if e_ver and i_ver and e_ver != i_ver:
            return False
    return True


def _identity_key(node):
    """Use the catalog's identifier normalization without merging different DOIs."""
    from ..store import _normalize_arxiv, _normalize_doi
    if not (node.get("doi") or node.get("arxiv_id")):
        if node.get("_source_origin") == "library":
            return "record:" + node["key"]
        return _node_key(node)
    normalized = dict(node)
    normalized["doi"] = _normalize_doi(node.get("doi") or "")
    normalized["arxiv_id"] = _normalize_arxiv(node.get("arxiv_id") or "")
    version = re.search(r"v\d+$", node.get("arxiv_id") or "", re.I)
    if version:
        normalized["arxiv_id"] += version.group().lower()
        if not normalized["doi"]:
            return "arxiv:" + normalized["arxiv_id"]
    return _node_key(normalized)


def _merge_observations(hit, node):
    if node.get("source_identities"):
        hit["source_identities"] = [o.model_dump() for o in merge_source_identities(
            hit.get("source_identities"), node["source_identities"])]
    fills = merge_inspire_fields(hit, node)
    if fills:
        hit.update(fills)
        hit.setdefault("_inspire_metadata", []).extend(
            {field: donor.get(field) for field in ("title", "authors", "year", "doi", "arxiv_id")}
            for donor in node.get("_inspire_metadata") or [node])


def _snapshot_paper(paper):
    """Build an isolated shallow copy of ``paper`` and its mutable list fields.

    Avoids deep-copying nested Pydantic structures across ~30k catalog records
    while completely isolating search serving from concurrent live mutations.
    """
    return paper.model_copy(
        deep=False,
        update={
            "authors": list(paper.authors),
            "inspire_document_urls": list(paper.inspire_document_urls),
            "source_identities": list(paper.source_identities),
            "publication_types": list(paper.publication_types),
            "identity_route_rejections": list(paper.identity_route_rejections),
        },
    )


def _safe_donor_for_projection(projected, donor: dict, live_view) -> dict:
    """Return a donor copy stripped of identifiers conflicting with or owned by another record."""
    from ..store import _normalize_arxiv, _normalize_doi

    safe = dict(donor)
    token_fn = getattr(live_view, "_identifier_token", None)
    owner_fn = getattr(live_view, "_identifier_owner", None)
    for field_name in _IDENTIFIERS:
        raw = safe.get(field_name) or ""
        if not raw:
            continue
        cur = getattr(projected, field_name, "") or ""
        if field_name == "doi":
            if cur and _normalize_doi(cur) != _normalize_doi(raw):
                safe[field_name] = ""
                continue
        elif field_name == "arxiv_id":
            if cur and _normalize_arxiv(cur) != _normalize_arxiv(raw):
                safe[field_name] = ""
                continue
        elif cur and cur != raw:
            safe[field_name] = ""
            continue
        if callable(token_fn) and callable(owner_fn):
            token = token_fn(field_name, raw, safe.get("source") or "")
            if token:
                owner = owner_fn(field_name, token)
                if owner is not None and owner.key != projected.key:
                    safe[field_name] = ""
    return safe


def _consolidate_vouched_donor(known_node: dict, donors: list[dict]) -> dict:
    """Merge folded external observations for one known library record into a single payload."""
    consolidated: dict = {"_registration_known_key": known_node["key"]}
    merged_identities = []
    for donor in donors:
        donor_ids = normalized_source_identities(donor, donor.get("source") or "")
        if donor_ids:
            merged_identities = merge_source_identities(merged_identities, donor_ids)
        inspire_fills = merge_inspire_fields(consolidated, donor)
        if inspire_fills:
            consolidated.update(inspire_fills)
            consolidated.setdefault("_inspire_metadata", []).extend(
                {f: d.get(f) for f in ("title", "authors", "year", "doi", "arxiv_id")}
                for d in (donor.get("_inspire_metadata") or [donor])
            )
        for field_name in _METADATA - {"source_identities", "inspire_record_id",
                                       "inspire_document_urls", "_inspire_metadata"}:
            val = donor.get(field_name)
            if val in (None, "", [], 0):
                continue
            cur = consolidated.get(field_name)
            if field_name == "citation_count":
                consolidated["citation_count"] = max(int(cur or 0), int(val or 0))
            elif field_name == "is_review":
                consolidated["is_review"] = bool(cur) or bool(val)
            elif not cur:
                if field_name in _IDENTIFIERS and not _nodes_compatible(known_node, {field_name: val}):
                    continue
                consolidated[field_name] = copy.deepcopy(val)
    if merged_identities:
        consolidated["source_identities"] = [obs.model_dump() for obs in merged_identities]
    return consolidated


def _relevance_key(scored, now_year):
    score, node = scored
    year = node.get("year")
    return (-score, -node.get("folded_vouch", 0), -node.get("external_rrf_score", 0),
            -_authority_value(node, now_year), -(year if isinstance(year, int) else 0),
            node.get("key") or _identity_key(node))


def _matches_entity(node, aliases):
    text = " ".join([node.get("title") or "", node.get("abstract") or "",
                     " ".join(node.get("authors") or [])])
    return any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I) for alias in aliases)


def preserve_named_coverage(scored, groups, limit, trace):
    """Reserve high-confidence entity witnesses within the declared result budget.

One candidate may cover several entities. When the budget cannot cover all of
them, query order determines priority and the trace records unmet coverage.
"""
    if len(groups) < 2 or limit <= 0:
        return scored
    masks = [sum(1 << g for g, aliases in enumerate(groups) if score >= 0.7
                 and _matches_entity(node, aliases)) for score, node in scored]
    available = 0
    for mask in masks:
        available |= mask
    # The planner bounds named entities. Dynamic programming retains one best
    # witness set per (coverage, slots), including joint witnesses; protecting a
    # particular paper would incorrectly declare a feasible cover impossible.
    states = {(0, 0): ()}

    def cost(indices):
        return sum(i >= limit for i in indices), sum(indices), indices

    for i, candidate_mask in enumerate(masks):
        if not candidate_mask:
            continue
        for (mask, count), indices in list(states.items()):
            combined = mask | candidate_mask
            if combined == mask or count >= min(limit, len(groups)):
                continue
            key, proposal = (combined, count + 1), indices + (i,)
            if key not in states or cost(proposal) < cost(states[key]):
                states[key] = proposal

    def preference(item):
        (mask, count), indices = item
        return (tuple(-bool(mask & (1 << g)) for g in range(len(groups))),
                cost(indices)[0], count, cost(indices)[1:])

    (covered, _), witnesses = min(states.items(), key=preference)
    selected = set(witnesses)
    for i in range(len(scored)):
        if len(selected) >= min(limit, len(scored)):
            break
        selected.add(i)
    for i in sorted(selected):
        if i >= limit:
            trace.record("ranking", candidate=scored[i][1], decision="reserved", reason="named_entity_coverage")
    for g, aliases in enumerate(groups):
        if not covered & (1 << g):
            trace.record("ranking", entity=aliases[0], decision="unmet",
                         reason="budget" if available & (1 << g) else "no_high_confidence_candidate")
    return [pair for i, pair in enumerate(scored) if i in selected] + [
        pair for i, pair in enumerate(scored) if i not in selected]


@dataclass
class SearchPipeline:
    library: object
    registration_queue: object
    llm_factory: Callable
    parse: Callable
    retrieve: Callable
    judge_ingest: Callable
    judge_return: Callable
    project: Callable
    reframe_hint: Callable
    progress: Callable
    authority_enabled: bool = SEARCH_AUTHORITY_PRIOR
    authority_weight: float = SEARCH_AUTHORITY_LAMBDA

    async def intent(self, query, llm, trace):
        await self.progress(1, "parsing intent")
        with observe_intent(trace.record_intent):
            async with concurrency.llm_sem:
                return await asyncio.to_thread(self.parse, query, llm)

    def plan(self, intent, trace, *, query=""):
        plan = PlannedSearch(intent, intent["search_terms"], intent.get("filters") or {},
                             intent.get("limit_suggested") or int(os.environ.get("PAPER_SEARCH_DEFAULT_LIMIT", "15")),
                             intent.get("ranking_hint", "by_relevance"), query=query)
        trace.record("plan", terms=len(plan.terms), kg_concepts=len(plan.kg_concepts), limit=plan.limit,
                     year_min=plan.filters.get("year_min"), year_max=plan.filters.get("year_max"),
                     ranking_hint=plan.ranking_hint)
        return plan

    def _local_snapshot(self):
        # Shallow-copy each Paper + its mutable list fields so concurrent
        # background writes never mutate the frozen serving snapshot, without
        # paying full deep-copy cost across ~30k nested Pydantic models.
        lock = getattr(self.library, "_merge_execution_lock", None)
        if lock is not None:
            with lock:
                local = [_snapshot_paper(p) for p in self.library.all_papers(include_quarantined=True)]
                view = _view(self.library, {p.key: p for p in local}, deep_copy=False, frozen=True)
        else:
            local = [_snapshot_paper(p) for p in self.library.all_papers(include_quarantined=True)]
            view = _view(self.library, {p.key: p for p in local}, deep_copy=False, frozen=True)
        return local, view

    async def retrieval(self, plan, trace):
        await self.progress(2, "searching external sources")
        local, view = await asyncio.to_thread(self._local_snapshot)
        with observe_source_gates(trace.record_backend):
            async with network_slot(concurrency.search_network_sem, self.library, actor="search"):
                (external, degraded), kg_ranks = await asyncio.gather(
                    self.retrieve(plan.terms, year_min=plan.filters.get("year_min"),
                                  year_max=plan.filters.get("year_max"), ranking_hint=plan.ranking_hint),
                    asyncio.to_thread(kg_local_search, plan.query, plan, view,
                                      embedding_loop=asyncio.get_running_loop()))
        result = RetrievedSearch(copy.deepcopy(external), local, degraded, kg_ranks)
        trace.record("retrieval", external=len(external), local=len(local), kg=len(kg_ranks))
        return result

    def fusion(self, plan, retrieved, trace):
        seen, local_by_key, canonical_keys = {}, {}, {}
        seen_axes = collections.defaultdict(list)
        primary_witnesses = collections.defaultdict(list)
        alias_witnesses = collections.defaultdict(list)
        excluded_identities = collections.defaultdict(list)
        bibliography = collections.defaultdict(list)
        snapshot = {p.key: p for p in retrieved.local}
        query_tokens = {token for term in plan.terms for token in _tokenize(term)}
        kg_trace_keys = set(retrieved.kg_ranks)
        local_trace_counts = collections.Counter()

        def register_seen(hit, witness=None, *, allow_existing_axes=True, is_alias=False):
            w = witness if witness is not None else hit
            witness_list = alias_witnesses[id(hit)] if is_alias else primary_witnesses[id(hit)]
            if not any(existing_w is w for existing_w in witness_list):
                witness_list.append(w)
            key_id = _identity_key(w)
            if allow_existing_axes or key_id not in seen:
                seen[key_id] = hit
            for axis in _exact_identity_axes(w):
                bucket = seen_axes[axis]
                if not allow_existing_axes and bucket:
                    continue
                if not any(existing_hit is hit and existing_w is w for existing_hit, existing_w in bucket):
                    bucket.append((hit, w))

        def lookup_seen(node):
            axes = _exact_identity_axes(node)
            if axes:
                touched = {}
                for axis in axes:
                    for hit, _witness in seen_axes.get(axis, ()):
                        touched[id(hit)] = hit
                if not touched:
                    return None, False, False
                if len(touched) > 1:
                    # Different axes resolve to different existing records;
                    # refuse to fold across distinct identities.
                    return None, True, False
                target = next(iter(touched.values()))
                if all(_nodes_compatible(w, node) for w in primary_witnesses[id(target)]):
                    return target, False, False
                # An explicit Merge.ALIAS record may legitimately carry a
                # different DOI/arXiv than its canonical target.
                if any(
                    _nodes_compatible(aw, node)
                    and bool(set(axes).intersection(_exact_identity_axes(aw)))
                    for aw in alias_witnesses[id(target)]
                ):
                    return target, False, True
                return None, True, False
            hit = seen.get(_identity_key(node))
            if hit is None:
                return None, False, False
            if all(_nodes_compatible(w, node) for w in primary_witnesses[id(hit)]):
                return hit, False, False
            return None, True, False

        def record_excluded(rec_dict, key):
            entry = (key, rec_dict)
            excluded_identities[_identity_key(rec_dict)].append(entry)
            for axis in _exact_identity_axes(rec_dict):
                excluded_identities[axis].append(entry)

        def lookup_excluded(node):
            keys_to_check = [_identity_key(node), *_exact_identity_axes(node)]
            for idx_key in keys_to_check:
                for canonical_key, rec_dict in excluded_identities.get(idx_key, ()):
                    if _nodes_compatible(rec_dict, node):
                        return canonical_key
            return None

        def trace_local(paper, canonical, reason, **fields):
            # Trace relevance only: all catalog identity/exclusion state below
            # must still be populated, even after a diagnostic reaches its cap.
            if local_trace_counts[reason] >= LIB_CAP * 2:
                return
            records = (paper, canonical) if canonical is not None and canonical is not paper else (paper,)
            if not any(record.key in kg_trace_keys or query_tokens.intersection(
                    _tokenize(_paper_text(paper_to_dict(record)))) for record in records):
                return
            local_trace_counts[reason] += 1
            trace.record("fusion", key=paper.key, reason=reason, **fields)

        with trace.batch():
            # Register identified versions first so an identifier-poor observation
            # cannot bridge two different DOIs or displace the identified record.
            for paper in sorted(retrieved.local, key=lambda p: (-bool(p.doi), -bool(p.arxiv_id), p.key)):
                path, canonical = set(), paper
                while canonical.acquisition_state.alias_key and canonical.key not in path:
                    path.add(canonical.key)
                    canonical = snapshot.get(canonical.acquisition_state.alias_key)
                    if canonical is None:
                        break
                if canonical is None or canonical.key in path:
                    trace_local(paper, canonical, "unresolved_alias", decision="dropped")
                    continue
                if serving_excluded(canonical):
                    record_excluded(paper_to_dict(paper), canonical.key)
                    record_excluded(paper_to_dict(canonical), canonical.key)
                    trace_local(paper, canonical, "serving_excluded", decision="dropped")
                    continue
                node = paper_to_dict(canonical)
                existing, _conflict, _via_alias = lookup_seen(node)
                bib_key = serving_bibliographic_key(node)
                matches = bibliography.get(bib_key, [])
                if (existing is None and not (node.get("doi") or node.get("arxiv_id"))
                        and len(matches) == 1):
                    existing = matches[0]
                    register_seen(existing, node)
                if existing is not None:
                    canonical_keys[paper.key] = existing["key"]
                    if paper.key != canonical.key:
                        register_seen(existing, paper_to_dict(paper), is_alias=True)
                    trace_local(paper, snapshot[existing["key"]], "local_duplicate", candidate=paper_to_dict(paper),
                                decision="folded", canonical_key=existing["key"])
                    continue
                register_seen(node)
                canonical_keys[paper.key] = node["key"]
                if paper.key != canonical.key:
                    register_seen(node, paper_to_dict(paper), is_alias=True)
                local_by_key[node["key"]] = node
                if bib_key is not None and (node.get("doi") or node.get("arxiv_id")):
                    bibliography[bib_key].append(node)
            corpus = [node for node in local_by_key.values() if plan.year_eligible(node)]
            local_ranks = bm25_per_term_ranklists(corpus, plan.terms)
            # Resolve KG evidence against this same canonical, year-eligible snapshot.
            # An alias or several matching chunks can grant only one vote per track.
            eligible_keys = {node["key"] for node in corpus}
            kg_keys = dict.fromkeys(canonical_keys.get(key, key) for key in retrieved.kg_ranks)
            kg_ranks = [local_by_key[key] for key in kg_keys if key in eligible_keys]
            buckets = collections.defaultdict(list)
            # Secondary identifiers/title and explicit aliases use preview's
            # version-compatible lookup, bound to this request's catalog snapshot.
            exclusion_view = (_view(self.library, snapshot, deep_copy=False, frozen=True)
                              if excluded_identities and retrieved.external else None)
            for node in retrieved.external:
                term, rank = node.get("term_idx", 0), node.get("rank", 0)
                excluded_key = lookup_excluded(node)
                if not excluded_key and exclusion_view is not None:
                    matched = _find(exclusion_view, node)
                    if matched is not None and serving_excluded(matched):
                        excluded_key = matched.key
                if excluded_key:
                    trace.record("fusion", candidate=node, key=excluded_key, arm="external",
                                 decision="dropped", reason="serving_excluded")
                    continue
                hit, has_conflict, via_alias = lookup_seen(node)
                if hit is None:
                    hit = node
                    hit["term_ranks"] = {}
                    register_seen(hit, allow_existing_axes=not has_conflict)
                else:
                    _merge_observations(hit, node)
                    register_seen(hit, node, is_alias=via_alias)
                    register_seen(hit)
                    trace.record("fusion", candidate=node, decision="folded", canonical_key=hit.get("key"))
                hit["best_external_rank"] = min(hit.get("best_external_rank", rank), rank)
                hit["external_rrf_score"] = hit.get("external_rrf_score", 0) + 1 / (RRF_K + rank)
                hit["_external_hit_count"] = hit.get("_external_hit_count", 0) + 1
                if hit.get("_source_origin") == "library":
                    hit["folded_vouch"] = hit.get("folded_vouch", 0) + 1
                    hit.setdefault("_folded_donors", []).append(node)
                    continue
                votes = hit.setdefault("term_ranks", {})
                if term not in votes:
                    buckets[term].append(hit)
                votes[term] = min(votes.get(term, rank), rank)
            external_ranks = [sorted(buckets.get(t, []), key=lambda n: (n["term_ranks"][t], _identity_key(n)))
                              for t in range(len(plan.terms))]
            ext_rrf = rrf_fuse(external_ranks, _identity_key)
            external_ranks = reorder_ranklists_by_rrf(external_ranks, ext_rrf, _identity_key)
            lib_key = lambda node: node["key"]
            ranked_lists = [("local", ranks) for ranks in local_ranks] + [("kg", kg_ranks)]
            pool_ranks = [ranks for _, ranks in ranked_lists]
            base_rrf = rrf_fuse(pool_ranks, lib_key)
            lib_rrf = dict(base_rrf)
            authority_active = (self.authority_enabled and self.authority_weight > 0
                                and plan.ranking_hint not in ("by_recency", "by_importance"))
            now_year = datetime.now(timezone.utc).year
            authority_applied = False
            if authority_active:
                top_cands = {node["key"] for rl in pool_ranks for node in rl[:_AUTHORITY_POOL_DEPTH]}
                scored_cands = [k for k in top_cands if _authority_value(local_by_key[k], now_year) > 0]
                if scored_cands:
                    authority_applied = True
                    auth_order = sorted(scored_cands, key=lambda k: (
                        -_authority_value(local_by_key[k], now_year), -base_rrf.get(k, 0.0), k))
                    auth_rank = {k: i for i, k in enumerate(auth_order)}
                    for k in auth_order:
                        lib_rrf[k] = lib_rrf.get(k, 0.0) + (
                            1.5 * self.authority_weight / (RRF_K + auth_rank[k]))
            if authority_applied:
                reordered_local = reorder_ranklists_by_rrf(local_ranks, lib_rrf, lib_key)
                if kg_ranks:
                    kg_base = reorder_ranklists_by_rrf([kg_ranks], base_rrf, lib_key)[0]
                    kg_head = kg_base[:_KG_TOP_PROTECT]
                    kg_tail = (
                        reorder_ranklists_by_rrf([kg_base[_KG_TOP_PROTECT:]], lib_rrf, lib_key)[0]
                        if len(kg_base) > _KG_TOP_PROTECT else []
                    )
                    reordered_kg = kg_head + kg_tail
                else:
                    reordered_kg = []
                pool_ranks = reordered_local + [reordered_kg]
            else:
                pool_ranks = reorder_ranklists_by_rrf(pool_ranks, lib_rrf, lib_key)
            for ranklist in external_ranks:
                for node in ranklist:
                    node["_rrf"] = ext_rrf.get(_identity_key(node), 0)
            external = round_robin(external_ranks, EXT_CAP, _identity_key)
            local = round_robin(pool_ranks, LIB_CAP, lib_key)
            selected = {n["key"] for n in local}
            vouched = [n for n in local_by_key.values() if n.get("folded_vouch")]
            also_include = [n for n in vouched if n["key"] not in selected and plan.year_eligible(n)]
            if authority_active:
                v_auth_order = sorted(also_include, key=lambda n: (
                    -_authority_value(n, now_year), -n["external_rrf_score"],
                    -n["folded_vouch"], -(n.get("year") or _YEAR_NEUTRAL), n["key"]))
                v_auth_rank = {n["key"]: i for i, n in enumerate(v_auth_order)}
                also_include.sort(key=lambda n: (
                    -(n["external_rrf_score"] + 2.0 * self.authority_weight / (RRF_K + v_auth_rank[n["key"]])),
                    -n["folded_vouch"], -_authority_value(n, now_year),
                    -(n.get("year") or _YEAR_NEUTRAL), n["key"]))
            else:
                also_include.sort(key=lambda n: (
                    -n["folded_vouch"], -n["external_rrf_score"], -(n.get("citation_count") or 0),
                    -(n.get("year") or _YEAR_NEUTRAL), n["key"]))
            local.extend(also_include[:ALSO_INCLUDE_CAP])
            selected = {n["key"] for n in local}
            ext_selected = {_identity_key(n) for n in external}
            contenders = selected | {node["key"] for ranks in pool_ranks for node in ranks[:LIB_CAP * 2]}
            for node in {id(n): n for n in seen.values()}.values():
                if (node.get("_source_origin") == "library" and not node.get("_external_hit_count")
                        and not node.get("folded_vouch") and node.get("key") not in contenders):
                    continue
                trace.record("fusion", candidate=node, key=node.get("key"),
                             decision="pooled" if node.get("key") in selected or _identity_key(node) in ext_selected else "dropped",
                             reason="pool_selection", folded_vouch=node.get("folded_vouch", 0),
                             external_rrf_score=node.get("external_rrf_score"), best_external_rank=node.get("best_external_rank"))
            trace.record("fusion", external_pool=len(external), local_pool=len(local))
            return FusedSearch(external, local, vouched, canonical_keys)

    def _prepare_return_pool(self, plan, fused, judgments, trace):
        """Run synchronous eligibility lookups and isolated previews off-loop."""
        with self.library._merge_execution_lock:
            live_view = _view(self.library, self.library._papers, deep_copy=False, frozen=True)
            live_view._identity_reservations = set(self.library._identity_reservations)
        eligible = []
        with trace.batch():
            for idx, node in enumerate(fused.external):
                candidate = copy.deepcopy(node)
                candidate["_registration"] = judgments.get(idx)
                known = live_view.find(doi=candidate.get("doi") or "", arxiv_id=candidate.get("arxiv_id") or "",
                                       title=candidate.get("title") or "") or live_view._find_preprint_journal_dup(candidate)
                rejection = registration_rejection(candidate, known_identity=known is not None)
                trace.record("gate", candidate=candidate, decision="skipped" if rejection else "eligible", reason=rejection)
                if not rejection:
                    eligible.append(candidate)
        # Deduplicate folded observations to one consolidated payload per known
        # library key and bound the combined submission to MAX_CANDIDATES while
        # prioritizing brand-new eligible discoveries so vouched observations
        # never overflow RegistrationQueue and drop new papers.
        vouched_ranked = sorted(
            fused.vouched,
            key=lambda n: (-n.get("folded_vouch", 0), -n.get("external_rrf_score", 0), n["key"]),
        )
        vouched_registration = []
        seen_vouched_keys: set[str] = set()
        for n in vouched_ranked:
            key = n["key"]
            if key in seen_vouched_keys:
                continue
            seen_vouched_keys.add(key)
            donors = n.get("_folded_donors") or [n]
            vouched_registration.append(_consolidate_vouched_donor(n, donors))
        eligible_budget = eligible[:MAX_CANDIDATES]
        remaining_slots = max(0, MAX_CANDIDATES - len(eligible_budget))
        registration = eligible_budget + vouched_registration[:remaining_slots]
        external, projections = preview_candidates(live_view, eligible, canonical_keys=fused.canonical_keys)
        # Preview may resolve a new identity to a local row suppressed by fusion.
        # Its canonical snapshot already carries the preview hits' rank evidence;
        # combine that with direct external folds before replacing the local row.
        local_evidence = {n["key"]: n for n in fused.local + fused.vouched}
        filtered_external = []
        for node in external:
            projected = projections.get(node["key"])
            if projected is not None and serving_excluded(projected):
                projections.pop(node["key"], None)
                trace.record("gate", candidate=node, key=node["key"],
                             decision="dropped", reason="serving_excluded")
                continue
            # Later observations can fill an unknown year on the isolated
            # projection after its first scoring snapshot was captured.
            node["year"] = projections[node["key"]].year
            donor = local_evidence.get(node["key"])
            if donor is not None:
                for field in ("folded_vouch", "external_rrf_score"):
                    node[field] = node.get(field, 0) + donor.get(field, 0)
                if "best_external_rank" in donor:
                    node["best_external_rank"] = min(node["best_external_rank"], donor["best_external_rank"])
            filtered_external.append(node)
        external = filtered_external
        external_keys = set(projections)
        local_pool = []
        for node in fused.local:
            key = node["key"]
            if key in external_keys:
                continue
            donors = node.get("_folded_donors")
            if donors:
                base = live_view.get(key)
                if base is not None:
                    projected = base.model_copy(deep=True)
                    for donor in donors:
                        safe_donor = _safe_donor_for_projection(projected, donor, live_view)
                        _fill(projected, safe_donor)
                    if serving_excluded(projected):
                        trace.record("gate", candidate=paper_to_dict(projected), key=key,
                                     decision="dropped", reason="serving_excluded")
                        continue
                    projections[key] = projected
                    enriched = paper_to_dict(projected)
                    for meta_field in (
                        "folded_vouch", "external_rrf_score",
                        "best_external_rank", "_external_hit_count",
                    ):
                        if meta_field in node:
                            enriched[meta_field] = node[meta_field]
                    local_pool.append(enriched)
                    continue
            local_pool.append(node)
        pool = local_pool + external
        pool = [n for n in pool if plan.year_eligible(n)]
        return pool, projections, registration

    async def return_gate(self, query, plan, fused, llm, trace):
        await self.progress(3, "judging new papers for ingest")
        judgments, ingest_dropped = await self.judge_ingest(fused.external, llm=llm)
        pool, projections, registration = await asyncio.to_thread(
            self._prepare_return_pool, plan, fused, judgments, trace
        )
        await self.progress(4, "scoring relevance")
        judgments, return_dropped = await self.judge_return(
            pool, query, search_terms=plan.terms, filters=plan.filters, llm=llm)
        scored = []
        for idx, node in enumerate(pool):
            judgment = judgments.get(idx)
            decision = "dropped" if judgment is None else "below_threshold" if judgment["score"] < RETURN_THRESHOLD else "accepted"
            trace.record("return_gate", candidate=node, key=node.get("key"), decision=decision,
                         score=judgment["score"] if judgment else None)
            if decision == "accepted":
                scored.append((judgment["score"], node))
        now_year = datetime.now(timezone.utc).year
        if plan.ranking_hint == "by_recency":
            scored.sort(key=lambda pair: (-pair[0], -(pair[1].get("year") or 0), pair[1]["key"]))
        elif plan.ranking_hint == "by_importance":
            scored.sort(key=lambda pair: (-pair[0], -(pair[1].get("citation_count") or 0), pair[1]["key"]))
        else:
            scored.sort(key=lambda pair: _relevance_key(pair, now_year))
            scored = _authority_reorder(scored, enabled=self.authority_enabled, lam=self.authority_weight)
        scored = preserve_named_coverage(scored, named_entity_groups(query, plan.intent), plan.limit, trace)
        return ReturnSelection(scored, projections, registration, ingest_dropped, return_dropped)

    async def serving(self, selection, plan, trace):
        await self.progress(5, "building response")
        results = []
        with trace.batch():
            for position, (_, candidate) in enumerate(selection.scored):
                trace.record("ranking", candidate=candidate, key=candidate.get("key"), position=position,
                             decision="returned" if position < plan.limit else "limit")
                if position >= plan.limit:
                    continue
                key = candidate["key"]
                paper = self.library.get(key)
                if paper is not None:
                    if serving_excluded(paper):
                        continue
                    record = self.project(selection.projections.get(key, paper), self.library)
                else:
                    record = {field: candidate.get(field) or default for field, default in (
                        ("title", ""), ("authors", []), ("year", None), ("venue", ""),
                        ("abstract", ""), ("doi", ""), ("arxiv_id", ""), ("citation_count", 0))}
                    record.update(key=key, text_status="pending", source_identities=normalized_source_identities(
                        candidate, candidate.get("source") or ""))
                if not plan.year_eligible(record):
                    continue
                results.append(record)
                trace.record("serving", candidate=record, key=record["key"], position=position, decision="served",
                             text_status=record.get("text_status"), text_available="text_path" in record)
            trace.record("serving", returned=len(results), eligible=len(selection.registration))
        return results

    async def run(self, query, *, service_fields=None):
        query = (query or "").strip()
        if not query:
            return {"status": "error", "results": [], "message": "empty query — pass a natural-language description of what you're working on and what literature you want to find"}
        trace, llm = ServingTrace(self.library), self.llm_factory()
        try:
            intent = await self.intent(query, None, trace)
        except ValueError:
            return {"status": "error", "results": [], "message": "could not parse the query into a search plan — try a clearer natural-language description of what you want to find",
                    "reframe_hint": self.reframe_hint()}
        plan = self.plan(intent, trace, query=query)
        retrieved = await self.retrieval(plan, trace)
        fused = await asyncio.to_thread(self.fusion, plan, retrieved, trace)
        selection = await self.return_gate(query, plan, fused, llm, trace)
        results = await self.serving(selection, plan, trace)
        unconfigured, degraded = _derive_source_health(retrieved.degraded, len(plan.terms))
        response = {"status": "ok", "results": results, **(service_fields or {}), "intent_parsed": {
            "search_terms": plan.terms,
            "kg_concepts": plan.kg_concepts,
            "filters_applied": {name: plan.filters.get(name) for name in ("year_min", "year_max", "citation_pref", "review_pref")},
            "limit_resolved": plan.limit, "sources_degraded": degraded, "sources_unconfigured": unconfigured,
            "judge_batches_dropped": {"ingest": selection.ingest_dropped, "return": selection.return_dropped},
            "reasoning": intent.get("reasoning", "")}}
        if not results and selection.ingest_dropped == 0 and selection.return_dropped == 0:
            response["reframe_hint"] = self.reframe_hint()
        try:
            self.registration_queue.submit(selection.registration, actor="search", reason=trace.trace_id, trace=trace)
        except Exception:
            logger.exception("registration submission failed (%s)", trace.trace_id)
            with trace.batch():
                for candidate in selection.registration[:120]:
                    trace.record("registration", candidate=candidate, decision="failed", reason="submission_failed")
        return response
