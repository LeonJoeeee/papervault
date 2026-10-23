"""Serving-stage regressions for the owner-approved S1 contract (#251)."""

import json
from types import SimpleNamespace

import pytest

from papervault.library import Library
from papervault.library.mcp import server as protocol
from papervault.library.models import Paper


def node(key, title, *, rank=0, term=0, cites=0, year=2013, doi=None):
    return dict(key=key, title=title, authors=["Tester, A."], year=year,
                abstract=title + ". Original detector measurements.",
                doi=doi or "10.1234/" + key, citation_count=cites,
                publication_types=["JournalArticle"], source="openalex",
                _source_origin="external", term_idx=term, rank=rank)


def add_record(lib, key, **data):
    paper = Paper(key=key, **{k: v for k, v in data.items() if k != "key"})
    lib._papers[key] = paper
    lib._reindex(paper)
    return paper


async def run_search(monkeypatch, lib, nodes, *, query="detector measurements",
                     terms=None, limit=2, scores=None, authority=False, hint="by_relevance", filters=None):
    seen = {}
    monkeypatch.setattr(protocol, "get_llm", lambda: SimpleNamespace())
    monkeypatch.setattr(protocol, "SEARCH_AUTHORITY_PRIOR", authority)
    monkeypatch.setattr(protocol, "parse_intent", lambda *a: {
        "search_terms": terms or ["detector measurements"], "filters": filters or {},
        "limit_suggested": limit, "ranking_hint": hint})

    async def retrieve(*a, **kw):
        return nodes, {}

    async def ingest(cands, **kw):
        return {i: dict(ingest_ok=True, llm_is_paper=True, tier="1A")
                for i in range(len(cands))}, 0

    async def judge(cands, *a, **kw):
        seen["pool"] = cands
        return {i: {"score": (scores or {}).get(c["doi"], 0.8)}
                for i, c in enumerate(cands)}, 0

    monkeypatch.setattr(protocol, "search_external_async", retrieve)
    monkeypatch.setattr(protocol, "judge_ingest", ingest)
    monkeypatch.setattr(protocol, "judge_return", judge)
    server = protocol.build_server(library=lib)
    reply = await server.call_tool("search_papers", {"query": query})
    result = json.loads("\n".join(c.text for c in reply))
    events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
    return result, seen, [e for e in events if e.get("event") == "serving_trace"]


async def test_fold_retains_external_rank_for_served_ties(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    # Identical relevance, votes, citations and year: external native rank must decide.
    early = node("ZOriginal", "Original detector measurements of energetic positrons", rank=0)
    late = node("AOther", "Other detector measurements of cosmic protons", rank=10)
    add_record(lib, **early)
    add_record(lib, **late)
    before = lib.index_path.read_bytes()
    result, seen, events = await run_search(monkeypatch, lib, [late, early], limit=1)
    assert [r["key"] for r in result["results"]] == ["ZOriginal"]
    folded = {c["key"]: c for c in seen["pool"]}
    assert folded["ZOriginal"]["best_external_rank"] == 0
    assert folded["ZOriginal"]["external_rrf_score"] > folded["AOther"]["external_rrf_score"]
    assert lib.index_path.read_bytes() == before
    assert any(e["stage"] == "fusion" and e.get("decision") == "folded" for e in events)
    assert any(e["stage"] == "serving" and e.get("key") == "ZOriginal" for e in events)


async def test_duplicate_library_records_take_one_served_slot(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    original = node("Collaboration2013", "Precision detector measurements of cosmic positron fraction")
    add_record(lib, **original)
    add_record(lib, **dict(original, key="Collaboration2013a"))
    companion = node("Adriani2009", "PAMELA detector measurements of positron abundance", term=1)
    result, seen, _ = await run_search(monkeypatch, lib, [original, companion], limit=2,
                                      terms=["positron fraction", "PAMELA positron abundance"])
    assert len({r["doi"] for r in result["results"]}) == 2
    assert [c["doi"] for c in seen["pool"]].count(original["doi"]) == 1


async def test_distinct_versions_with_same_title_are_not_collapsed(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    a = node("Preprint", "Precision detector measurements of cosmic positron fraction")
    b = node("Published", a["title"], rank=1)
    add_record(lib, **a)
    add_record(lib, **b)
    result, _, _ = await run_search(monkeypatch, lib, [a, b])
    assert {r["key"] for r in result["results"]} == {"Preprint", "Published"}


async def test_external_distinct_dois_with_same_title_survive_preview(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    a = node("Preprint", "Precision detector measurements of cosmic positron fraction")
    b = node("Published", a["title"], rank=1)
    result, _, _ = await run_search(monkeypatch, lib, [a, b])
    assert {r["doi"] for r in result["results"]} == {a["doi"], b["doi"]}


@pytest.mark.parametrize("ids", [("1001.1234v1", "1002.1234v1"), ("1001.1234", "1002.1234")])
async def test_external_distinct_arxiv_bases_survive_preview(tmp_path, monkeypatch, ids):
    lib = Library(tmp_path)
    a = node("First", "Precision detector measurements of cosmic positron fraction")
    b = node("Second", a["title"], rank=1)
    a.update(doi="", arxiv_id=ids[0])
    b.update(doi="", arxiv_id=ids[1])
    result, _, _ = await run_search(monkeypatch, lib, [a, b])
    assert {r["arxiv_id"] for r in result["results"]} == set(ids)


async def test_identifier_poor_duplicate_does_not_take_a_second_slot(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    original = node("Collaboration2013", "The cosmic-ray positron energy spectrum measured by PAMELA")
    alias = dict(original, key="Collaboration2013a", doi="", arxiv_id="",
                 title="Cosmic-Ray Positron Energy Spectrum Measured by PAMELA")
    for candidate in [original, alias]:
        add_record(lib, **candidate)
    # Make the named topic rare enough for positive BM25 weights, as in the
    # real catalog; a two-record corpus would accidentally hide both rows.
    for i in range(5):
        add_record(lib, **node(f"Other{i}", f"Unrelated photospheric magnetic reconnection study {i}"))
    other = node("Aguilar2013", "Alpha Magnetic Spectrometer positron fraction measurement", term=1)
    result, seen, _ = await run_search(monkeypatch, lib, [original, other], limit=3,
                                      terms=["PAMELA positron spectrum", "AMS-02 positron fraction"])
    assert len(result["results"]) == 2
    assert len(seen["pool"]) == 2
    assert "Collaboration2013a" not in {r["key"] for r in result["results"]}


@pytest.mark.parametrize("second_hit", [None, "identified", "identifier_poor"])
@pytest.mark.parametrize("limit", [1, 2])
async def test_preview_cannot_reintroduce_suppressed_twin_under_slot_pressure(
        tmp_path, monkeypatch, second_hit, limit):
    lib = Library(tmp_path)
    original = node("Collaboration2013", "The cosmic-ray positron energy spectrum measured by PAMELA")
    original["arxiv_id"] = "1308.0133"
    twin = dict(original, key="Collaboration2013a", doi="", arxiv_id="",
                title="Cosmic-Ray Positron Energy Spectrum Measured by PAMELA")
    for candidate in [original, twin]:
        add_record(lib, **candidate)
    for i in range(5):
        add_record(lib, **node(f"Other{i}", f"Unrelated photospheric magnetic reconnection study {i}"))
    # Exercise the real preview lookup: fusion suppresses this row, but the
    # identifier-poor external observation still resolves to its exact title.
    poor_hit = dict(twin, rank=0)
    assert lib.find(title=poor_hit["title"]).key == "Collaboration2013a"
    external = [poor_hit,
                node("Competitor", "Other detector measurements of energetic particles", rank=1),
                node("Extra", "Further detector measurements of cosmic particles", rank=2)]
    if second_hit == "identified":
        external.append(dict(original, rank=4))
    elif second_hit == "identifier_poor":
        external.append(dict(original, doi="", arxiv_id="", rank=4))
    before = {p.key: p.model_dump() for p in lib.all_papers()}
    index_before = lib.index_path.read_bytes()
    result, seen, events = await run_search(
        monkeypatch, lib, external, limit=limit, terms=["PAMELA positron spectrum"])
    pool = {c["key"]: c for c in seen["pool"]}
    assert "Collaboration2013a" not in pool
    assert len(pool) == 3 > limit
    canonical = pool["Collaboration2013"]
    assert canonical["folded_vouch"] == (2 if second_hit else 1)
    assert canonical["best_external_rank"] == 0
    assert canonical["external_rrf_score"] == pytest.approx(
        1 / 60 + (1 / 64 if second_hit else 0))
    assert canonical["doi"] == original["doi"]
    assert canonical["arxiv_id"] == "1308.0133"
    served = [r["key"] for r in result["results"]]
    assert served == ["Collaboration2013", "doi:10.1234/competitor"][:limit]
    assert [e["key"] for e in events if e["stage"] == "serving"
            and e.get("decision") == "served"] == served
    assert not any(e.get("key") == "Collaboration2013a"
                   for e in events if e["stage"] in {"return_gate", "ranking", "serving"})
    assert {p.key: p.model_dump() for p in lib.all_papers()} == before
    assert lib.index_path.read_bytes() == index_before


async def test_enriched_canonical_preview_keeps_later_external_vouches(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    original = node("Collaboration2013", "The cosmic-ray positron energy spectrum measured by PAMELA")
    twin = dict(original, key="Collaboration2013a", doi="", arxiv_id="",
                title="Cosmic-Ray Positron Energy Spectrum Measured by PAMELA")
    competitor = node("Competitor", "Other detector measurements of energetic particles", rank=1)
    for candidate in [original, twin, competitor]:
        add_record(lib, **candidate)
    first = dict(twin, arxiv_id="1308.0133", rank=0)
    later = dict(twin, arxiv_id="1308.0133v1", rank=4,
                 title="Revised instrument data release with extended positron statistics")
    # Only preview's enriched copy knows this arXiv ID; the live catalog remains
    # unchanged and cannot find the later observation by title or identifiers.
    assert lib.find(arxiv_id=later["arxiv_id"], title=later["title"]) is None
    assert lib._find_preprint_journal_dup(later) is None
    result, seen, _ = await run_search(
        monkeypatch, lib, [first, later, competitor, dict(competitor, rank=5)],
        limit=1, terms=["PAMELA positron spectrum"])
    pool = {c["key"]: c for c in seen["pool"]}
    assert set(pool) == {"Collaboration2013", "Competitor"}
    canonical = pool["Collaboration2013"]
    assert canonical["folded_vouch"] == 2
    assert canonical["best_external_rank"] == 0
    assert canonical["external_rrf_score"] == pytest.approx(1 / 60 + 1 / 64)
    assert [r["key"] for r in result["results"]] == ["Collaboration2013"]
    assert lib.get("Collaboration2013").arxiv_id == ""
    assert lib.get("Collaboration2013a").arxiv_id == ""


@pytest.mark.parametrize("merge_enabled", [False, True])
async def test_explicit_alias_resolves_before_pooling(tmp_path, monkeypatch, merge_enabled):
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1" if merge_enabled else "0")
    from papervault.library.state_v2 import Merge, TerminalState
    lib = Library(tmp_path)
    a = node("Canonical", "PAMELA detector measurements of positron abundance")
    b = node("Alias", "PAMELA detector measurements of energetic particle spectrum")
    add_record(lib, **a)
    alias = add_record(lib, **b)
    alias.terminal_state = TerminalState(merge=Merge.ALIAS, alias_key="Canonical", quarantine=True)
    result, seen, events = await run_search(monkeypatch, lib, [b, a], limit=3)
    assert any(e["stage"] == "fusion" and e.get("candidate") == "doi:" + b["doi"].lower()
               and e.get("decision") == "folded" and e.get("canonical_key") == "Canonical"
               for e in events)
    assert [r["key"] for r in result["results"]] == ["Canonical"]
    assert [c["key"] for c in seen["pool"]] == ["Canonical"]


@pytest.mark.parametrize("kind", ["bibliography", "arxiv_version"])
async def test_ambiguous_or_versioned_library_rows_are_not_folded(tmp_path, monkeypatch, kind):
    lib = Library(tmp_path)
    a = node("A", "Precision detector measurements of energetic particle abundances")
    b = dict(a, key="B", authors=["Different, B."], year=2014)
    a["doi"] = b["doi"] = ""
    if kind == "arxiv_version":
        a["arxiv_id"], b["arxiv_id"] = "1001.1234v1", "1001.1234v2"
    for candidate in [a, b]:
        add_record(lib, **candidate)
    for i in range(5):
        add_record(lib, **node(f"Other{i}", f"Unrelated photospheric magnetic reconnection study {i}"))
    result, _, _ = await run_search(monkeypatch, lib, [], limit=3,
                                    terms=["detector particle abundances"])
    assert {r["key"] for r in result["results"]} == {"A", "B"}


async def test_joint_entity_witness_satisfies_both_needs_with_one_slot(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    a = node("P", "PAMELA positron abundance measurements")
    b = node("Both", "PAMELA and AMS-02 positron abundance measurements", rank=1)
    result, _, _ = await run_search(
        monkeypatch, lib, [a, b], query="Find PAMELA and AMS-02 measurements", limit=1,
        scores={a["doi"]: 0.9, b["doi"]: 0.8})
    assert result["results"][0]["doi"] == b["doi"]


@pytest.mark.parametrize("ams_score,expected_ams", [(0.7, True), (0.69, False)])
async def test_multi_entity_coverage_is_reserved_only_at_high_confidence(
        tmp_path, monkeypatch, ams_score, expected_ams):
    lib = Library(tmp_path)
    nodes = [
        node("PAMELA1", "PAMELA positron fraction detector measurements", cites=1000),
        node("PAMELA2", "PAMELA positron abundance detector measurements", rank=1, cites=900),
        node("AMS", "Alpha Magnetic Spectrometer positron fraction detector measurements", term=1),
    ]
    result, _, events = await run_search(
        monkeypatch, lib, nodes, query="Find PAMELA and AMS-02 positron measurements",
        terms=["PAMELA positron fraction", "AMS-02 positron fraction"], limit=2,
        scores={nodes[0]["doi"]: 0.9, nodes[1]["doi"]: 0.85, nodes[2]["doi"]: ams_score})
    assert len(result["results"]) == 2
    assert any(r["doi"] == nodes[2]["doi"] for r in result["results"]) is expected_ams
    assert any("PAMELA" in r["title"] for r in result["results"])
    if expected_ams:
        assert any(e.get("reason") == "named_entity_coverage" for e in events)


async def test_non_entity_facets_do_not_reserve_low_ranked_slots(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    nodes = [node("A", "Detector measurements of energetic protons"),
             node("B", "Detector measurements with transport models", term=1)]
    result, _, _ = await run_search(monkeypatch, lib, nodes, limit=1,
                                    terms=["proton measurements", "transport models"],
                                    scores={nodes[0]["doi"]: 0.9, nodes[1]["doi"]: 0.7})
    assert result["results"][0]["doi"] == nodes[0]["doi"]


async def test_relevance_ties_are_deterministic_across_library_order(tmp_path, monkeypatch):
    candidates = [node("A", "Detector measurements of proton transport"),
                  node("B", "Detector measurements of electron transport")]
    served = []
    for i, order in enumerate([candidates, candidates[::-1]]):
        lib = Library(tmp_path / str(i))
        for candidate in order:
            add_record(lib, **candidate)
        result, _, _ = await run_search(monkeypatch, lib, [], limit=2)
        served.append([r["key"] for r in result["results"]])
    assert served[0] == served[1]


def test_authority_prior_defaults_to_gentle_enabled_blend(monkeypatch):
    from papervault import config
    monkeypatch.delenv("PAPERVAULT_SEARCH_AUTHORITY_PRIOR", raising=False)
    assert config.search_authority_prior_enabled() is True
    assert protocol.SEARCH_AUTHORITY_LAMBDA == 0.25
    monkeypatch.setenv("PAPERVAULT_SEARCH_AUTHORITY_PRIOR", "0")
    assert config.search_authority_prior_enabled() is False


@pytest.mark.parametrize("year,filters", [(1990, {"year_min": 2020}), (2025, {"year_max": 2020})])
async def test_preview_rechecks_year_after_resolving_unknown_year_hit(tmp_path, monkeypatch, year, filters):
    lib = Library(tmp_path)
    local = node("Dated", "Precision detector measurements of cosmic positron fraction", year=year)
    add_record(lib, **local)
    result, seen, _ = await run_search(
        monkeypatch, lib, [dict(local, year=None, doi="")], filters=filters)
    assert result["results"] == []
    assert seen["pool"] == []


async def test_year_bounds_use_enriched_preview_projection(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    title = "Precision detector measurements of cosmic positron fraction"
    local = node("Undated", title, year=None)
    local.update(doi="", arxiv_id="")
    add_record(lib, **local)
    first = node("DoiObservation", title, year=None)
    second = node("PreprintObservation", title, year=1990, rank=1)
    second.update(doi="", arxiv_id="2001.00001v1")
    result, seen, _ = await run_search(monkeypatch, lib, [first, second], filters={"year_min": 2020})
    assert result["results"] == []
    assert seen["pool"] == []
    assert lib.get("Undated").year is None


async def test_bibliographic_canonical_mapping_cannot_bridge_conflicting_doi(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    title = "Precision detector measurements of cosmic positron fraction"
    canonical = node("Canonical", title, doi="10.1234/first")
    poor = dict(canonical, key="IdentifierPoor", doi="")
    add_record(lib, **canonical)
    add_record(lib, **poor)
    distinct = node("Distinct", title, doi="10.1234/second")
    result, seen, _ = await run_search(monkeypatch, lib, [distinct], limit=3)
    assert "10.1234/second" in {r["doi"] for r in result["results"]}
    assert "10.1234/second" in {r["doi"] for r in seen["pool"]}
    assert lib.get("Canonical").doi == "10.1234/first"


@pytest.fixture
def local_authority_fusion():
    from papervault.library.services import search_pipeline as pipeline

    # More relevant papers than LIB_CAP: the landmark starts below the BM25 cut.
    papers = [Paper(key=f"Relevant{i:03d}", title=f"Detector measurements case {i}",
                    doi=f"10.1234/relevant{i}", year=2013)
              for i in range(50)]
    papers.append(Paper(key="ZLandmark", title="Detector measurements landmark study",
                        doi="10.1234/landmark", year=1980, citation_count=50000))
    # Nonmatching records keep the real BM25 query terms discriminative.
    papers.extend(Paper(key=f"Other{i:03d}", title=f"Unrelated astrophysical topic {i}",
                        doi=f"10.1234/other{i}", year=2013,
                        citation_count=1000000 if i == 0 else 0)
                  for i in range(150))
    papers.append(Paper(key="OutsideYear", title="Detector measurements early",
                        doi="10.1234/early", year=1900, citation_count=1000000))
    searcher = pipeline.SearchPipeline(
        library=None, registration_queue=None, llm_factory=None, parse=None,
        retrieve=None, judge_ingest=None, judge_return=None, project=None,
        reframe_hint=None, progress=None, authority_enabled=False, authority_weight=0.25)
    plan = pipeline.PlannedSearch({}, ["detector measurements"], {"year_min": 1950},
                                  15, "by_relevance")
    retrieved = pipeline.RetrievedSearch([], papers, {})
    return searcher, plan, retrieved


def fuse_local(searcher, plan, retrieved):
    return searcher.fusion(plan, retrieved, SimpleNamespace(record=lambda *args, **kwargs: None))


def test_local_authority_prior_lifts_landmark_before_bm25_pool_cut(local_authority_fusion):
    searcher, plan, retrieved = local_authority_fusion
    baseline = fuse_local(searcher, plan, retrieved)
    assert len(baseline.local) == 30
    assert "ZLandmark" not in {p["key"] for p in baseline.local}
    searcher.authority_enabled = True
    fused = fuse_local(searcher, plan, retrieved)
    keys = [p["key"] for p in fused.local]
    assert "ZLandmark" in keys
    assert len(keys) == 30
    assert "OutsideYear" not in keys
    assert not any(k.startswith("Other") for k in keys)
    assert fused.external == baseline.external
    assert fused.canonical_keys == baseline.canonical_keys


@pytest.mark.parametrize("enabled,weight,hint,zero_citations", [
    (False, 0.25, "by_relevance", False),
    (True, 0.0, "by_relevance", False),
    (True, 0.25, "by_recency", False),
    (True, 0.25, "by_importance", False),
    (True, 0.25, "by_relevance", True),
])
def test_local_authority_bypasses_are_byte_identical(
        local_authority_fusion, enabled, weight, hint, zero_citations):
    from dataclasses import asdict, replace

    searcher, plan, retrieved = local_authority_fusion
    plan = replace(plan, ranking_hint=hint)
    if zero_citations:
        for paper in retrieved.local:
            paper.citation_count = 0
    baseline = json.dumps(asdict(fuse_local(searcher, plan, retrieved)), sort_keys=True)
    searcher.authority_enabled = enabled
    searcher.authority_weight = weight
    actual = json.dumps(asdict(fuse_local(searcher, plan, retrieved)), sort_keys=True)
    assert actual == baseline


def test_local_authority_uses_age_normalized_citations(local_authority_fusion):
    from datetime import datetime, timezone

    searcher, plan, retrieved = local_authority_fusion
    landmark = next(p for p in retrieved.local if p.key == "ZLandmark")
    now_year = datetime.now(timezone.utc).year
    landmark.year, landmark.citation_count = now_year - 50, 10000
    retrieved.local.append(Paper(
        key="ZRecent", title="Detector measurements recent study", doi="10.1234/recent",
        year=now_year - 1, citation_count=1000))
    searcher.authority_enabled = True
    keys = [p["key"] for p in fuse_local(searcher, plan, retrieved).local]
    assert "ZRecent" in keys and "ZLandmark" in keys
    assert keys.index("ZRecent") < keys.index("ZLandmark")
