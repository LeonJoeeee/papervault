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
        seen["ingest_pool"] = cands
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


def fuse_library(lib, external):
    from papervault.library.services.search_pipeline import PlannedSearch, RetrievedSearch, SearchPipeline
    from papervault.library.services.registration import ServingTrace

    searcher = SearchPipeline(
        library=lib, registration_queue=None, llm_factory=None, parse=None,
        retrieve=None, judge_ingest=None, judge_return=None, project=None,
        reframe_hint=None, progress=None, authority_enabled=False)
    plan = PlannedSearch({}, ["detector measurements"], {}, 10, "by_relevance")
    retrieved = RetrievedSearch(external, [p.model_copy(deep=True) for p in lib.all_papers(include_quarantined=True)], {})
    return searcher.fusion(plan, retrieved, ServingTrace(lib))


def test_planned_search_preserves_query_and_independent_kg_concepts():
    from papervault.library.services.search_pipeline import PlannedSearch

    intent = {"kg_concepts": ["diffusive shock acceleration", "Parker transport equation"]}
    plan = PlannedSearch(intent, ["SEP transport"], {}, 10, "by_relevance",
                         query="How are energetic particles accelerated?")
    assert plan.query == "How are energetic particles accelerated?"
    assert plan.kg_concepts == ["diffusive shock acceleration", "Parker transport equation"]
    legacy = PlannedSearch({}, ["SEP transport"], {}, 10, "by_relevance")
    assert legacy.kg_concepts == ["SEP transport"]


@pytest.mark.asyncio
async def test_retrieval_runs_kg_off_loop_against_the_local_snapshot(tmp_path, monkeypatch):
    import asyncio
    import threading
    from papervault.library.services import search_pipeline as pipeline
    from papervault.library.services.registration import ServingTrace

    lib = Library(tmp_path)
    paper = add_record(lib, **node("Local", "Original detector measurements"))
    event_thread = threading.get_ident()
    event_loop = asyncio.get_running_loop()
    kg_observations = []

    def kg_search(query, plan, view, *, embedding_loop):
        assert embedding_loop is event_loop
        kg_observations.append((query, plan.kg_concepts, view.get("Local").title,
                                threading.get_ident() != event_thread))
        return ["Local"]

    async def external(*args, **kwargs):
        paper.title = "Changed while external fetch runs"
        return [], {}

    async def progress(*args):
        pass

    monkeypatch.setattr(pipeline, "kg_local_search", kg_search, raising=False)
    searcher = pipeline.SearchPipeline(lib, None, None, None, external, None, None,
                                       None, None, progress)
    plan = SimpleNamespace(terms=["detector measurements"], filters={}, ranking_hint="by_relevance",
                           query="Find detector measurements", kg_concepts=["particle detector"])
    result = await searcher.retrieval(plan, ServingTrace(lib))
    assert result.kg_ranks == ["Local"]
    assert kg_observations == [("Find detector measurements", ["particle detector"],
                                "Original detector measurements", True)]
    assert result.local[0].title == "Original detector measurements"


async def test_retrieval_uses_search_permits_while_download_permits_are_saturated(tmp_path, monkeypatch):
    import asyncio
    from papervault.library.services import concurrency, search_pipeline as pipeline
    from papervault.library.services.registration import ServingTrace

    lib = Library(tmp_path)
    downloads, searches = asyncio.Semaphore(4), asyncio.Semaphore(1)
    monkeypatch.setattr(concurrency, "network_sem", downloads)
    monkeypatch.setattr(concurrency, "search_network_sem", searches, raising=False)
    for _ in range(4):
        await downloads.acquire()

    async def external(*args, **kwargs):
        assert searches.locked()
        return [node("External", "Detector measurements")], {}

    async def progress(*args):
        pass

    monkeypatch.setattr(pipeline, "kg_local_search", lambda *a, **k: [])
    searcher = pipeline.SearchPipeline(lib, None, None, None, external, None, None,
                                       None, None, progress)
    plan = pipeline.PlannedSearch({}, ["detector measurements"], {}, 10, "by_relevance")
    try:
        result = await asyncio.wait_for(searcher.retrieval(plan, ServingTrace(lib)), 1)
        assert [p["key"] for p in result.external] == ["External"]
        assert searches.locked() is False
    finally:
        for _ in range(4):
            downloads.release()


def test_local_snapshot_copies_each_paper_once_and_reuses_it_in_frozen_view(tmp_path, monkeypatch):
    from papervault.library.services.search_pipeline import SearchPipeline

    lib = Library(tmp_path)
    original = add_record(lib, **node("Local", "Detector measurements"))
    copy_paper, copies = Paper.model_copy, []

    def counted(paper, **kwargs):
        copies.append((paper.key, kwargs))
        return copy_paper(paper, **kwargs)

    monkeypatch.setattr(Paper, "model_copy", counted)
    searcher = SearchPipeline(lib, None, None, None, None, None, None, None, None, None)
    local, view = searcher._local_snapshot()
    assert copies == [("Local", {"deep": True})]
    assert view.get("Local") is local[0]
    assert local[0] is not original
    assert all(index.frozen for index in view._lookup_indexes.values())
    original.authors.append("A live edit")
    assert local[0].authors == ["Tester, A."]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["all_papers", "model_copy", "_view"])
async def test_retrieval_prepares_isolated_snapshot_off_loop(tmp_path, monkeypatch, stage):
    import asyncio
    import threading
    from papervault.library.services import search_pipeline as pipeline
    from papervault.library.services.registration import ServingTrace

    lib = Library(tmp_path)
    paper = add_record(lib, **node("Local", "Original detector measurements"))
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    owner = {"all_papers": Library, "model_copy": Paper, "_view": pipeline}[stage]
    original = getattr(owner, stage)
    calls = []

    def checked(*args, **kwargs):
        assert threading.get_ident() != loop_thread, "snapshot preparation ran on the event loop"
        released = threading.Event()
        loop.call_soon_threadsafe(released.set)
        assert released.wait(5), "event loop could not run while snapshot worker was waiting"
        calls.append(kwargs)
        return original(*args, **kwargs)

    async def external(*args, **kwargs):
        paper.authors.append("Changed during retrieval")
        return [], {}

    async def progress(*args):
        pass

    def kg_search(query, plan, view, **kwargs):
        assert view.get("Local").authors == ["Tester, A."]
        return ["Local"]

    monkeypatch.setattr(owner, stage, checked)
    monkeypatch.setattr(pipeline, "kg_local_search", kg_search)
    searcher = pipeline.SearchPipeline(lib, None, None, None, external, None, None,
                                       None, None, progress)
    plan = pipeline.PlannedSearch({}, ["detector measurements"], {}, 10, "by_relevance")
    result = await searcher.retrieval(plan, ServingTrace(lib))
    assert calls
    if stage == "model_copy":
        assert all(call.get("deep") is True for call in calls)
    assert result.local[0].authors == ["Tester, A."]
    assert paper.authors == ["Tester, A.", "Changed during retrieval"]
    assert result.kg_ranks == ["Local"]


@pytest.mark.asyncio
async def test_pipeline_lets_intent_parser_select_its_own_role(tmp_path):
    from papervault.library.services.search_pipeline import SearchPipeline

    judge_handle = object()
    lib = Library(tmp_path)

    def parse(query, llm):
        assert llm is None, "A shared judge handle bypasses the dedicated intent route"
        return {"search_terms": ["detector measurements"], "kg_concepts": ["particle detector"]}

    async def retrieve(*args, **kwargs):
        return [], {}

    async def judge(*args, llm, **kwargs):
        assert llm is judge_handle
        return {}, 0

    async def progress(*args):
        pass

    queue = SimpleNamespace(submit=lambda *args, **kwargs: None)
    searcher = SearchPipeline(lib, queue, lambda: judge_handle, parse, retrieve, judge, judge,
                              None, lambda: "reframe", progress)
    result = await searcher.run("detector measurements")
    assert result["status"] == "ok"
    assert result["intent_parsed"]["kg_concepts"] == ["particle detector"]


@pytest.mark.parametrize("fields", [
    {"venue": "AGU Fall Meeting Abstracts"}, {"venue": "EGU General Assembly Abstracts"},
    {"venue": "AAS Meeting Abstracts"}, {"venue": "APS April Meeting Abstracts"},
    {"venue": "COSPAR Scientific Assembly"}, {"doi": "10.5194/egusphere-egu26-123"},
    {"publication_types": ["dataset"]}, {"authors": ["Anon"]}, {"authors": ["unknown"]},
])
@pytest.mark.parametrize("external_hit", [False, True])
def test_fusion_excludes_legacy_garbage_from_all_candidate_pools(tmp_path, fields, external_hit):
    lib = Library(tmp_path)
    incoming = node("Legacy", "Detector measurements of cosmic particles")
    paper = add_record(lib, **dict(incoming, **fields))
    incoming["doi"] = paper.doi
    for i in range(5):
        add_record(lib, **dict(node(f"Other{i}", f"Unrelated astrophysical topic {i}"), abstract="Stellar spectra."))
    before = paper.model_dump()
    fused = fuse_library(lib, [incoming] if external_hit else [])
    assert fused.local == []
    assert fused.external == []
    assert fused.vouched == []
    assert "Legacy" not in fused.canonical_keys
    events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
    assert any(e.get("key") == "Legacy" and e.get("reason") == "serving_excluded"
               and e.get("decision") == "dropped" for e in events)
    assert not any(e.get("reason") == "unresolved_alias" for e in events)
    assert paper.model_dump() == before
    assert lib.find(doi=paper.doi) is paper


@pytest.mark.parametrize("match", ["arxiv", "title", "alias", "alias_chain", "preprint"])
def test_fusion_blocks_secondary_matches_to_excluded_canonical_records(tmp_path, match):
    from papervault.library.state_v2 import Merge, TerminalState

    lib = Library(tmp_path)
    incoming = node("Legacy", "Detector measurements of cosmic particles")
    paper = add_record(lib, **dict(incoming, arxiv_id="2401.12345", venue="AGU Fall Meeting Abstracts"))
    if match == "arxiv":
        incoming.update(doi="", arxiv_id=paper.arxiv_id, title="A different source title")
    elif match == "title":
        incoming.update(doi="", arxiv_id="")
    elif match == "preprint":
        paper.doi = ""
        incoming["title"] += " revised"
    else:
        alias = add_record(lib, **node("Alias", "Other detector measurements"))
        alias.terminal_state = TerminalState(merge=Merge.ALIAS, alias_key=paper.key, quarantine=True)
        incoming = node("Alias", alias.title)
        if match == "alias_chain":
            middle = add_record(lib, **node("Middle", "Intermediate detector measurements"))
            middle.terminal_state = TerminalState(merge=Merge.ALIAS, alias_key=paper.key)
            alias.terminal_state = TerminalState(merge=Merge.ALIAS, alias_key=middle.key)
            alias.arxiv_id = "2402.54321v2"
            incoming.update(doi="", arxiv_id=alias.arxiv_id, title="Changed source title")
    fused = fuse_library(lib, [incoming])
    assert fused.external == []
    assert fused.vouched == []
    assert all(n["key"] != paper.key for n in fused.local)


@pytest.mark.parametrize("catalog_size", [0, 200])
def test_fusion_full_snapshot_skips_unrelated_author_comparisons(
    tmp_path, monkeypatch, catalog_size
):
    from papervault.library import store

    lib = Library(tmp_path)
    for i in range(catalog_size):
        add_record(
            lib,
            key=f"Allowed{i}",
            title=f"Unrelated observations {i}",
            arxiv_id=f"2401.{i:05}",
            authors=[f"Other Family{i}"],
        )
    add_record(
        lib,
        key="Blocked",
        title="Excluded observations",
        arxiv_id="2402.12345",
        domain_status="off_domain",
        authors=["Alice Smith"],
    )
    comparisons = []
    original = store._author_lastname_jaccard

    def observe(left, right):
        comparisons.append(right)
        return original(left, right)

    monkeypatch.setattr(store, "_author_lastname_jaccard", observe)
    incoming = dict(node("New", "New detector measurements"), authors=["Alice Smith"])
    fused = fuse_library(lib, [incoming])
    assert [n["key"] for n in fused.external] == ["New"]
    assert comparisons == [["Alice Smith"]]


def test_fusion_false_positive_exact_doi(tmp_path):
    lib = Library(tmp_path)
    add_record(
        lib, key="Allowed", title="Shared title", doi="10.1234/allowed", authors=["Alice Smith"]
    )
    add_record(
        lib,
        key="Blocked",
        title="Shared title",
        domain_status="off_domain",
        authors=["Other Author"],
    )
    fused = fuse_library(
        lib,
        [dict(node("External", "Shared title", doi="10.1234/allowed"), authors=["Alice Smith"])],
    )
    assert fused.external == []
    assert [n["key"] for n in fused.vouched] == ["Allowed"]


def test_fusion_exclusion_view_reuses_isolated_snapshot_without_copying(tmp_path, monkeypatch):
    from papervault.library.services import search_pipeline as pipeline

    lib = Library(tmp_path)
    add_record(lib, **node("Blocked", "Excluded detector measurements"), domain_status="off_domain")
    original = pipeline._view
    views = []

    def observe(library, snapshot, **kwargs):
        def unexpected_copy(*args, **kw):
            pytest.fail("fusion copied an already isolated snapshot")

        with monkeypatch.context() as patch:
            patch.setattr(Paper, "model_copy", unexpected_copy)
            view = original(library, snapshot, **kwargs)
        assert all(view._papers[key] is paper for key, paper in snapshot.items())
        views.append(view)
        return view

    monkeypatch.setattr(pipeline, "_view", observe)
    fused = fuse_library(lib, [node("New", "New detector measurements")])
    assert [n["key"] for n in fused.external] == ["New"]
    assert len(views) == 1
    assert all(index.frozen for index in views[0]._lookup_indexes.values())


@pytest.mark.parametrize("empty", [False, True])
def test_frozen_view_builds_field_buckets_and_alias_list_only_once(tmp_path, empty):
    from papervault.library.services.registration import _find, _view

    class CountedPapers(dict):
        scans = 0
        alias_scans = 0

        def items(self):
            self.scans += 1
            return super().items()

        def values(self):
            self.alias_scans += 1
            return super().values()

    lib = Library(tmp_path)
    if not empty:
        add_record(lib, **node("Local", "Unrelated measurements"))
    view = _view(lib, lib._papers, deep_copy=False, frozen=True)
    view._papers = CountedPapers(view._papers)
    for _ in range(4):
        assert _find(view, dict(doi="10.1234/missing", title="Missing paper",
                                authors=["Other Author"])) is None
        assert _find(view, dict(arxiv_id="2401.99999", title="Other missing paper",
                                authors=["Other Author"])) is None
    assert view._papers.scans == 4  # DOI, arXiv, title and authors, including empty indexes
    assert view._papers.alias_scans <= 1


def test_fusion_false_negative_alias_title(tmp_path, monkeypatch):
    from papervault.library.state_v2 import Merge, TerminalState

    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    add_record(lib, key="BlockedPoor", title="Shared title", domain_status="off_domain")
    add_record(
        lib,
        key="BlockedDoi",
        title="Shared title",
        doi="10.1234/blocked",
        domain_status="off_domain",
    )
    add_record(lib, key="Allowed", title="Different title", doi="10.1234/allowed")
    add_record(
        lib,
        key="Alias",
        title="Shared title",
        doi="10.1234/alias",
        terminal_state=TerminalState(merge=Merge.ALIAS, alias_key="Allowed"),
    )
    fused = fuse_library(lib, [node("External", "Shared title", doi="10.1234/new")])
    assert fused.external == []
    assert fused.vouched == []


@pytest.mark.parametrize(
    "field,normalizer,value",
    [
        ("doi", "_normalize_doi", "10.1234/missing"),
        ("arxiv_id", "_normalize_arxiv", "2401.99999"),
        ("title", "normalize_title", "Missing title"),
    ],
)
def test_lookup_misses_do_not_renormalize_unchanged_catalog(
    tmp_path, monkeypatch, field, normalizer, value
):
    from papervault.library import store

    lib = Library(tmp_path)
    for i in range(100):
        add_record(
            lib,
            key=f"Paper{i}",
            title=f"Retained title {i}",
            doi=f"10.1234/{i}",
            arxiv_id=f"2401.{i:05}",
        )
    assert lib.find(**{field: value}) is None  # warm the lazy fallback index
    calls = []
    original = getattr(store, normalizer)

    def observe(raw):
        calls.append(raw)
        return original(raw)

    monkeypatch.setattr(store, normalizer, observe)
    for _ in range(3):
        assert lib.find(**{field: value}) is None
    assert all(raw in ("", value) for raw in calls)


@pytest.mark.parametrize(
    "field,value,changed",
    [
        ("doi", "10.1234/shared", "10.1234/changed"),
        ("arxiv_id", "2401.12345", "2402.12345"),
        ("title", "Shared title", "Changed title"),
    ],
)
def test_lookup_index_reconciles_mutation_replacement_removal_and_order(
    tmp_path, field, value, changed
):
    lib = Library(tmp_path)
    first = add_record(lib, key="First", **{field: value})
    second = add_record(lib, key="Second", **{field: value})
    index = getattr(lib, {"doi": "_by_doi", "arxiv_id": "_by_arxiv", "title": "_by_title"}[field])
    assert lib.find(**{field: value}) is second  # primary index retains last-writer precedence
    index.clear()
    assert lib.find(**{field: value}) is first  # fallback uses dictionary order
    setattr(first, field, changed)
    assert lib.find(**{field: value}) is second
    del lib._papers[second.key]
    assert lib.find(**{field: value}) is None
    replacement = Paper(key="First", **{field: value})
    lib._papers["First"] = replacement
    assert lib.find(**{field: value}) is replacement
    lib._papers["Second"] = second
    lib._papers["First"] = lib._papers.pop("First")
    index.clear()
    assert lib.find(**{field: value}) is second


def test_lookup_alias_fallback_reuses_tokens_and_reads_mutations(tmp_path, monkeypatch):
    from papervault.library import store
    from papervault.library.state_v2 import Merge, TerminalState

    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    first = add_record(lib, key="First", title="Shared title")
    target = add_record(lib, key="Target", title="Different title")
    alias = add_record(
        lib,
        key="Alias",
        title="Shared title",
        terminal_state=TerminalState(merge=Merge.ALIAS, alias_key="Target"),
    )
    assert lib.find(title="Shared title") is first
    calls = []
    original = store.normalize_title

    def observe(raw):
        calls.append(raw)
        return original(raw)

    monkeypatch.setattr(store, "normalize_title", observe)
    lib._reindex(alias)  # reinstate alias-target mismatch on the primary index
    assert lib.find(title="Shared title") is first
    assert calls.count("Shared title") == 2  # reindex and query, no catalog renormalization
    first.title = "Changed title"
    assert lib.find(title="Shared title") is target


@pytest.mark.parametrize("change", ["authors", "doi", "arxiv_id", "title", "year", "alias"])
def test_preprint_index_reads_in_place_mutations(tmp_path, monkeypatch, change):
    from papervault.library.state_v2 import Merge, TerminalState

    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    paper = add_record(
        lib,
        key="Preprint",
        title="Detector measurements of cosmic particles",
        authors=["Alice Smith"],
        arxiv_id="2401.12345",
        year=2024,
    )
    incoming = dict(title=paper.title, authors=["A. Smith"], doi="10.1234/journal", year=2024)
    assert lib._find_preprint_journal_dup(incoming) is paper
    if change == "authors":
        paper.authors[:] = ["Bob Jones"]
    elif change == "alias":
        paper.terminal_state = TerminalState(merge=Merge.ALIAS, alias_key="Other")
    else:
        setattr(
            paper,
            change,
            {"doi": "10.1234/existing", "arxiv_id": "", "title": "Unrelated", "year": 2010}[change],
        )
    assert lib._find_preprint_journal_dup(incoming) is None


def test_preprint_index_reconciles_order_and_author_edits(tmp_path):
    lib = Library(tmp_path)
    first = add_record(
        lib,
        key="First",
        title="Detector measurements",
        authors=["Alice Smith"],
        arxiv_id="2401.12345",
    )
    second = add_record(
        lib, key="Second", title=first.title, authors=["Bob Jones"], arxiv_id="2401.54321"
    )
    incoming = dict(title=first.title, authors=["A. Smith"], doi="10.1234/journal")
    assert lib._find_preprint_journal_dup(incoming) is first
    first.authors[:] = ["Bob Jones"]
    second.authors[:] = ["Alice Smith"]
    assert lib._find_preprint_journal_dup(incoming) is second
    lib._papers["First"] = first.model_copy(update={"authors": ["Alice Smith"]})
    assert lib._find_preprint_journal_dup(incoming) is lib._papers["First"]
    lib._papers["First"] = lib._papers.pop("First")
    assert lib._find_preprint_journal_dup(incoming) is second
    del lib._papers["Second"]
    assert lib._find_preprint_journal_dup(incoming) is lib._papers["First"]


@pytest.mark.parametrize("stage", ["fusion", "eligibility", "preview"])
async def test_run_performs_cpu_passes_off_event_loop(tmp_path, monkeypatch, stage):
    import threading
    from papervault.library.services import search_pipeline as pipeline

    lib = Library(tmp_path)
    retained = add_record(lib, **node("Retained", "Unrelated observations"))
    before = retained.model_dump()
    event_thread = threading.get_ident()
    threads = []
    owner, name = {
        "fusion": (pipeline.SearchPipeline, "fusion"),
        "eligibility": (pipeline, "registration_rejection"),
        "preview": (pipeline, "preview_candidates"),
    }[stage]
    original = getattr(owner, name)

    def observe(*args, **kwargs):
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, observe)
    result, seen, _ = await run_search(monkeypatch, lib, [node("New", "Detector measurements")])
    assert result["status"] == "ok"
    assert any(n["doi"] == "10.1234/New" for n in seen["pool"])
    assert retained.model_dump() == before
    assert lib.keys() == ["Retained"]
    assert threads and all(thread != event_thread for thread in threads)


@pytest.mark.parametrize("overlap", [6, 7])
@pytest.mark.parametrize("incoming_identifier", ["doi", "arxiv_id"])
def test_preprint_match_checks_author_threshold_before_title(
    tmp_path, monkeypatch, overlap, incoming_identifier
):
    from papervault.library import store

    lib = Library(tmp_path)
    authors = [f"A. Lastname{i}" for i in range(10)]
    identifiers = {"doi": "10.1234/journal", "arxiv_id": "2401.12345"}
    retained_identifier = "arxiv_id" if incoming_identifier == "doi" else "doi"
    retained = add_record(
        lib,
        key="Retained",
        title="Detector measurements of cosmic particles",
        authors=authors[:overlap],
        year=2024,
        **{retained_identifier: identifiers[retained_identifier]},
    )
    incoming = dict(
        title=retained.title,
        authors=authors,
        year=2026,
        **{incoming_identifier: identifiers[incoming_identifier]},
    )
    comparisons = []
    original = store._title_similarity

    def observe(left, right):
        comparisons.append((left, right))
        return original(left, right)

    monkeypatch.setattr(store, "_title_similarity", observe)
    assert lib._find_preprint_journal_dup(incoming) is (retained if overlap == 7 else None)
    assert comparisons == ([(retained.title, retained.title)] if overlap == 7 else [])


def test_fusion_distinguishes_unresolved_alias_from_serving_exclusion(tmp_path):
    from papervault.library.state_v2 import Merge, TerminalState

    lib = Library(tmp_path)
    alias = add_record(lib, **node("Alias", "Detector measurements"))
    alias.terminal_state = TerminalState(merge=Merge.ALIAS, alias_key="Missing", quarantine=True)
    assert fuse_library(lib, []).local == []
    events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
    assert any(e.get("key") == "Alias" and e.get("reason") == "unresolved_alias" for e in events)


@pytest.mark.parametrize("different", ["doi", "arxiv_version"])
def test_fusion_does_not_exclude_distinct_versions_of_a_blocked_title(tmp_path, different):
    lib = Library(tmp_path)
    incoming = node("Allowed", "Detector measurements of cosmic particles")
    fields = {"doi": "10.1234/blocked"}
    if different == "arxiv_version":
        fields = {"doi": "", "arxiv_id": "2401.12345v1"}
        incoming.update(doi="", arxiv_id="2401.12345v2")
    add_record(lib, key="Blocked", title=incoming["title"], venue="AGU Fall Meeting Abstracts", **fields)
    fused = fuse_library(lib, [incoming])
    assert [n["key"] for n in fused.external] == ["Allowed"]


@pytest.mark.parametrize("fields", [{"venue": "AGU Fall Meeting Abstracts"},
                                    {"publication_types": ["dataset"]}, {"authors": ["Anon"]}])
async def test_search_does_not_judge_or_register_excluded_legacy_matches(tmp_path, monkeypatch, fields):
    lib = Library(tmp_path)
    incoming = node("Legacy", "Detector measurements of cosmic particles")
    paper = add_record(lib, **dict(incoming, **fields))
    before = paper.model_dump()
    index_before = lib.index_path.read_bytes()
    result, seen, events = await run_search(monkeypatch, lib, [incoming])
    assert result["results"] == []
    assert seen["ingest_pool"] == []
    assert seen["pool"] == []
    assert not any(e["stage"] == "registration" and e.get("decision") == "submitted" for e in events)
    assert lib.keys() == ["Legacy"]
    assert paper.model_dump() == before
    assert lib.index_path.read_bytes() == index_before


@pytest.mark.parametrize("fields", [{}, {"authors": [], "doi": "10.1234/collaboration"}])
async def test_search_preserves_valid_minimal_legacy_papers(tmp_path, monkeypatch, fields):
    lib = Library(tmp_path)
    add_record(lib, key="Anon2010", title="Detector measurements of cosmic particles", **fields)
    for i in range(5):
        add_record(lib, **dict(node(f"Other{i}", f"Unrelated astrophysical topic {i}"), abstract="Stellar spectra."))
    result, _, _ = await run_search(monkeypatch, lib, [])
    assert [n["key"] for n in result["results"]] == ["Anon2010"]


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
    assert len(baseline.local) == 45
    assert "ZLandmark" not in {p["key"] for p in baseline.local}
    searcher.authority_enabled = True
    fused = fuse_local(searcher, plan, retrieved)
    keys = [p["key"] for p in fused.local]
    assert "ZLandmark" in keys
    assert len(keys) == 45
    assert "OutsideYear" not in keys
    assert not any(k.startswith("Other") for k in keys)
    assert fused.external == baseline.external
    assert fused.canonical_keys == baseline.canonical_keys


def test_kg_only_paper_reaches_local_pool_with_alias_year_and_policy_fences(local_authority_fusion):
    from papervault.library.services.search_pipeline import RetrievedSearch
    from papervault.library.state_v2 import Merge, TerminalState

    searcher, plan, retrieved = local_authority_fusion
    semantic = Paper(key="Semantic", title="Suprathermal populations at interplanetary discontinuities",
                     doi="10.1234/semantic", year=2020)
    alias = Paper(key="OldAlias", title="Old source title", doi="10.1234/alias")
    alias.terminal_state = TerminalState(merge=Merge.ALIAS, alias_key="Semantic", quarantine=True)
    excluded = Paper(key="Excluded", title="Unrelated paper", doi="10.1234/excluded", year=2020)
    excluded.domain_status = "off_domain"
    retrieved.local.extend([semantic, alias, excluded])
    retrieved = RetrievedSearch([], retrieved.local, {},
                                kg_ranks=["Excluded", "OutsideYear", "Missing", "OldAlias", "Semantic"])
    fused = fuse_local(searcher, plan, retrieved)
    keys = [p["key"] for p in fused.local]
    assert "Semantic" in keys
    assert len(keys) == len(set(keys)) == 45
    assert not {"Excluded", "OutsideYear", "Missing", "OldAlias"} & set(keys)


def test_kg_rrf_consensus_lifts_deep_bm25_hit_before_pool_cut(local_authority_fusion, monkeypatch):
    from papervault.library.services import search_pipeline as pipeline

    searcher, plan, retrieved = local_authority_fusion
    retrieved.local = [Paper(key=f"Rank{i:03d}", title=f"Detector measurements {i}",
                             doi=f"10.1234/rank{i}", year=2020) for i in range(100)]
    monkeypatch.setattr(pipeline, "bm25_per_term_ranklists", lambda corpus, terms: [
        sorted(corpus, key=lambda n: n["key"])])
    retrieved = pipeline.RetrievedSearch([], retrieved.local, {}, kg_ranks=["Rank090"])
    keys = [p["key"] for p in fuse_local(searcher, plan, retrieved).local]
    assert keys[0] == "Rank090"
    assert len(keys) == 45


def test_authority_prior_applies_to_kg_only_candidates(local_authority_fusion):
    from papervault.library.services.search_pipeline import RetrievedSearch

    searcher, plan, retrieved = local_authority_fusion
    retrieved.local = [Paper(key=f"Semantic{i:03d}", title=f"Suprathermal populations {i}",
                             doi=f"10.1234/semantic{i}", year=2020,
                             citation_count=10000 if i == 70 else 0) for i in range(100)]
    retrieved = RetrievedSearch([], retrieved.local, {},
                                kg_ranks=[f"Semantic{i:03d}" for i in range(100)])
    baseline = fuse_local(searcher, plan, retrieved)
    assert "Semantic070" not in {p["key"] for p in baseline.local}
    searcher.authority_enabled = True
    keys = [p["key"] for p in fuse_local(searcher, plan, retrieved).local]
    assert "Semantic070" in keys
    assert len(keys) == 45


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


@pytest.mark.parametrize("n_terms", [1, 2])
@pytest.mark.parametrize("rank,weight,selected", [(300, 0.25, True), (799, 1.0, True), (800, 1.0, False)])
def test_local_authority_depth_and_strength(
        local_authority_fusion, monkeypatch, n_terms, rank, weight, selected):
    from dataclasses import replace
    from papervault.library.services import search_pipeline as pipeline

    searcher, plan, retrieved = local_authority_fusion
    retrieved.local = [Paper(key=f"Rank{i:03d}", title=f"Detector measurements case {i}",
                             doi=f"10.1234/rank{i}", year=2025,
                             citation_count=10000 if i == rank else 0)
                       for i in range(801)]
    # Supply a controlled BM25 rank boundary; fusion, RRF and shortlisting stay real.
    monkeypatch.setattr(pipeline, "bm25_per_term_ranklists", lambda corpus, terms: [
        sorted(corpus, key=lambda n: n["key"]) for _ in terms])
    plan = replace(plan, terms=[f"term{i}" for i in range(n_terms)])
    searcher.authority_enabled, searcher.authority_weight = True, weight
    keys = [n["key"] for n in fuse_local(searcher, plan, retrieved).local]
    assert len(keys) == 45
    assert (f"Rank{rank:03d}" in keys) is selected


@pytest.fixture
def vouched_authority_fusion(local_authority_fusion):
    from datetime import datetime, timezone

    searcher, plan, retrieved = local_authority_fusion
    now_year = datetime.now(timezone.utc).year
    # These records do not match BM25; their only path is the vouch supplement.
    vouched = [Paper(key=f"Vouch{i:02d}", title=f"Independent astrophysical observation {i}",
                     doi=f"10.1234/vouch{i}", year=now_year - 1, citation_count=i + 1)
               for i in range(49)]
    vouched.extend([
        Paper(key="VouchLandmark", title="Independent canonical astrophysical study",
              doi="10.1234/vouchlandmark", year=now_year - 1, citation_count=100),
        Paper(key="VouchNative", title="Independent uncited astrophysical study",
              doi="10.1234/vouchnative", year=now_year - 1),
        Paper(key="VouchRepeated", title="Independent repeatedly observed astrophysical stub",
              doi="10.1234/vouchrepeated", year=now_year - 1),
        Paper(key="VouchOutsideYear", title="Independent very early astrophysical study",
              doi="10.1234/vouchoutside", year=1900, citation_count=1000000),
    ])
    retrieved.local.extend(vouched)
    retrieved.external = [dict(doi=p.doi, term_idx=0,
                               rank=8 if p.key == "VouchLandmark" else
                               0 if p.key in {"VouchNative", "VouchOutsideYear"} else 1000)
                          for p in vouched]
    retrieved.external.append(dict(doi="10.1234/vouchrepeated", term_idx=0, rank=1000))
    return searcher, plan, retrieved


def test_vouched_supplement_uses_authority_blended_with_external_rank(vouched_authority_fusion):
    searcher, plan, retrieved = vouched_authority_fusion
    baseline = fuse_local(searcher, plan, retrieved)
    assert len(baseline.local) == 90  # 45 BM25 + 45 additional externally vouched records
    assert [n["key"] for n in baseline.local[45:48]] == [
        "VouchRepeated", "VouchNative", "VouchLandmark"]
    searcher.authority_enabled = True
    fused = fuse_local(searcher, plan, retrieved)
    keys = [n["key"] for n in fused.local]
    # At external rank 8, the landmark needs the specified 2*lambda blend to
    # beat native rank 0 with authority rank 50+. A 1*lambda blend loses.
    assert keys[45:47] == ["VouchLandmark", "VouchNative"]
    assert len(keys) == len(set(keys)) == 90
    assert "VouchOutsideYear" not in keys
    assert fused.external == []


@pytest.mark.parametrize("enabled,weight,hint", [
    (False, 0.25, "by_relevance"), (True, 0.0, "by_relevance"),
    (True, 0.25, "by_recency"), (True, 0.25, "by_importance"),
])
def test_vouched_authority_bypasses_are_byte_identical(
        vouched_authority_fusion, enabled, weight, hint):
    from dataclasses import asdict, replace

    searcher, plan, retrieved = vouched_authority_fusion
    plan = replace(plan, ranking_hint=hint)
    baseline = json.dumps(asdict(fuse_local(searcher, plan, retrieved)), sort_keys=True)
    searcher.authority_enabled, searcher.authority_weight = enabled, weight
    actual = json.dumps(asdict(fuse_local(searcher, plan, retrieved)), sort_keys=True)
    assert actual == baseline


def test_vouched_authority_normalizes_age_and_venue(vouched_authority_fusion):
    from datetime import datetime, timezone

    searcher, plan, retrieved = vouched_authority_fusion
    now_year = datetime.now(timezone.utc).year
    records = {p.key: p for p in retrieved.local}
    records["VouchLandmark"].year = now_year - 50
    records["VouchLandmark"].citation_count = 1000  # 20/year
    records["VouchNative"].citation_count = 30
    records["VouchNative"].venue = "PRL"  # 45/year after venue weighting
    records["VouchRepeated"].citation_count = 35  # 35/year
    retrieved.external = [dict(doi=records[key].doi, term_idx=0, rank=0)
                          for key in ("VouchLandmark", "VouchNative", "VouchRepeated")]
    searcher.authority_enabled = True
    keys = [n["key"] for n in fuse_local(searcher, plan, retrieved).local]
    assert keys[-3:] == ["VouchNative", "VouchRepeated", "VouchLandmark"]


def test_preprint_index_does_not_renormalize_unchanged_authors(tmp_path, monkeypatch):
    from papervault.library import store

    lib = Library(tmp_path)
    for i in range(100):
        add_record(
            lib,
            key=f"Paper{i}",
            title=f"Retained title {i}",
            authors=[f"Alice Family{i}"],
            arxiv_id=f"2401.{i:05}",
        )
    incoming = dict(title="New title", doi="10.1234/new", authors=["Bob Newname"])
    assert lib._find_preprint_journal_dup(incoming) is None
    calls = []
    original = store._author_lastnames

    def observe(names):
        calls.append(tuple(names))
        return original(names)

    monkeypatch.setattr(store, "_author_lastnames", observe)
    for _ in range(3):
        assert lib._find_preprint_journal_dup(incoming) is None
    assert calls == [("Bob Newname",)] * 3


def test_lookup_caches_are_isolated_in_preview_views(tmp_path, monkeypatch):
    from papervault.library import store
    from papervault.library.services.registration import _view

    lib = Library(tmp_path)
    paper = add_record(lib, key="Retained", title="Original title", authors=["Alice Smith"])
    assert lib.find(title="Missing") is None
    view = _view(lib, lib._papers)
    view._papers[paper.key].title = "Preview title"
    assert view.find(title="Preview title").title == "Preview title"
    calls = []
    original = store.normalize_title

    def observe(raw):
        calls.append(raw)
        return original(raw)

    monkeypatch.setattr(store, "normalize_title", observe)
    assert lib.find(title="Preview title") is None
    assert paper.title == "Original title"
    assert "Original title" not in calls
