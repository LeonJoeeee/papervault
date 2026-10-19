"""Serving remains a read while independently admitted registration is pending."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from papervault.library import Library
from papervault.library.mcp.server import build_server


def candidate(**fields):
    return {"title": "Cosmic ray transport measurements in the heliosphere",
            "authors": ["Tester, A."], "year": 2024, "doi": "10.1234/registration",
            "abstract": "Measurements of cosmic ray transport.", "source": "core",
            "paper_id": "123", "publication_types": ["JournalArticle"],
            "_source_origin": "external", "term_idx": 0, "rank": 0, **fields}


def wire_search(monkeypatch, nodes, *, score=0.9, dropped=0):
    monkeypatch.setattr("papervault.library.mcp.server.get_llm", lambda: SimpleNamespace())
    monkeypatch.setattr("papervault.library.mcp.server.parse_intent", lambda *a, **k: {
        "search_terms": ["cosmic ray transport"], "filters": {}, "limit_suggested": 10,
        "ranking_hint": "by_relevance"})

    async def external(*a, **k):
        return nodes, {}

    async def ingest(cands, **k):
        return {i: {"ingest_ok": True, "llm_is_paper": True, "tier": "1A"}
                for i in range(len(cands))}, 0

    async def ret(cands, *a, **k):
        return {i: {"score": score} for i in range(len(cands))}, dropped

    monkeypatch.setattr("papervault.library.mcp.server.search_external_async", external)
    monkeypatch.setattr("papervault.library.mcp.server.judge_ingest", ingest)
    monkeypatch.setattr("papervault.library.mcp.server.judge_return", ret)


async def search(server):
    result = await server.call_tool("search_papers", {"query": "find cosmic ray measurements"})
    return json.loads("\n".join(c.text for c in result))


@pytest.mark.parametrize("score", [0.9, 0.1])
async def test_search_does_not_write_catalog_or_enqueue_downloads(tmp_path, monkeypatch, score):
    lib = Library(tmp_path)
    server = build_server(library=lib)
    wire_search(monkeypatch, [candidate()], score=score)
    before = lib.index_path.read_bytes()
    calls = []
    for name in ("upsert", "save"):
        original = getattr(lib, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(lib, name, spy)
    monkeypatch.setattr(server._paper_download_queue, "add", lambda *a, **k: calls.append("download"))
    result = await search(server)
    assert result["status"] == "ok"
    assert len(result["results"]) == (1 if score > 0.4 else 0)
    assert calls == []
    assert lib.keys() == []
    assert lib.index_path.read_bytes() == before


def admitted(**fields):
    return candidate(_registration={"ingest_ok": True, "llm_is_paper": True, "tier": "1A"}, **fields)


def trace_events(lib):
    return [e for line in lib.manifest_path.read_text().splitlines()
            if (e := json.loads(line)).get("event") == "serving_trace"]


@pytest.mark.parametrize("score", [0.9, 0.1])
@pytest.mark.parametrize("capacity", [0, 1])
async def test_registration_admission_is_independent_of_served_results(
        tmp_path, monkeypatch, score, capacity):
    from papervault.library.services.registration import RegistrationQueue, RegistrationService

    lib = Library(tmp_path)
    server = build_server(library=lib)
    queue = RegistrationQueue(RegistrationService(lib, server._paper_download_queue), capacity=capacity)
    server._paper_registration_queue = queue
    wire_search(monkeypatch, [candidate()], score=score)
    await queue.start()
    try:
        result = await search(server)
        assert len(result["results"]) == (1 if score > 0.4 else 0)
        await asyncio.wait_for(queue._queue.join(), 2)
        assert len(lib.keys()) == capacity
        decisions = {e.get("decision") for e in trace_events(lib)}
        assert ("registered" if capacity else "rejected") in decisions
    finally:
        await queue.stop()


async def test_no_eligible_candidates_submit_nothing(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    server = build_server(library=lib)
    wire_search(monkeypatch, [candidate(publication_types=["Dataset"])])
    result = await search(server)
    assert result["results"] == []
    assert server._paper_registration_queue._queue.empty()
    assert not any(e.get("decision") == "submitted" for e in trace_events(lib))


async def test_stalled_registration_does_not_delay_other_searches(tmp_path, monkeypatch):
    import threading
    from papervault.library.services.registration import RegistrationQueue, RegistrationService

    lib = Library(tmp_path)
    server = build_server(library=lib)
    service = RegistrationService(lib, server._paper_download_queue)
    entered, release = threading.Event(), threading.Event()
    original = service._register

    def stalled(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_register", stalled)
    queue = RegistrationQueue(service, capacity=2)
    server._paper_registration_queue = queue
    wire_search(monkeypatch, [candidate()])
    await queue.start()
    try:
        first = await asyncio.wait_for(search(server), 1)
        assert await asyncio.to_thread(entered.wait, 1)
        second = await asyncio.wait_for(search(server), 1)
        assert first["results"] == second["results"]
        assert lib.keys() == []
    finally:
        release.set()
        await queue.stop()


async def test_failed_registration_is_recorded_and_does_not_disable_reads(tmp_path, monkeypatch):
    from papervault.library.services.registration import RegistrationService

    lib = Library(tmp_path)
    server = build_server(library=lib)
    queue = server._paper_registration_queue

    def fail(*a, **k):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(RegistrationService, "_register", fail)
    wire_search(monkeypatch, [candidate()])
    await queue.start()
    try:
        first = await search(server)
        await asyncio.wait_for(queue._queue.join(), 2)
        second = await search(server)
        assert first["results"] == second["results"]
        assert lib.keys() == []
        assert any(e.get("decision") == "failed" for e in trace_events(lib))
    finally:
        await queue.stop()


async def test_registration_keeps_metadata_when_download_budget_is_zero(tmp_path):
    from papervault.library.services.registration import RegistrationService

    lib = Library(tmp_path)
    queued = []
    service = RegistrationService(lib, SimpleNamespace(add=queued.append), max_downloads=0)
    outcomes = await service.register_candidates([admitted()], actor="test", reason="test-command")
    assert outcomes[0]["status"] == "registered"
    assert outcomes[0]["download"] == "budget_rejected"
    assert lib.find(doi="10.1234/registration").publication_types == ["JournalArticle"]
    assert queued == []


async def test_registration_reuses_worker_filled_identity_and_retains_restrictions(tmp_path):
    from papervault.library.services.registration import RegistrationService

    lib = Library(tmp_path)
    paper, _ = lib.upsert(candidate(doi="", _source_origin="library"))
    lib.save(force=True)
    paper.doi = "10.1234/registration"  # bypass stale identity cache, as a worker does
    paper.download_status = "metadata_only"
    queued = []
    service = RegistrationService(lib, SimpleNamespace(add=queued.append))
    outcome = await service.register_candidates([admitted()], actor="test", reason="duplicate-command")
    assert outcome[0]["status"] == "duplicate"
    assert lib.keys() == [paper.key]
    assert lib.get(paper.key) is paper
    assert paper.download_status == "metadata_only"
    assert queued == []


async def test_external_projection_has_stable_source_identity_before_registration(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    server = build_server(library=lib)
    wire_search(monkeypatch, [candidate()])
    first = (await search(server))["results"][0]
    second = (await search(server))["results"][0]
    assert first == second
    assert first["key"] == "doi:10.1234/registration"
    assert first["text_status"] == "pending"
    assert first["source_identities"][0]["source_ids"] == {"core:work": ["123"]}
    assert first["source_identities"][0]["source"] == "core"
    assert lib.keys() == []


async def test_registration_failure_returns_candidate_outcome(tmp_path, monkeypatch):
    from papervault.library.services.registration import RegistrationService

    lib = Library(tmp_path)
    service = RegistrationService(lib, SimpleNamespace(add=lambda key: None))
    monkeypatch.setattr(service, "_register", lambda *a, **k: (_ for _ in ()).throw(OSError("disk failed")))
    outcomes = await service.register_candidates([admitted()], actor="test", reason="failing-command")
    assert outcomes[0]["status"] == "failed"
    assert outcomes[0]["key"] is None
    assert lib.keys() == []


async def test_duplicate_registration_cannot_change_policy_or_promote_identity(tmp_path):
    from papervault.library.services.registration import RegistrationService
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState

    lib = Library(tmp_path)
    paper, _ = lib.upsert(candidate(doi="", arxiv_id="2401.12345"))
    paper.terminal_state = TerminalState(policy=Policy.CLEAR, retrieval=Retrieval.ELIGIBLE)
    lib.save(force=True)
    service = RegistrationService(lib, SimpleNamespace(add=lambda key: pytest.fail("duplicate acquisition")))
    outcomes = await service.register_candidates([admitted(terminal_state={"policy": "clear"})],
                                                  actor="test", reason="observation-only")
    assert outcomes[0]["status"] == "duplicate"
    assert paper.doi == ""  # a new scalar identifier needs the established promotion path
    assert len(paper.source_identities) == 1
    assert paper.source_identities[0].dois == ["10.1234/registration"]


async def test_library_fold_does_not_mutate_live_metadata_during_serving(tmp_path, monkeypatch):
    from papervault.library.search import _normalize_paper

    lib = Library(tmp_path)
    paper, _ = lib.upsert(candidate())
    lib.save(force=True)
    before = paper.model_dump(mode="json")
    external = _normalize_paper(candidate(source="openalex", paper_id="W123"), "openalex")
    external.update(_source_origin="external", term_idx=0, rank=0)
    server = build_server(library=lib)
    wire_search(monkeypatch, [external])
    result = await search(server)
    assert len(result["results"]) == 1
    assert paper.model_dump(mode="json") == before
    assert "source_identities" not in result["results"][0]


async def test_slow_catalog_save_does_not_hold_search_identity_lock(tmp_path, monkeypatch):
    import threading
    from papervault.library.services.registration import RegistrationService

    lib = Library(tmp_path)
    server = build_server(library=lib)
    wire_search(monkeypatch, [candidate(doi="10.1234/other",
        title="Other cosmic ray transport measurements in the heliosphere")])
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = Library._save_index

    def slow_save(self, **kwargs):
        entered.set()
        release.wait(3)
        try:
            return original(self, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(Library, "_save_index", slow_save)
    service = RegistrationService(lib, SimpleNamespace(add=lambda key: None))
    registration = asyncio.create_task(service.register_candidates(
        [admitted()], actor="test", reason="slow-io"))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        result = await asyncio.wait_for(search(server), 1)
        assert len(result["results"]) == 1
        assert not finished.is_set(), "search waited for registration's catalog IO"
    finally:
        release.set()
        await registration


async def test_shutdown_records_committed_registration_outcome(tmp_path, monkeypatch):
    import threading
    from papervault.library.services.registration import RegistrationQueue, RegistrationService, ServingTrace

    lib = Library(tmp_path)
    service = RegistrationService(lib, SimpleNamespace(add=lambda key: None))
    entered, release = threading.Event(), threading.Event()
    original = service._register

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_register", blocked)
    queue = RegistrationQueue(service)
    await queue.start()
    queue.submit([admitted()], actor="test", reason="shutdown-commit", trace=ServingTrace(lib))
    assert await asyncio.to_thread(entered.wait, 1)
    stopping = asyncio.create_task(queue.stop())
    await asyncio.sleep(0)  # deliver cancellation while the executor still owns its command
    release.set()
    await asyncio.wait_for(stopping, 2)
    assert len(lib.keys()) == 1
    decisions = [e.get("decision") for e in trace_events(lib) if e["stage"] == "registration"]
    assert decisions == ["submitted", "accepted", "registered"]


async def test_positive_capacity_queue_is_bounded_and_rejects_overload(tmp_path):
    from papervault.library.services.registration import RegistrationQueue, RegistrationService, ServingTrace

    lib = Library(tmp_path)
    service = RegistrationService(lib, SimpleNamespace(add=lambda key: None), max_candidates=0)
    queue = RegistrationQueue(service, capacity=1)
    await queue.start()
    try:
        trace = ServingTrace(lib)
        queue.submit([admitted()], actor="test", reason="first", trace=trace)
        queue.submit([admitted(doi="10.1234/second")], actor="test", reason="second", trace=trace)
        assert queue._queue.qsize() == 1
        await asyncio.wait_for(queue._queue.join(), 2)
        assert lib.keys() == []
        reasons = {e.get("reason") for e in trace_events(lib)}
        assert "registration_queue_full" in reasons
        assert "registration_budget" in reasons
    finally:
        await queue.stop()


async def test_serving_deduplicates_secondary_arxiv_before_limit(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    server = build_server(library=lib)
    nodes = [
        candidate(doi="10.1234/first", arxiv_id="2401.12345",
                  title="Cosmic ray transport measurements in the heliosphere"),
        candidate(doi="", arxiv_id="2401.12345", rank=1,
                  title="Cosmic ray transport measurements across the heliosphere"),
        candidate(doi="10.1234/distinct", rank=2,
                  title="Distinct independent cosmic ray instrument measurements"),
    ]
    wire_search(monkeypatch, nodes)
    monkeypatch.setattr("papervault.library.mcp.server.parse_intent", lambda *a, **k: {
        "search_terms": ["cosmic ray transport"], "filters": {}, "limit_suggested": 2,
        "ranking_hint": "by_relevance"})
    result = await search(server)
    assert [r["doi"] for r in result["results"]] == ["10.1234/first", "10.1234/distinct"]
    assert lib.keys() == []


@pytest.mark.parametrize("incoming_title", [candidate()["title"], "AMS-02"])
async def test_serving_scores_rich_local_projection_without_persisting_it(tmp_path, monkeypatch, incoming_title):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(candidate(doi="", arxiv_id="2401.12345", abstract="", citation_count=0))
    lib.save(force=True)
    before = paper.model_dump(mode="json")
    incoming = candidate(title=incoming_title, doi="10.1234/published", arxiv_id="2401.12345", citation_count=99,
                         abstract="New measurement evidence for cosmic ray transport.")
    server = build_server(library=lib)
    wire_search(monkeypatch, [incoming])

    async def evidence_sensitive_return(cands, *a, **k):
        return {i: {"score": 0.9 if "measurement evidence" in c.get("abstract", "")
                   and c.get("citation_count") == 99 else 0.1}
                for i, c in enumerate(cands)}, 0

    monkeypatch.setattr("papervault.library.mcp.server.judge_return", evidence_sensitive_return)
    result = await search(server)
    assert len(result["results"]) == 1
    assert result["results"][0]["abstract"] == incoming["abstract"]
    assert result["results"][0]["citation_count"] == 99
    assert result["results"][0]["key"] == paper.key
    assert paper.model_dump(mode="json") == before
    assert Library(tmp_path).get(paper.key).model_dump(mode="json") == before


@pytest.mark.parametrize("score", [0.1, 0.9])
async def test_command_budget_rejection_does_not_change_serving(tmp_path, monkeypatch, score):
    from papervault.library.services.registration import RegistrationQueue, RegistrationService

    lib = Library(tmp_path)
    server = build_server(library=lib)
    queue = RegistrationQueue(RegistrationService(lib, server._paper_download_queue, max_candidates=0))
    server._paper_registration_queue = queue
    wire_search(monkeypatch, [candidate()], score=score)
    await queue.start()
    try:
        result = await search(server)
        assert len(result["results"]) == (1 if score > 0.4 else 0)
        await asyncio.wait_for(queue._queue.join(), 2)
        assert lib.keys() == []
        assert any(e.get("decision") == "rejected" and e.get("reason") == "registration_budget"
                   for e in trace_events(lib))
    finally:
        await queue.stop()


async def test_conflicting_duplicate_metadata_is_retained_as_registration_evidence(tmp_path):
    from papervault.library.services.registration import RegistrationService

    lib = Library(tmp_path)
    service = RegistrationService(lib, SimpleNamespace(add=lambda key: None))
    first = await service.register_candidates([admitted()], actor="test", reason="original")
    alternate = admitted(title="Alternative source title for cosmic ray transport measurements")
    result = await service.register_candidates([alternate], actor="test", reason="alternate")
    assert result[0]["status"] == "duplicate"
    assert lib.get(first[0]["key"]).title == candidate()["title"]
    events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
    evidence = next(e for e in events if e.get("event") == "registration" and e.get("reason") == "alternate")
    assert evidence["observations"][0]["title"] == alternate["title"]
