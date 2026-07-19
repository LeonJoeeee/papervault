"""Tests for papervault.library.mcp.server — the two executor-facing tools
(``get_paper`` batch resolver + ``search_papers`` 3-LLM discovery) and the
read-only ``library://`` resources, via the FastMCP ``call_tool`` /
``read_resource`` async interface.

The MCP surface is the **library↔executor interface** (2026-05): get_paper returns
the *minimal record* (9 fields + a ``text_path`` / ``text_status`` reference) and
search_papers runs intent → ingest-gate → return-gate. The old per-tool surfaces
(get_full_text / get_bibtex / cite_check / get_book_chapter) moved off MCP
(text via ``text_path`` + Read; bibtex / cite-check via the ``paper-library`` CLI).

External IO is mocked at fixture level so tests are hermetic:
- ``papervault.library.search.search_all`` / ``search_external_async``
- ``papervault.library.fetch.fetch_by_doi`` / ``fetch_by_arxiv``
- the two judge LLMs (``judge_ingest`` / ``judge_return``) for search tests

D13 (2026-05-31): ``materialize_paper`` was deleted. ``get_paper`` is
read-only and only enqueues download/extract work via the stage queues'
``add()`` (never blocks on completion), so these tests no longer mock any
materialize facade — the real read-only path is exercised directly.
"""

from __future__ import annotations

import json

import pytest

from papervault.library import Library


# ---------------- fixtures --------------------------------------------------


@pytest.fixture
def populated_lib(tmp_path):
    """Library with two papers; only Wei2024 has full extracts on disk."""
    lib = Library(tmp_path)
    lib.upsert({
        "title": "Solar wind modulation",
        "authors": ["Potgieter"],
        "year": 2013,
        "doi": "10.1234/abc",
        "is_review": True,
        "abstract": "review of solar wind transport",
        "citation_count": 100,
    })
    lib.upsert({
        "title": "Cosmic ray PINN",
        "authors": ["Wei"],
        "year": 2024,
        "doi": "10.1234/def",
        "abstract": "neural network cosmic ray",
        "citation_count": 50,
    })
    # Wei2024 is "complete": pdf + md + txt all on disk. The txt is a real
    # (≥ byte-floor) pypdf dump — a few-byte stub would be (correctly) refused
    # by serve-safety as sub-floor (F3), so it must be a realistic size here.
    lib.txt_path("Wei2024").write_text("body of Wei2024. " * 40)
    lib.md_path("Wei2024").write_text("# Wei2024\nmarkdown body")
    lib.pdf_path("Wei2024").write_bytes(b"%PDF-1.0 fake content")
    lib.save()
    return lib


@pytest.fixture
def server(populated_lib, monkeypatch):
    """A FastMCP server bound to populated_lib with all external IO mocked.

    The mocks default to "no-op" / "nothing found" so a test that doesn't
    override them is hermetic by construction. Tests that exercise the
    new-fetch path override the mock per-test.
    """
    # Mock external multi-backend search to return no candidates by default.
    def fake_search_all(*args, **kwargs):
        return []
    monkeypatch.setattr("papervault.library.search.search_all", fake_search_all)
    monkeypatch.setattr("papervault.library.mcp.server.search_all", fake_search_all)

    async def fake_search_external_async(terms, *, year_min=None, year_max=None,
                                         ranking_hint="by_relevance"):
        return [], {}   # V6: (ext_raw, degraded_map) 2-tuple (§3)
    monkeypatch.setattr("papervault.library.mcp.server.search_external_async",
                        fake_search_external_async)

    # Mock direct DOI / arxiv lookups to "not found" by default.
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", lambda d: None)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)

    from papervault.library.mcp.server import build_server
    return build_server(library=populated_lib)


# ---------------- async helpers ---------------------------------------------

# The minimal record get_paper / search_papers return (no internal fields leak).
MIN_FIELDS = {"key", "title", "authors", "year", "venue", "abstract",
              "doi", "arxiv_id", "citation_count", "text_path", "text_status"}


async def _call(server, name, args):
    """Call an MCP tool and return the parsed JSON payload."""
    result = await server.call_tool(name, args)
    text = "\n".join(c.text for c in result)
    return json.loads(text)


async def _get_one(server, identifier):
    """get_paper with a single identifier → the single result item."""
    out = await _call(server, "get_paper", {"identifiers": identifier})
    assert out["status"] == "ok"
    assert len(out["results"]) == 1
    return out["results"][0]


async def _read(server, uri):
    """Read an MCP resource; return raw content (string)."""
    result = await server.read_resource(uri)
    return "\n".join(getattr(c, "content", "") for c in result)


# ============== get_paper (batch resolver) ==================================


@pytest.mark.asyncio
async def test_get_paper_known_key_complete(server):
    """A key for a paper whose extract is on disk → found + text_path (md)."""
    item = await _get_one(server, "Wei2024")
    assert item["status"] == "found"
    assert item["identifier"] == "Wei2024"
    assert item["key"] == "Wei2024"
    assert item["title"] == "Cosmic ray PINN"
    assert item["text_path"].endswith("Wei2024.md")
    assert "text_status" not in item  # ready → path, not pending


@pytest.mark.asyncio
async def test_get_paper_known_key_pending_when_no_extract(server):
    """A paper with metadata but no extract on disk → found + text_status pending."""
    item = await _get_one(server, "Potgieter2013")
    assert item["status"] == "found"
    assert item["key"] == "Potgieter2013"
    assert item["text_status"] == "pending"
    assert "text_path" not in item


# ---- serve-safety: terminal download_status → honest text_status (D6/D7) ----


@pytest.mark.parametrize(
    "status,expected_text_status",
    [
        ("extract_failed", "extract_failed"),
        ("failed", "download_failed"),
        ("metadata_only", "metadata_only"),
        ("pending", "pending"),
    ],
)
def test_paper_dict_terminal_status_maps_to_text_status(
    populated_lib, status, expected_text_status
):
    """``_paper_dict`` surfaces a terminal download_status as the matching
    honest text_status (never a bare 'pending' for a known-dead paper), and
    NEVER sets text_path when there's no extract on disk. abstract always
    present (D6)."""
    from papervault.library.mcp.server import _paper_dict

    p = populated_lib.get("Potgieter2013")
    p.download_status = status
    rec = _paper_dict(p, populated_lib)
    assert rec["text_status"] == expected_text_status
    assert "text_path" not in rec
    assert rec["abstract"]  # abstract always returned regardless of status


def test_paper_dict_extract_on_disk_beats_terminal_status(populated_lib):
    """If a real extract is on disk, text_path wins even when the record's
    status string says something terminal (disk fact is authoritative for
    the served text)."""
    from papervault.library.mcp.server import _paper_dict

    p = populated_lib.get("Wei2024")  # has md + txt + pdf on disk
    p.download_status = "metadata_only"  # stale/contradictory status
    rec = _paper_dict(p, populated_lib)
    assert rec["text_path"].endswith("Wei2024.md")
    assert "text_status" not in rec


# ---- serve-safety: a txt-only extract serves, but a thin scan txt does not --


def test_paper_dict_txt_only_real_serves_as_text_path(populated_lib):
    """A record with NO md but a substantial txt on disk serves that txt as
    text_path (SDD §5: has_extract(txt) no md → text_path(txt))."""
    from papervault.library.mcp.server import _paper_dict

    p = populated_lib.get("Potgieter2013")  # no md/txt on disk yet
    populated_lib.txt_path("Potgieter2013").write_text("real extracted body " * 60)
    p.download_status = "ok"
    rec = _paper_dict(p, populated_lib)
    assert rec["text_path"].endswith("Potgieter2013.txt")
    assert "text_status" not in rec


def test_paper_dict_thin_txt_only_does_not_impersonate_full_text(populated_lib):
    """A scanned PDF yields a near-empty txt. Serve-safety must NOT hand that
    out as text_path (D6: never let an almost-empty scan txt impersonate full
    text) — it falls through to an honest text_status instead, abstract intact."""
    from papervault.library.mcp.server import _paper_dict

    p = populated_lib.get("Potgieter2013")  # no md on disk
    populated_lib.txt_path("Potgieter2013").write_text("\f \n")  # scan noise
    p.download_status = "ok"
    rec = _paper_dict(p, populated_lib)
    assert "text_path" not in rec
    assert rec["text_status"] == "pending"
    assert rec["abstract"]  # abstract always returned (D6)


def test_paper_dict_thin_txt_only_terminal_status_shows_through(populated_lib):
    """A thin txt on a terminal record falls through to the terminal
    text_status, not a bare 'pending' (honest about why there's no full text)."""
    from papervault.library.mcp.server import _paper_dict

    p = populated_lib.get("Potgieter2013")
    populated_lib.txt_path("Potgieter2013").write_text("x")  # 1 byte
    p.download_status = "extract_failed"
    rec = _paper_dict(p, populated_lib)
    assert "text_path" not in rec
    assert rec["text_status"] == "extract_failed"


@pytest.mark.parametrize(
    "status,expected_text_status",
    [
        ("extract_failed", "extract_failed"),
        ("failed", "download_failed"),
        ("metadata_only", "metadata_only"),
    ],
)
def test_paper_dict_fat_txt_under_terminal_status_is_not_served(
    populated_lib, status, expected_text_status
):
    """FAIL-CLOSED (lens-3 driller): a SUBSTANTIAL (≥ byte floor) txt on a
    TERMINAL record must NOT be served as text_path.

    A leftover on-disk txt (e.g. one of the no-pdf migration rows, or any
    historical txt) on a record the pipeline flipped to a terminal status is the
    SAME source the completeness gate / extraction judged unusable. Even though
    it can be well over the byte floor, serving it as text_path would impersonate
    full text the pipeline explicitly said it does not have (violates §4.3
    text_path ⟺ real ∧ gated, the text_path XOR text_status invariant, and D6).
    Serve-safety must fall through to the honest terminal text_status. (The new
    txt door is also narrowed to ¬has_pdf ∧ ¬has_md, SDD §3.4 — but the terminal
    guard alone already fail-closes this case.)"""
    from papervault.library.mcp.server import _paper_dict

    p = populated_lib.get("Potgieter2013")  # no md on disk
    # A fat txt — a paywall/truncated PDF's pypdf dump, well over the 500-byte
    # serve floor, but NOT a real complete extract.
    populated_lib.txt_path("Potgieter2013").write_text("paywalled fragment " * 80)
    assert populated_lib.txt_path("Potgieter2013").stat().st_size >= 500
    p.download_status = status
    rec = _paper_dict(p, populated_lib)
    assert "text_path" not in rec, "fake full-text path served under terminal status"
    assert rec["text_status"] == expected_text_status
    assert rec["abstract"]  # abstract still returned (D6)


@pytest.mark.asyncio
async def test_get_paper_is_minimal_record(server):
    """Found items expose ONLY the minimal record fields — no insight /
    in_library_state / has_pdf / eta leak into the executor's context."""
    item = await _get_one(server, "Wei2024")
    extra = set(item) - MIN_FIELDS - {"identifier", "status"}
    assert extra == set(), f"unexpected fields leaked: {extra}"


@pytest.mark.asyncio
async def test_get_paper_batch_returns_one_per_identifier(server):
    """A list of identifiers → one result per identifier, in order, each tagged."""
    out = await _call(server, "get_paper",
                      {"identifiers": ["Wei2024", "Potgieter2013"]})
    assert out["status"] == "ok"
    assert [r["identifier"] for r in out["results"]] == ["Wei2024", "Potgieter2013"]
    assert [r["key"] for r in out["results"]] == ["Wei2024", "Potgieter2013"]
    assert all(r["status"] == "found" for r in out["results"])


@pytest.mark.asyncio
async def test_get_paper_doi_in_library_short_circuits(populated_lib, monkeypatch):
    """An existing DOI resolves locally; fetch_by_doi must NOT be called."""
    fetch_calls = []

    def tracking_fetch_by_doi(doi):
        fetch_calls.append(doi)
        return None

    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", tracking_fetch_by_doi)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)

    from papervault.library.mcp.server import build_server
    server = build_server(library=populated_lib)

    item = await _get_one(server, "10.1234/abc")
    assert item["status"] == "found"
    assert item["key"] == "Potgieter2013"
    assert fetch_calls == []  # short-circuited; no network


@pytest.mark.asyncio
async def test_get_paper_new_doi_is_not_found_read_only(populated_lib, monkeypatch):
    """get_paper is READ-ONLY: a well-formed DOI not in the library returns not_found,
    NEVER fetches, and writes nothing. search_papers is the only ingest path."""
    fetch_calls = []

    def tracking_fetch_by_doi(doi):
        fetch_calls.append(doi)  # must never be called
        return {"title": "should never be fetched", "doi": doi}

    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", tracking_fetch_by_doi)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)

    from papervault.library.mcp.server import build_server
    server = build_server(library=populated_lib)

    before = len(populated_lib.all_papers())
    item = await _get_one(server, "10.9999/new")
    assert item["status"] == "not_found"
    assert item["identifier"] == "10.9999/new"
    assert fetch_calls == []                              # read-only: never fetched
    assert len(populated_lib.all_papers()) == before     # wrote nothing
    assert "hints" in item                               # not_found is augmented


@pytest.mark.asyncio
async def test_get_paper_new_arxiv_is_not_found_read_only(populated_lib, monkeypatch):
    """get_paper is READ-ONLY: an arxiv id not in the library returns not_found,
    NEVER fetches, and writes nothing."""
    arxiv_calls = []

    def tracking_fetch_by_arxiv(a):
        arxiv_calls.append(a)  # must never be called
        return {"title": "should never be fetched", "arxiv_id": a}

    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", lambda d: None)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", tracking_fetch_by_arxiv)

    from papervault.library.mcp.server import build_server
    server = build_server(library=populated_lib)

    before = len(populated_lib.all_papers())
    item = await _get_one(server, "2401.99999")
    assert item["status"] == "not_found"
    assert arxiv_calls == []                              # read-only: never fetched
    assert len(populated_lib.all_papers()) == before     # wrote nothing


@pytest.mark.asyncio
async def test_get_paper_fuzzy_single_match(populated_lib, monkeypatch):
    """Fuzzy text resolving to one library entry returns it."""
    class FakeLLM:
        def call(self, msgs):
            return '{"matches": [{"i": 1, "confidence": 0.95}]}'

    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", lambda d: None)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)
    monkeypatch.setattr("papervault.library.llm.get_llm", lambda: FakeLLM())

    from papervault.library.mcp.server import build_server
    server = build_server(library=populated_lib)

    item = await _get_one(server, "cosmic ray neural")
    assert item["status"] == "found"
    assert item["key"] == "Wei2024"


@pytest.mark.asyncio
async def test_get_paper_fuzzy_ambiguous(populated_lib, monkeypatch):
    """Fuzzy text matching multiple papers → ambiguous item + candidates + hints."""
    class FakeLLM:
        def call(self, msgs):
            return ('{"matches": [{"i": 1, "confidence": 0.8},'
                    '{"i": 2, "confidence": 0.75}]}')

    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", lambda d: None)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)
    monkeypatch.setattr("papervault.library.llm.get_llm", lambda: FakeLLM())

    from papervault.library.mcp.server import build_server
    server = build_server(library=populated_lib)

    item = await _get_one(server, "cosmic solar review")
    assert item["status"] == "ambiguous"
    assert len(item["candidates"]) >= 2
    assert "hints" in item


@pytest.mark.asyncio
async def test_get_paper_not_found(server):
    """A string that's not a DOI/arxiv and matches no library paper → not_found + hints."""
    item = await _get_one(server, "qwertyzzz nonsense-token-xyzzy")
    assert item["status"] == "not_found"
    assert item["identifier"] == "qwertyzzz nonsense-token-xyzzy"
    hints_text = " ".join(item["hints"])
    assert "search_papers" in hints_text


@pytest.mark.asyncio
async def test_get_paper_empty_identifier(server):
    """Empty string short-circuits to not_found."""
    item = await _get_one(server, "")
    assert item["status"] == "not_found"


@pytest.mark.asyncio
async def test_get_paper_empty_identifier_hint_does_not_overclaim(server):
    """Boundary fix #7: the empty-identifier not_found hint must NOT assert a
    fuzzy step ran (it short-circuits before fuzzy) nor cite a phantom 0.7
    floor. It says 'empty identifier' and points at the real next step."""
    item = await _get_one(server, "   ")
    assert item["status"] == "not_found"
    hints_text = " ".join(item["hints"]).lower()
    assert "empty identifier" in hints_text
    assert "0.7" not in hints_text                 # phantom floor removed
    assert "fuzzy resolution found no" not in hints_text  # no step-that-didn't-run claim


# ---- drill fixes #1/#2/#4: exact identifier-form resolution ----------------


@pytest.mark.asyncio
async def test_get_paper_arxiv_prefixed_form_resolves_same_as_bare(monkeypatch, tmp_path):
    """Boundary fix #1: get_paper('arXiv:1711.10561') resolves to the SAME held
    paper as the bare '1711.10561' — the canonical prefixed form no longer
    falsely reports not-in-library."""
    lib = Library(tmp_path)
    lib.upsert({"title": "Physics-informed deep learning for PDE inversion",
                "authors": ["Raissi"], "year": 2019, "arxiv_id": "1711.10561v9",
                "abstract": "PINN inversion."})
    lib.save()
    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])

    async def fake_ext(terms, **k):
        return [], {}
    monkeypatch.setattr("papervault.library.mcp.server.search_external_async", fake_ext)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", lambda d: None)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)
    from papervault.library.mcp.server import build_server
    srv = build_server(library=lib)

    bare = await _get_one(srv, "1711.10561")
    prefixed = await _get_one(srv, "arXiv:1711.10561")
    assert bare["status"] == "found"
    assert prefixed["status"] == "found"
    assert prefixed["key"] == bare["key"]            # same held paper, not not_found


@pytest.mark.asyncio
async def test_get_paper_doi_prefixed_and_url_forms_resolve(server):
    """Boundary fix #2: 'doi:10.1234/abc' and the resolver-URL form resolve to
    the SAME held paper as the bare DOI (Potgieter2013 in the fixture), instead
    of falling to the fuzzy path → not_found."""
    bare = await _get_one(server, "10.1234/abc")
    prefixed = await _get_one(server, "doi:10.1234/abc")
    url = await _get_one(server, "https://doi.org/10.1234/abc")
    dx_url = await _get_one(server, "http://dx.doi.org/10.1234/abc")
    assert bare["status"] == "found" and bare["key"] == "Potgieter2013"
    for form in (prefixed, url, dx_url):
        assert form["status"] == "found"
        assert form["key"] == "Potgieter2013"


@pytest.mark.asyncio
async def test_get_paper_exact_title_resolves_without_fuzzy(populated_lib, monkeypatch):
    """Boundary fix #4: a VERBATIM title resolves via the exact _by_title index
    to ITS paper, BEFORE (and instead of) the LLM fuzzy resolver. The fuzzy LLM
    here would wrongly point at the OTHER paper if consulted — proving the exact
    index short-circuits ahead of it."""
    # Fuzzy LLM is rigged to confidently pick candidate #1 (the WRONG paper) —
    # if the exact-title step were skipped this would mis-resolve.
    llm = _FakeLLM('{"matches": [{"i": 1, "confidence": 0.99}]}')
    server = _fuzzy_server(populated_lib, monkeypatch, llm)
    item = await _get_one(server, "Cosmic ray PINN")   # Wei2024's exact title
    assert item["status"] == "found"
    assert item["key"] == "Wei2024"                    # the correct paper, exact hit
    assert item["title"] == "Cosmic ray PINN"


@pytest.mark.asyncio
async def test_get_paper_typo_title_still_goes_to_fuzzy(populated_lib, monkeypatch):
    """Boundary fix #4 guard: a TYPO'd / near title misses the exact index and
    still flows to the fuzzy resolver (unchanged) — here it returns ambiguous."""
    # Sub-floor fuzzy confidence (0.80 < 0.85) → ambiguous, proving fuzzy ran.
    llm = _FakeLLM('{"matches": [{"i": 1, "confidence": 0.80}]}')
    server = _fuzzy_server(populated_lib, monkeypatch, llm)
    item = await _get_one(server, "Cosmik ray PIN")    # NOT an exact-title key
    assert item["status"] == "ambiguous"


# ===== drill fixes: ambiguous one-projection / enqueue / hints / dedup ======


def _fuzzy_server(populated_lib, monkeypatch, llm):
    """Build a server whose resolver LLM is `llm` (external IO mocked off)."""
    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", lambda d: None)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)
    monkeypatch.setattr("papervault.library.llm.get_llm", lambda: llm)
    from papervault.library.mcp.server import build_server
    return build_server(library=populated_lib)


class _FakeLLM:
    def __init__(self, response):
        self._response = response

    def call(self, msgs):
        return self._response


@pytest.mark.asyncio
async def test_get_paper_ambiguous_candidates_are_one_projection(populated_lib, monkeypatch):
    """fix #1 (SDD §5 I-PROJ): ambiguous candidates must be the SAME minimal
    _paper_dict projection as found/search — NOT the raw resolver _candidate_dict.
    No score / has_pdf / has_extract_md / in_library / is_review leak; venue /
    doi / arxiv_id / a text reference ARE present."""
    llm = _FakeLLM('{"matches": [{"i": 1, "confidence": 0.8},'
                   '{"i": 2, "confidence": 0.75}]}')
    server = _fuzzy_server(populated_lib, monkeypatch, llm)
    item = await _get_one(server, "cosmic solar review")
    assert item["status"] == "ambiguous"
    cands = item["candidates"]
    assert len(cands) >= 2
    for c in cands:
        leaked = {"score", "score_kind", "has_pdf", "has_extract_md",
                  "in_library", "is_review"} & set(c)
        assert leaked == set(), f"raw resolver fields leaked into candidate: {leaked}"
        # The minimal-record contract: 9 fields + exactly one text reference.
        assert set(c) - MIN_FIELDS == set(), f"unexpected fields: {set(c) - MIN_FIELDS}"
        assert ("text_path" in c) ^ ("text_status" in c)
        assert "key" in c and "doi" in c  # citable fields restored


@pytest.mark.asyncio
async def test_get_paper_ambiguous_single_candidate_lead_hint(populated_lib, monkeypatch):
    """fix #5: when the ambiguous branch fires for ONE sub-threshold candidate,
    the lead hint is the actionable 'call get_paper(<key>)', NOT the wrong
    'Multiple library papers match'."""
    # One in-library candidate at 0.80 (below the 0.85 auto-resolve floor) → ambiguous.
    llm = _FakeLLM('{"matches": [{"i": 1, "confidence": 0.8}]}')
    server = _fuzzy_server(populated_lib, monkeypatch, llm)
    item = await _get_one(server, "cosmic ray neural")
    assert item["status"] == "ambiguous"
    assert len(item["candidates"]) == 1
    lead = item["hints"][0]
    assert "Multiple library papers match" not in lead
    assert "get_paper(" in lead


@pytest.mark.asyncio
async def test_get_paper_score_gap_escape_auto_resolves(populated_lib, monkeypatch):
    """fix #4: a dominant #1 (≥0.85) leading #2 by ≥0.2 auto-resolves to FOUND
    instead of bouncing to ambiguous on count==2."""
    # #1 = 0.98 (Wei2024 idx depends on candidate order; both in-library, big gap).
    llm = _FakeLLM('{"matches": [{"i": 1, "confidence": 0.98},'
                   '{"i": 2, "confidence": 0.50}]}')
    server = _fuzzy_server(populated_lib, monkeypatch, llm)
    item = await _get_one(server, "cosmic solar review")
    assert item["status"] == "found"


@pytest.mark.asyncio
async def test_get_paper_close_pair_stays_ambiguous(populated_lib, monkeypatch):
    """fix #4 guard: a genuinely close #1-vs-#2 pair (gap < 0.2) still returns
    ambiguous — the gap escape only fires for a clear dominant."""
    llm = _FakeLLM('{"matches": [{"i": 1, "confidence": 0.90},'
                   '{"i": 2, "confidence": 0.85}]}')
    server = _fuzzy_server(populated_lib, monkeypatch, llm)
    item = await _get_one(server, "cosmic solar review")
    assert item["status"] == "ambiguous"


def test_should_auto_resolve_keyword_floor_is_stricter():
    """fix #7a: the auto-resolve gate holds keyword-derived scores to a STRICTER
    floor than LLM scores. A 0.90 score auto-resolves on the LLM path but NOT on
    the keyword path (0.95 floor) — so a token-saturated WRONG paper can't
    auto-`found` on overlap alone during an LLM outage."""
    from papervault.library.mcp.server import _should_auto_resolve
    # 0.90, sole candidate, LLM provenance → auto-resolve (≥0.85).
    assert _should_auto_resolve([{"key": "A", "score": 0.90, "score_kind": "llm"}])
    # SAME 0.90 from the keyword fallback → NOT auto-resolved (needs ≥0.95).
    assert not _should_auto_resolve(
        [{"key": "A", "score": 0.90, "score_kind": "keyword"}])
    # A keyword score ≥0.95, sole candidate → still allowed (the floor, not a ban).
    assert _should_auto_resolve(
        [{"key": "A", "score": 0.97, "score_kind": "keyword"}])


def test_should_auto_resolve_score_gap_escape():
    """fix #4: a dominant #1 (≥0.85) leading #2 by ≥0.2 auto-resolves even with a
    second in-library candidate; a close pair (gap <0.2) does not."""
    from papervault.library.mcp.server import _should_auto_resolve
    assert _should_auto_resolve([
        {"key": "A", "score": 0.98, "score_kind": "llm"},
        {"key": "B", "score": 0.50, "score_kind": "llm"}])      # gap 0.48 → resolve
    assert not _should_auto_resolve([
        {"key": "A", "score": 0.90, "score_kind": "llm"},
        {"key": "B", "score": 0.85, "score_kind": "llm"}])      # gap 0.05 → ambiguous
    assert not _should_auto_resolve([
        {"key": "A", "score": 0.80, "score_kind": "llm"}])      # below floor → ambiguous


@pytest.mark.asyncio
async def test_get_paper_new_doi_not_found_drops_fuzzy_hints(populated_lib, monkeypatch):
    """fix #6: a well-formed-but-not-held DOI keeps ONLY the search_papers
    pointer; the false 'did not match a DOI pattern' / 'try Lastname Year'
    fuzzy hints are dropped."""
    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", lambda d: None)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)
    from papervault.library.mcp.server import build_server
    server = build_server(library=populated_lib)
    item = await _get_one(server, "10.9999/not-held")
    assert item["status"] == "not_found"
    hints_text = " ".join(item["hints"])
    assert "search_papers" in hints_text
    assert "Lastname Year" not in hints_text
    assert "did not match" not in hints_text


@pytest.mark.asyncio
async def test_get_paper_fuzzy_miss_keeps_fuzzy_hints(server):
    """fix #6 guard: a genuine fuzzy-text miss (not DOI/arxiv-shaped) STILL gets
    the fuzzy hints — only the DOI/arxiv case is trimmed."""
    item = await _get_one(server, "qwertyzzz nonsense-token-xyzzy")
    assert item["status"] == "not_found"
    hints_text = " ".join(item["hints"])
    assert "Lastname Year" in hints_text or "Try a more specific" in hints_text
    assert "search_papers" in hints_text


# ---- fix #3 / #12: terminal enqueue gate + batch dedup (stub queues) --------


class _RecordingQueue:
    """Minimal queue stub that records add(key, priority) calls and reports
    _started=True so the get_paper enqueue branch runs."""
    def __init__(self):
        self._started = True
        self.added: list[tuple] = []

    def add(self, key, priority=None):
        self.added.append((key, priority))


def _server_with_recording_queues(populated_lib, monkeypatch):
    monkeypatch.setattr("papervault.library.mcp.server.search_all", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.fetch.fetch_by_doi", lambda d: None)
    monkeypatch.setattr("papervault.library.fetch.fetch_by_arxiv", lambda a: None)
    from papervault.library.mcp.server import build_server
    dq, eq = _RecordingQueue(), _RecordingQueue()
    server = build_server(library=populated_lib, download_queue=dq, extract_queue=eq)
    return server, dq, eq


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["metadata_only", "failed", "extract_failed"])
async def test_get_paper_terminal_paper_not_re_enqueued(populated_lib, monkeypatch, status):
    """fix #3 (SDD §5 I-TERM-ENQ): a TERMINAL paper is never re-enqueued for
    URGENT re-download / re-OCR from the foreground get_paper path."""
    p = populated_lib.get("Potgieter2013")  # no pdf, no md on disk
    p.download_status = status
    populated_lib.save()
    server, dq, eq = _server_with_recording_queues(populated_lib, monkeypatch)
    item = await _get_one(server, "Potgieter2013")
    assert item["status"] == "found"
    assert dq.added == [], "terminal paper re-enqueued for URGENT download"
    assert eq.added == [], "terminal paper re-enqueued for URGENT extract"


@pytest.mark.asyncio
async def test_get_paper_pending_paper_is_enqueued(populated_lib, monkeypatch):
    """fix #3 guard: a NON-terminal (pending) paper missing its PDF is still
    URGENT-enqueued for download — the gate fail-closes terminals, not progress."""
    p = populated_lib.get("Potgieter2013")
    p.download_status = "pending"
    populated_lib.save()
    server, dq, eq = _server_with_recording_queues(populated_lib, monkeypatch)
    await _get_one(server, "Potgieter2013")
    assert [k for k, _ in dq.added] == ["Potgieter2013"]


@pytest.mark.asyncio
async def test_get_paper_batch_dedup_aliasing_identifiers(populated_lib, monkeypatch):
    """fix #12: two identifiers aliasing the same paper (key + its DOI) yield
    ONE record and ONE URGENT enqueue, not two."""
    p = populated_lib.get("Potgieter2013")
    p.download_status = "pending"
    populated_lib.save()
    server, dq, eq = _server_with_recording_queues(populated_lib, monkeypatch)
    out = await _call(server, "get_paper",
                      {"identifiers": ["Potgieter2013", "10.1234/abc"]})
    found = [r for r in out["results"] if r.get("status") == "found"]
    assert len(found) == 1, "aliasing identifiers returned duplicate records"
    assert found[0]["identifier"] == "Potgieter2013"  # first echo kept
    assert len([k for k, _ in dq.added if k == "Potgieter2013"]) == 1  # one enqueue


# ============== search_papers (3-LLM, hermetic) =============================


# V6 (Stage D): search_external_async returns TAGGED ranked-lists — every node
# carries {_source_origin="external", term_idx, rank(native)} stamped at fetch
# time. The §4a fold rebuilds per-term buckets from those tags; the §4c
# round-robin fair-shares across terms; the ingest gate runs on ext_pool.
def _fake_plan(**over):
    # TWO distinct sub-topics so the round-robin's per-term fair-share is
    # exercised (term 0 = a DEEP ranked list, term 1 = ONE niche paper).
    plan = {"search_terms": ["cosmic ray pinn transport", "gcr solar cycle forecasting"],
            "filters": {}, "limit_suggested": 5,
            "ranking_hint": "by_relevance", "reasoning": "test"}
    plan.update(over)
    return plan


# The internal pipeline tags that must NEVER leak into an output record.
_INTERNAL_TAGS = {"_source_origin", "term_idx", "rank", "term_ranks", "_rrf",
                  "paper_id", "url", "source", "_bm25_score", "publication_types"}

# Term-0 candidates: an in-domain real paper (rank 0), an in-domain DATASET
# (metadata says non-paper, rank 1), an off-domain medical paper (rank 2), plus
# DEEP filler in-domain papers (ranks 3+) that would crowd out term 1 WITHOUT
# fair-share. Term-1 candidate: a single niche paper (rank 0) — fair-share must
# surface it despite term-0's much longer list.
def _term0_cands():
    cands = [
        {"title": "Cosmic ray PINN transport inversion via physics-informed nets",
         "authors": ["A"], "year": 2024, "venue": "ApJ",
         "abstract": "PINN inversion of cosmic ray transport.", "doi": "10.1/a",
         "arxiv_id": "", "citation_count": 3, "publication_types": ["JournalArticle"]},
        {"title": "Cosmic ray PINN transport dataset 2024",
         "authors": ["B"], "year": 2024, "venue": "Zenodo",
         "abstract": "A dataset of cosmic ray transport simulations.", "doi": "10.1/b",
         "arxiv_id": "", "citation_count": 1, "publication_types": ["Dataset"]},
        {"title": "Medical trial of cosmic-themed PINN transport drug",
         "authors": ["C"], "year": 2024, "venue": "NEJM",
         "abstract": "A clinical trial unrelated to space physics.", "doi": "10.1/c",
         "arxiv_id": "", "citation_count": 9, "publication_types": ["JournalArticle"]},
    ]
    # Deep in-domain filler so term 0's list dwarfs term 1's single entry.
    for i in range(40):
        cands.append(
            {"title": f"Cosmic ray transport filler study {i}",
             "authors": ["F"], "year": 2023, "venue": "JGR",
             "abstract": "transport filler", "doi": f"10.2/f{i}",
             "arxiv_id": "", "citation_count": 0,
             "publication_types": ["JournalArticle"]})
    return cands


_TERM1_NICHE = {
    "title": "GCR solar cycle long-horizon forecasting with neural processes",
    "authors": ["N"], "year": 2024, "venue": "SpaceWeather",
    "abstract": "Neural-process forecasting of galactic cosmic ray solar-cycle modulation.",
    "doi": "10.3/niche", "arxiv_id": "", "citation_count": 7,
    "publication_types": ["JournalArticle"]}


def _tag(c, term_idx, rank):
    """Stamp the fan-out provenance tags exactly as ``search_external_async``
    does (``_source_origin``/``term_idx``/``rank``), on a fresh copy."""
    d = dict(c)
    d.update(_source_origin="external", term_idx=term_idx, rank=rank)
    return d


def _fan_out():
    """The flat, TAGGED ``ext_raw`` that the V6 ``search_external_async`` returns:
    term 0's deep ranked list concatenated with term 1's single niche entry."""
    raw = [_tag(c, 0, r) for r, c in enumerate(_term0_cands())]
    raw.append(_tag(_TERM1_NICHE, 1, 0))
    return raw


def _wire_search(monkeypatch, *, ingest, ret):
    """Patch search_papers' collaborators: intent (sync), external fan-out
    (async, now TAGGED ranked-lists), and the two judge LLMs (async)."""
    monkeypatch.setattr("papervault.library.mcp.server.parse_intent",
                        lambda query, llm=None: _fake_plan())

    async def fake_external(terms, *, year_min=None, year_max=None,
                            ranking_hint="by_relevance"):
        return _fan_out(), {}   # V6: (ext_raw, degraded_map) 2-tuple (§3)
    monkeypatch.setattr("papervault.library.mcp.server.search_external_async", fake_external)
    monkeypatch.setattr("papervault.library.mcp.server.judge_ingest", ingest)
    monkeypatch.setattr("papervault.library.mcp.server.judge_return", ret)


@pytest.mark.asyncio
async def test_search_malformed_intent_returns_error(server, monkeypatch):
    """§1: a fully-unparseable intent (parse_intent raises ValueError) returns a
    structured {status: "error"} — it never throws a raw exception out of the tool,
    and never reaches the external fan-out / judges."""
    def boom(query, llm=None):
        raise ValueError("no JSON object in LLM response")
    monkeypatch.setattr("papervault.library.mcp.server.parse_intent", boom)

    out = await _call(server, "search_papers", {"query": "anything at all"})
    assert out["status"] == "error"
    assert out["results"] == []
    assert "message" in out


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", "\t\n  "])
async def test_search_empty_query_fails_closed_no_ingest(
        server, populated_lib, monkeypatch, query):
    """Boundary fix #3: an empty / whitespace-only query MUST fail CLOSED with
    {status:error} BEFORE any intent parse, external fan-out, or ingest — and it
    must NOT mutate the library (the drill saw '   ' INGEST a stray paper)."""
    # Tripwires: if the guard leaks, the pipeline would touch these.
    def boom_parse(*a, **k):
        raise AssertionError("parse_intent ran on an empty query — guard leaked")
    monkeypatch.setattr("papervault.library.mcp.server.parse_intent", boom_parse)

    async def boom_ext(*a, **k):
        raise AssertionError("external fan-out ran on an empty query")
    monkeypatch.setattr("papervault.library.mcp.server.search_external_async", boom_ext)
    monkeypatch.setattr(
        "papervault.library.mcp.server.judge_ingest",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ingest gate ran on empty query")))

    keys_before = set(populated_lib.keys())
    out = await _call(server, "search_papers", {"query": query})
    assert out["status"] == "error"
    assert out["results"] == []
    assert "message" in out
    # No mutation: same library on disk and in memory.
    assert set(populated_lib.keys()) == keys_before
    from papervault.library import Library
    assert set(Library(populated_lib.root).keys()) == keys_before


def test_parse_intent_coerces_string_year_bounds():
    """A model that returns year_min/year_max (or citation_pref) as a string or
    float must NOT reach YEAR_DROP uncoerced — int comparison against a str would
    TypeError-crash the whole search. parse_intent coerces to int; an uncoercible
    bound degrades to None (= no bound)."""
    from papervault.library.services.intent_parser import parse_intent

    class FakeLLM:
        def __init__(self, payload):
            self._payload = payload

        def call(self, msgs):
            return self._payload

    plan = parse_intent("recent work", FakeLLM(
        '{"search_terms": ["x"], "filters": {"year_min": "2023", "year_max": 2024.0, '
        '"citation_pref": "5", "review_pref": null}, "limit_suggested": null, '
        '"ranking_hint": "by_relevance", "reasoning": ""}'))
    assert plan["filters"]["year_min"] == 2023
    assert isinstance(plan["filters"]["year_min"], int)
    assert plan["filters"]["year_max"] == 2024
    assert plan["filters"]["citation_pref"] == 5

    plan2 = parse_intent("junk bound", FakeLLM(
        '{"search_terms": ["x"], "filters": {"year_min": "not-a-year", "year_max": null, '
        '"citation_pref": null, "review_pref": null}, "limit_suggested": null, '
        '"ranking_hint": "by_relevance", "reasoning": ""}'))
    assert plan2["filters"]["year_min"] is None


def test_parse_intent_fail_closed_on_no_json():
    """Fail-CLOSED at the tool boundary (SDD §5/§8): a reply with NO ``{...}``
    object at all must RAISE ValueError — parse_intent must NOT silently fall
    back to a ``[query]`` plan. The S0 raise is what lets the caller return
    ``{status: error}`` instead of running a fan-out on a hallucinated plan.
    Pins the invariant so a non-greedy-regex / silent-fallback regression is
    caught at the PARSER level (the caller-wrapper test only mocks parse_intent
    to raise — it does not drive the parser itself)."""
    from papervault.library.services.intent_parser import parse_intent

    class FakeLLM:
        def __init__(self, payload):
            self._payload = payload

        def call(self, msgs):
            return self._payload

    # Prose-only reply, no JSON object anywhere.
    with pytest.raises(ValueError):
        parse_intent("anything", FakeLLM(
            "I'm sorry, I can't produce a search plan for that request."))


def test_parse_intent_fail_closed_on_malformed_json():
    """Fail-CLOSED at the tool boundary (SDD §5/§8): a reply that DOES contain a
    ``{...}`` span but whose contents are not valid JSON (e.g. an unquoted token)
    must RAISE ValueError — the greedy-brace extract finds a span, but
    ``json.loads`` fails and the parser must NOT degrade to a ``[query]`` plan.
    Guards the malformed-JSON arm of the fail-CLOSED contract."""
    from papervault.library.services.intent_parser import parse_intent

    class FakeLLM:
        def __init__(self, payload):
            self._payload = payload

        def call(self, msgs):
            return self._payload

    # A brace span is present, but the JSON inside is malformed (unquoted token).
    with pytest.raises(ValueError):
        parse_intent("anything", FakeLLM('{"search_terms": [unquoted]}'))


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_parse_intent_empty_query_never_builds_whitespace_anchor(query):
    """Boundary fix #3 (defense-in-depth): when the LLM returns empty
    search_terms AND the query is itself empty-after-strip, the parser RAISES
    rather than re-injecting an all-whitespace anchor term. It never produces a
    blank-only search_terms entry."""
    from papervault.library.services.intent_parser import parse_intent

    class FakeLLM:
        def __init__(self, payload):
            self._payload = payload

        def call(self, msgs):
            return self._payload

    # LLM correctly returns no usable terms for an empty intent.
    with pytest.raises(ValueError):
        parse_intent(query, FakeLLM(
            '{"search_terms": [], "filters": {}, "limit_suggested": null, '
            '"ranking_hint": "by_relevance", "reasoning": "empty"}'))


def test_parse_intent_empty_terms_nonempty_query_reinjects_stripped_anchor():
    """Boundary fix #3 guard: when the LLM returns no terms but the query is
    NON-empty, the parser re-injects the STRIPPED query as the single anchor
    term — never the raw unstripped string, never a blank entry."""
    from papervault.library.services.intent_parser import parse_intent

    class FakeLLM:
        def call(self, msgs):
            return ('{"search_terms": [], "filters": {}, "limit_suggested": null, '
                    '"ranking_hint": "by_relevance", "reasoning": ""}')

    plan = parse_intent("   cosmic ray transport   ", FakeLLM())
    assert plan["search_terms"] == ["cosmic ray transport"]   # stripped, single anchor


def test_parse_intent_a1_anchor_term0_survives_normalization():
    """A1 anchoring: the LLM-emitted term-0 (the verbatim CORE concept) stays at
    index 0 after the strip → order-preserving-dedup → prefix-cap normalization.
    Even with leading blanks, a later exact-duplicate of term-0, and >8 terms,
    the anchor remains ``search_terms[0]``."""
    from papervault.library.services.intent_parser import parse_intent

    class FakeLLM:
        def __init__(self, payload):
            self._payload = payload

        def call(self, msgs):
            return self._payload

    # term-0 = the anchor; a blank slips in, the anchor is exact-duped later, and
    # the list overflows the cap — the anchor must still head the result.
    terms = (['  cosmic ray transport  ', '', 'physics-informed neural network',
              'cosmic ray transport',  # exact dup of the (stripped) anchor
              'Parker equation', 'SEP inversion', 'heliosphere modulation',
              'neutron monitor', 'solar maximum', 'extra facet nine', 'extra facet ten'])
    payload = ('{"search_terms": %s, "filters": {}, "limit_suggested": null, '
               '"ranking_hint": "by_relevance", "reasoning": ""}'
               % json.dumps(terms))
    plan = parse_intent("...", FakeLLM(payload))
    assert plan["search_terms"][0] == "cosmic ray transport"   # anchor at index 0
    assert len(plan["search_terms"]) <= 8                       # prefix cap honored
    # order-preserving dedup dropped the exact dup but kept distinct facets
    assert plan["search_terms"].count("cosmic ray transport") == 1
    assert "physics-informed neural network" in plan["search_terms"]


def test_parse_intent_a3_prompt_carries_facet_and_anchor_rules():
    """A3/A1: the system prompt instructs facet-orthogonality (distinct facets,
    not synonym rewordings) + the over-split guard + the term-0 anchor rule. This
    pins the contract the prompt is responsible for (a real LLM is needed to
    exercise the behavior; this guards against an accidental prompt regression)."""
    from papervault.library.services.intent_parser import _SYSTEM_PROMPT

    p = _SYSTEM_PROMPT.lower()
    assert "anchor" in p and "term 0" in p                      # A1 anchor rule
    assert "facet" in p                                         # A3 facet vocabulary
    assert "over-split" in p or "fake facet" in p               # over-split guard
    # the four facet axes are named so the LLM has the taxonomy
    for axis in ("phenomenon", "method", "system", "regime"):
        assert axis in p, f"facet axis {axis!r} missing from prompt"


@pytest.mark.asyncio
async def test_search_ingest_gate_and_minimal_records(server, populated_lib, monkeypatch):
    """The ingest gate drops the dataset (metadata) and the medical paper (tier 3);
    the in-domain paper is ingested; results are pure minimal records (no tag leak)."""
    async def fake_ingest(cands, *, llm=None):
        # Every cand is a TAGGED external node, never a library node (no re-judging in-library nodes).
        assert all(c.get("_source_origin") == "external" for c in cands)
        out = {}
        for i, c in enumerate(cands):
            tier = "3" if "medical" in c["title"].lower() else "1A"
            out[i] = {"reason": "", "tier": tier, "ingest_ok": tier != "3",
                      "llm_is_paper": True}  # dataset relies on metadata gate
        return out, 0   # V6: (judgments, judge_batches_dropped) 2-tuple (§8)

    async def fake_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        return {i: {"reason": "", "score": 0.9} for i in range(len(cands))}, 0

    _wire_search(monkeypatch, ingest=fake_ingest, ret=fake_return)
    server._paper_download_queue.add = lambda key, **kw: None  # no-op enqueue

    out = await _call(server, "search_papers", {"query": "cosmic ray pinn inversion"})
    assert out["status"] == "ok"

    titles = [p.title for p in populated_lib.all_papers()]
    assert any("physics-informed nets" in t for t in titles)        # in-domain → ingested
    assert not any("dataset 2024" in t for t in titles)             # Dataset metadata → dropped
    assert not any(t.startswith("Medical trial") for t in titles)   # tier 3 → dropped
    # FAIR-SHARE: term 1's single niche paper is NOT starved by term 0's deep
    # list — the round-robin interleaves terms, so it reaches the ingest gate
    # and is ingested.
    assert any("long-horizon forecasting with neural processes" in t for t in titles)

    # NO-LEAK: output records are minimal; no internal pipeline tag leaks.
    for r in out["results"]:
        assert set(r) <= MIN_FIELDS, f"non-minimal field leaked: {set(r) - MIN_FIELDS}"
        assert not (_INTERNAL_TAGS & set(r)), f"internal tag leaked: {_INTERNAL_TAGS & set(r)}"
    assert "stats" not in out and "query" not in out  # lean envelope

    # V6 §8: source-health signals are surfaced in intent_parsed. The mocked
    # fan-out degraded nothing, so sources_degraded is empty; sources_unconfigured
    # depends on env (always-configured-when-absent backends never appear).
    ip = out["intent_parsed"]
    assert ip["sources_degraded"] == []
    assert isinstance(ip["sources_unconfigured"], list)


@pytest.mark.asyncio
async def test_search_ingest_plugs_reject_egu_abstract_and_contentless_stub(
        server, populated_lib, monkeypatch):
    """Junk-ingress PLUGs (2026-06-03): an EGU conference abstract (egusphere-egu DOI)
    and a content-less ghost stub (no doi/arxiv/abstract) are NOT ingested EVEN WHEN
    the gate's is_paper would wave them in; a normal in-domain paper still IS."""
    egu = {"title": "EGU abstract on SEP transport", "authors": ["A"], "year": 2025,
           "venue": "EGU General Assembly", "abstract": "Conference abstract on SEP.",
           "doi": "10.5194/egusphere-egu25-9999", "arxiv_id": "", "citation_count": 1,
           "publication_types": ["article"]}  # pubtype would pass is_paper → PLUG A must catch via DOI
    ghost = {"title": "Ghost stub no content", "authors": ["B"], "year": 2023, "venue": "",
             "abstract": "", "doi": "", "arxiv_id": "", "citation_count": 0,
             "publication_types": ["article"]}  # is_paper passes → PLUG B (content floor) must catch
    real = {"title": "Real PINN inversion paper", "authors": ["C"], "year": 2024, "venue": "ApJ",
            "abstract": "A real physics-informed inversion of a transport coefficient.",
            "doi": "10.1/realpaper", "arxiv_id": "", "citation_count": 5,
            "publication_types": ["JournalArticle"]}

    async def fake_external(terms, *, year_min=None, year_max=None, ranking_hint="by_relevance"):
        return [_tag(egu, 0, 0), _tag(ghost, 0, 1), _tag(real, 0, 2)], {}

    async def fake_ingest(cands, *, llm=None):  # gate WOULD pass all 3 — only the plugs stop egu/ghost
        return {i: {"reason": "", "tier": "1A", "ingest_ok": True, "llm_is_paper": True}
                for i in range(len(cands))}, 0

    async def fake_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        return {i: {"reason": "", "score": 0.9} for i in range(len(cands))}, 0

    monkeypatch.setattr("papervault.library.mcp.server.parse_intent", lambda query, llm=None: _fake_plan())
    monkeypatch.setattr("papervault.library.mcp.server.search_external_async", fake_external)
    monkeypatch.setattr("papervault.library.mcp.server.judge_ingest", fake_ingest)
    monkeypatch.setattr("papervault.library.mcp.server.judge_return", fake_return)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "sep transport"})
    assert out["status"] == "ok"
    titles = [p.title for p in populated_lib.all_papers()]
    assert any("Real PINN inversion" in t for t in titles)   # PLUG-clear → ingested
    assert not any("EGU abstract" in t for t in titles)      # PLUG A (egusphere-egu DOI) → rejected
    assert not any("Ghost stub" in t for t in titles)        # PLUG B (no doi/arxiv/abstract) → rejected


@pytest.mark.asyncio
async def test_search_fair_share_no_term_starved(server, populated_lib, monkeypatch):
    """The §4c round-robin gives the niche term-1 candidate a fair-share slot in
    the FIRST cycle, ahead of all but the head of term 0's deep list — so the
    ingest judge sees it near the front, never crowded out at EXT_CAP."""
    seen_titles: list[str] = []

    async def fake_ingest(cands, *, llm=None):
        seen_titles.extend(c["title"] for c in cands)
        return {i: {"reason": "", "tier": "1A", "ingest_ok": True, "llm_is_paper": True}
                for i in range(len(cands))}, 0

    async def fake_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        return {i: {"reason": "", "score": 0.9} for i in range(len(cands))}, 0

    _wire_search(monkeypatch, ingest=fake_ingest, ret=fake_return)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray pinn inversion"})
    assert out["status"] == "ok"
    # The niche paper is seen by the ingest gate (fair-share, not starved)...
    niche = "GCR solar cycle long-horizon forecasting with neural processes"
    assert niche in seen_titles
    # ...and it lands in the SECOND emitted slot (round-robin cycle 1: term-0
    # head, THEN term-1 head), i.e. ahead of term 0's deep filler tail.
    assert seen_titles.index(niche) == 1


@pytest.mark.asyncio
async def test_search_rrf_consensus_floats_up_niche_still_surfaces(
        server, populated_lib, monkeypatch):
    """THE LOCKED RRF ⊕ fair-share COMPOSITION, end-to-end through search_papers:
    a paper surfaced by BOTH terms (cross-term CONSENSUS) but low in term-0's
    native order is floated by RRF to term-0's HEAD (so it reaches the ingest
    gate FIRST), WHILE the round-robin floor still surfaces the niche term-1 paper
    that only one term ranks. Both invariants hold simultaneously."""
    seen_titles: list[str] = []

    async def fake_ingest(cands, *, llm=None):
        seen_titles.extend(c["title"] for c in cands)
        return {i: {"reason": "", "tier": "1A", "ingest_ok": True, "llm_is_paper": True}
                for i in range(len(cands))}, 0

    async def fake_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        return {i: {"reason": "", "score": 0.9} for i in range(len(cands))}, 0

    # A consensus paper that BOTH terms surface — low in term-0 (rank 5), head of
    # term-1 (rank 0). Without RRF it would sit deep in term-0's list; with RRF it
    # floats to term-0's head. A SEPARATE niche paper is alone in term-1 (rank 1).
    consensus = {"title": "Cross-cutting cosmic ray PINN GCR consensus paper",
                 "authors": ["X"], "year": 2024, "venue": "ApJ",
                 "abstract": "consensus", "doi": "10.7/consensus", "arxiv_id": "",
                 "citation_count": 4, "publication_types": ["JournalArticle"]}
    niche = {"title": "Lone niche GCR forecasting neural process method",
             "authors": ["N"], "year": 2024, "venue": "SpaceWeather",
             "abstract": "niche", "doi": "10.7/lone", "arxiv_id": "",
             "citation_count": 2, "publication_types": ["JournalArticle"]}

    def fan_out():
        # term-0: a deep list; consensus paper is buried at rank 5.
        raw = [_tag(c, 0, r) for r, c in enumerate(_term0_cands())]
        raw.append(_tag(consensus, 0, len(raw)))     # consensus deep in term-0
        # term-1: consensus paper is the HEAD (rank 0), niche is rank 1.
        raw.append(_tag(consensus, 1, 0))
        raw.append(_tag(niche, 1, 1))
        return raw

    monkeypatch.setattr("papervault.library.mcp.server.parse_intent",
                        lambda query, llm=None: _fake_plan())

    async def fake_external(terms, *, year_min=None, year_max=None,
                            ranking_hint="by_relevance"):
        return fan_out(), {}
    monkeypatch.setattr("papervault.library.mcp.server.search_external_async", fake_external)
    monkeypatch.setattr("papervault.library.mcp.server.judge_ingest", fake_ingest)
    monkeypatch.setattr("papervault.library.mcp.server.judge_return", fake_return)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray pinn + gcr forecasting"})
    assert out["status"] == "ok"

    cons_title = "Cross-cutting cosmic ray PINN GCR consensus paper"
    niche_title = "Lone niche GCR forecasting neural process method"
    # CONSENSUS: RRF floated it to term-0's head → it is the FIRST candidate the
    # ingest gate sees (ahead of term-0's natural rank-0 head).
    assert seen_titles[0] == cons_title
    # NICHE: not dropped by the consensus score — the round-robin floor surfaces it.
    assert niche_title in seen_titles
    # No internal tag (incl. _rrf) leaks into the output records.
    for r in out["results"]:
        assert not (_INTERNAL_TAGS & set(r)), f"internal tag leaked: {_INTERNAL_TAGS & set(r)}"


@pytest.mark.asyncio
async def test_search_judge_drop_is_not_fail_open(server, populated_lib, monkeypatch):
    """If the ingest judge drops its batch (LLM persistently failed), NOTHING is
    ingested (stop-the-bleeding: never fail-open). If the return judge drops, nothing returns."""
    before = {p.key for p in populated_lib.all_papers()}

    async def empty_ingest(cands, *, llm=None):
        return {}, 1   # dropped the (only) batch

    async def empty_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        return {}, 1

    _wire_search(monkeypatch, ingest=empty_ingest, ret=empty_return)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray pinn inversion"})
    assert out["status"] == "ok"
    assert out["results"] == []  # return judge dropped → nothing returned
    after = {p.key for p in populated_lib.all_papers()}
    assert after == before  # ingest judge dropped → no new papers (no fail-open)
    # §8: the per-gate dropped-batch counts are surfaced in intent_parsed so a
    # silent recall loss is machine-visible.
    assert out["intent_parsed"]["judge_batches_dropped"] == {"ingest": 1, "return": 1}


def _wire_two_equal_score(monkeypatch, *, ranking_hint, fan_out):
    """Wire search_papers with a given ranking_hint and a two-paper fan-out where
    BOTH papers get the SAME return-judge score — so the FINAL order is decided
    purely by the ranking_hint secondary tiebreak (§3 fix)."""
    monkeypatch.setattr("papervault.library.mcp.server.parse_intent",
                        lambda query, llm=None: _fake_plan(ranking_hint=ranking_hint))

    async def fake_external(terms, *, year_min=None, year_max=None,
                            ranking_hint="by_relevance"):
        return fan_out, {}
    monkeypatch.setattr("papervault.library.mcp.server.search_external_async", fake_external)

    async def fake_ingest(cands, *, llm=None):
        return {i: {"reason": "", "tier": "1A", "ingest_ok": True, "llm_is_paper": True}
                for i in range(len(cands))}, 0

    async def fake_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        # EVERY candidate scores identically → ties broken only by the hint.
        return {i: {"reason": "", "score": 0.8} for i in range(len(cands))}, 0

    monkeypatch.setattr("papervault.library.mcp.server.judge_ingest", fake_ingest)
    monkeypatch.setattr("papervault.library.mcp.server.judge_return", fake_return)


# Two in-domain papers, EQUAL return score, distinct year + citation_count.
# Ordered in the fan-out OLD-first / LOW-citation-first so a passing test must
# prove the secondary key REORDERED them (not just preserved input order).
_OLD_LOWCITE = {
    "title": "Older less-cited cosmic ray transport study with a long title",
    "authors": ["A"], "year": 2018, "venue": "JGR",
    "abstract": "cosmic ray transport", "doi": "10.7/old", "arxiv_id": "",
    "citation_count": 5, "publication_types": ["JournalArticle"]}
_NEW_HIGHCITE = {
    "title": "Newer highly-cited cosmic ray transport study with a long title",
    "authors": ["B"], "year": 2024, "venue": "ApJ",
    "abstract": "cosmic ray transport", "doi": "10.7/new", "arxiv_id": "",
    "citation_count": 99, "publication_types": ["JournalArticle"]}


@pytest.mark.asyncio
async def test_search_sort_secondary_by_recency(server, populated_lib, monkeypatch):
    """§3: equal-score papers are ordered by the ranking_hint secondary key.
    by_recency → newer year first, even though the older paper leads the input."""
    fan_out = [_tag(_OLD_LOWCITE, 0, 0), _tag(_NEW_HIGHCITE, 0, 1)]
    _wire_two_equal_score(monkeypatch, ranking_hint="by_recency", fan_out=fan_out)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray transport"})
    years = [r["year"] for r in out["results"] if r["year"] in (2018, 2024)]
    assert years == [2024, 2018]   # newer first (year DESC secondary)


@pytest.mark.asyncio
async def test_search_sort_secondary_by_importance(server, populated_lib, monkeypatch):
    """§3: by_importance → higher citation_count first among equal-score papers."""
    fan_out = [_tag(_OLD_LOWCITE, 0, 0), _tag(_NEW_HIGHCITE, 0, 1)]
    _wire_two_equal_score(monkeypatch, ranking_hint="by_importance", fan_out=fan_out)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray transport"})
    cites = [r["citation_count"] for r in out["results"] if r["citation_count"] in (5, 99)]
    assert cites == [99, 5]   # higher citation_count first


# ── ranking_hint × authority-prior interaction (finding 3) ─────────────────────
# PI decision: an EXPLICIT caller ranking_hint (by_recency / by_importance) is the
# caller's own ranking authority and WINS — the authority prior is SKIPPED entirely
# so the caller's requested secondary order survives. These drive the WHOLE MCP path
# with the prior turned ON. The two papers get EQUAL return score, so IF the prior
# ran, authority (per-year citation rank) would fully decide their order. The data is
# built so authority DISAGREES with the hint — a passing hint assertion therefore
# PROVES the prior was skipped, not merely that it happened to agree.
_OLD_HIGHAUTH = {  # 2010 / 3000 cites → ~187 per-yr: OLDER, far HIGHER authority
    "title": "Older landmark cosmic ray transport study with a long descriptive title",
    "authors": ["A"], "year": 2010, "venue": "JGR",
    "abstract": "cosmic ray transport", "doi": "10.9/oldhigh", "arxiv_id": "",
    "citation_count": 3000, "publication_types": ["JournalArticle"]}
_NEW_LOWAUTH = {  # 2024 / 10 cites → ~5 per-yr: NEWER, far LOWER authority
    "title": "Newer lightly-cited cosmic ray transport study with a long title",
    "authors": ["B"], "year": 2024, "venue": "ApJ",
    "abstract": "cosmic ray transport", "doi": "10.9/newlow", "arxiv_id": "",
    "citation_count": 10, "publication_types": ["JournalArticle"]}
# by_importance disagreement: MORE raw cites but ancient → LOWER per-year, vs FEWER
# raw cites but fresh → HIGHER per-year. Raw-count order (the hint) and per-year
# authority order are opposite, robustly for well over a century of ``now``.
_HIRAW_LOWAUTH = {  # 1980 / 400 cites → ~9 per-yr: MORE raw cites, LOWER authority
    "title": "Ancient heavily-cited cosmic ray transport study with a long title",
    "authors": ["C"], "year": 1980, "venue": "JGR",
    "abstract": "cosmic ray transport", "doi": "10.9/hiraw", "arxiv_id": "",
    "citation_count": 400, "publication_types": ["JournalArticle"]}
_LORAW_HIGHAUTH = {  # 2025 / 300 cites → ~300 per-yr: FEWER raw cites, HIGHER authority
    "title": "Fresh fast-rising cosmic ray transport study with a long title",
    "authors": ["D"], "year": 2025, "venue": "ApJ",
    "abstract": "cosmic ray transport", "doi": "10.9/loraw", "arxiv_id": "",
    "citation_count": 300, "publication_types": ["JournalArticle"]}


@pytest.mark.asyncio
async def test_search_authority_prior_skipped_when_ranking_hint_by_recency(
        server, populated_lib, monkeypatch):
    """Finding 3: prior ON + explicit ranking_hint=by_recency → the prior is SKIPPED,
    the newer paper leads (recency wins) even though it has FAR lower authority. If
    the prior had run it would have surfaced the older high-authority paper instead."""
    monkeypatch.setattr("papervault.library.mcp.server.SEARCH_AUTHORITY_PRIOR", True)
    # OLD (high authority) first in the fan-out → a pass proves the hint REORDERED.
    fan_out = [_tag(_OLD_HIGHAUTH, 0, 0), _tag(_NEW_LOWAUTH, 0, 1)]
    _wire_two_equal_score(monkeypatch, ranking_hint="by_recency", fan_out=fan_out)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray transport"})
    years = [r["year"] for r in out["results"] if r["year"] in (2010, 2024)]
    assert years == [2024, 2010]   # recency wins; prior did NOT reorder to authority


@pytest.mark.asyncio
async def test_search_authority_prior_skipped_when_ranking_hint_by_importance(
        server, populated_lib, monkeypatch):
    """Finding 3: prior ON + explicit ranking_hint=by_importance → the prior is
    SKIPPED, the higher RAW-citation paper leads even though the other paper has
    higher per-year authority. If the prior had run, per-year authority would have
    flipped the order."""
    monkeypatch.setattr("papervault.library.mcp.server.SEARCH_AUTHORITY_PRIOR", True)
    # LOW-raw (but high authority) first → a pass proves the hint REORDERED to raw DESC.
    fan_out = [_tag(_LORAW_HIGHAUTH, 0, 0), _tag(_HIRAW_LOWAUTH, 0, 1)]
    _wire_two_equal_score(monkeypatch, ranking_hint="by_importance", fan_out=fan_out)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray transport"})
    cites = [r["citation_count"] for r in out["results"] if r["citation_count"] in (300, 400)]
    assert cites == [400, 300]   # raw-citation order wins; prior did NOT reorder


@pytest.mark.asyncio
async def test_search_authority_prior_reorders_on_default_hint(
        server, populated_lib, monkeypatch):
    """Positive control: prior ON + the DEFAULT by_relevance hint → the prior DOES
    run through the MCP path, lifting the equal-score high-authority paper above the
    low-authority one that led the fan-out. Proves the skip is specific to explicit
    hints, not the flag being inert."""
    monkeypatch.setattr("papervault.library.mcp.server.SEARCH_AUTHORITY_PRIOR", True)
    # LOW authority first → only the prior can move HIGH authority to the top.
    fan_out = [_tag(_NEW_LOWAUTH, 0, 0), _tag(_OLD_HIGHAUTH, 0, 1)]
    _wire_two_equal_score(monkeypatch, ranking_hint="by_relevance", fan_out=fan_out)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray transport"})
    years = [r["year"] for r in out["results"] if r["year"] in (2010, 2024)]
    assert years == [2010, 2024]   # high-authority 2010 paper lifted above the 2024 one


@pytest.mark.asyncio
async def test_search_judge_batches_dropped_zero_when_healthy(server, populated_lib,
                                                              monkeypatch):
    """§8: a healthy run (no dropped batches) surfaces judge_batches_dropped=0 for
    both gates — the field is always present, not just on failure."""
    async def fake_ingest(cands, *, llm=None):
        return {i: {"reason": "", "tier": "1A", "ingest_ok": True, "llm_is_paper": True}
                for i in range(len(cands))}, 0

    async def fake_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        return {i: {"reason": "", "score": 0.9} for i in range(len(cands))}, 0

    _wire_search(monkeypatch, ingest=fake_ingest, ret=fake_return)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray pinn inversion"})
    assert out["intent_parsed"]["judge_batches_dropped"] == {"ingest": 0, "return": 0}


@pytest.mark.asyncio
async def test_search_ingest_upsert_passes_no_llm(server, populated_lib, monkeypatch):
    """The dead borderline-merge LLM path is gone: search ingest calls
    library.upsert WITHOUT an ``llm`` (so no blocking sync llm.call ever runs on
    the event loop under lib_write_lock during ingest)."""
    seen_kwargs: list[dict] = []
    real_upsert = populated_lib.upsert

    def spy_upsert(paper_data, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return real_upsert(paper_data, **kwargs)

    monkeypatch.setattr(populated_lib, "upsert", spy_upsert)

    async def fake_ingest(cands, *, llm=None):
        return {i: {"reason": "", "tier": "1A", "ingest_ok": True, "llm_is_paper": True}
                for i in range(len(cands))}, 0

    async def fake_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        return {i: {"reason": "", "score": 0.9} for i in range(len(cands))}, 0

    _wire_search(monkeypatch, ingest=fake_ingest, ret=fake_return)
    server._paper_download_queue.add = lambda key, **kw: None

    out = await _call(server, "search_papers", {"query": "cosmic ray pinn inversion"})
    assert out["status"] == "ok"
    assert seen_kwargs, "ingest path should have upserted at least one candidate"
    # No call site passed an llm — the borderline-LLM branch is fully dead.
    assert all("llm" not in kw for kw in seen_kwargs)


# ============== resources ==================================================


@pytest.mark.asyncio
async def test_all_resource_descriptions_are_nonempty(server):
    """Boundary fix #5: ALL 4 @mcp.resource handlers must serve a real, non-empty
    description (the drill found every one empty). Covers the concrete bib + the
    3 templated resources, and pins their load-bearing prose."""
    concrete = await server.list_resources()
    templates = await server.list_resource_templates()

    by_uri = {str(r.uri): (r.description or "") for r in concrete}
    by_tmpl = {t.uriTemplate: (t.description or "") for t in templates}

    assert by_uri.get("library://bib", "").strip(), "library://bib description empty"
    for tmpl in ("library://paper/{key}", "library://extract/{key}.md",
                 "library://extract/{key}.txt"):
        assert by_tmpl.get(tmpl, "").strip(), f"{tmpl} description empty"

    # Load-bearing content: the bib warns it's a bulk dump; paper documents the
    # not_found shape; the extracts carry the serve-safety / absence prose.
    assert "bulk" in by_uri["library://bib"].lower() or "8mb" in by_uri["library://bib"].lower()
    assert "not_found" in by_tmpl["library://paper/{key}"]
    for tmpl in ("library://extract/{key}.md", "library://extract/{key}.txt"):
        d = by_tmpl[tmpl]
        assert "not_found" in d           # missing-key shape
        assert "serve-safe" in d or "serve-safety" in d  # the moved prose


def test_search_papers_docstring_documents_intent_parsed_observability():
    """Boundary fix #6: the search_papers docstring Returns section must document
    the full intent_parsed payload — the observability signals a caller needs to
    detect silent recall loss / backend outages, not just 'search terms'."""
    from papervault.library.mcp.server import build_server
    from papervault.library import Library
    import tempfile
    srv = build_server(library=Library(tempfile.mkdtemp()))
    # FastMCP stores the tool's description (derived from the docstring).
    tool = srv._tool_manager.get_tool("search_papers")
    doc = tool.description or ""
    for field in ("sources_degraded", "sources_unconfigured",
                  "judge_batches_dropped", "reasoning"):
        assert field in doc, f"intent_parsed.{field} not documented in search_papers docstring"
    assert "recall" in doc.lower()  # the 'silently lost recall — retry' signal


@pytest.mark.asyncio
async def test_resource_bib(server):
    out = await _read(server, "library://bib")
    assert "@article{Potgieter2013" in out
    assert "@article{Wei2024" in out


@pytest.mark.asyncio
async def test_resource_paper_known(server):
    raw = await _read(server, "library://paper/Wei2024")
    payload = json.loads(raw)
    assert payload["key"] == "Wei2024"
    assert payload["title"] == "Cosmic ray PINN"


@pytest.mark.asyncio
async def test_resource_paper_unknown(server):
    raw = await _read(server, "library://paper/NoSuchKey9999")
    payload = json.loads(raw)
    assert payload["error"] == "not_found"


@pytest.mark.asyncio
async def test_resource_extract_md(server):
    out = await _read(server, "library://extract/Wei2024.md")
    assert "markdown body" in out


@pytest.mark.asyncio
async def test_resource_extract_txt(populated_lib, server):
    """The ``.txt`` resource serves a txt as full text ONLY for a no-pdf
    migration row (¬has_pdf ∧ ¬has_md), 2026-06-06 txt-drop (SDD §3.4). Wei2024
    HAS a pdf+md, so its leftover txt is NOT served via this resource — a no-pdf
    row (Potgieter2013) with a real txt is the served case."""
    # Wei2024 has a pdf+md → its txt is NOT served (narrowed serve door).
    assert await _read(server, "library://extract/Wei2024.txt") == ""
    # A no-pdf row with a real (≥floor) txt under a non-terminal status IS served.
    p = populated_lib.get("Potgieter2013")        # no pdf / no md on disk
    populated_lib.txt_path("Potgieter2013").write_text("no-pdf migration body " * 60)
    p.download_status = "ok"
    populated_lib.save()
    out = await _read(server, "library://extract/Potgieter2013.txt")
    assert "no-pdf migration body" in out


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["md", "txt"])
async def test_resource_extract_missing_key_is_structured_not_found(server, fmt):
    """fix #14 (SDD §5 I-ABSENCE): a MISSING key returns a structured not_found
    (like library://paper), distinct from the fail-closed '' a held-but-unservable
    paper returns. The caller can tell 'wrong key' from 'no full text'."""
    raw = await _read(server, f"library://extract/__no_such_key_zzz__.{fmt}")
    payload = json.loads(raw)
    assert payload["error"] == "not_found"
    assert payload["key"] == "__no_such_key_zzz__"


@pytest.mark.asyncio
async def test_resource_extract_held_but_unservable_stays_empty_string(populated_lib, server):
    """fix #14 guard: a paper that EXISTS but has no servable text keeps the
    fail-closed '' (not a structured not_found) — the two cases stay distinct."""
    p = populated_lib.get("Potgieter2013")  # in library, no extract on disk
    p.download_status = "metadata_only"
    populated_lib.save()
    out = await _read(server, "library://extract/Potgieter2013.txt")
    assert out == ""


# ---- F3: the extract resources must obey serve-safety (no raw bypass) -------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["extract_failed", "failed", "metadata_only"])
async def test_resource_extract_txt_terminal_status_is_not_served(
        populated_lib, server, status):
    """F3: a TERMINAL paper's leftover fat pypdf txt (the SAME PDF the gate
    judged incomplete) must NOT be served raw by the library://extract/{key}.txt
    resource — it is fail-closed exactly like the get_paper/search chokepoint.
    Before the fix the resource did a bare path.read_text() with no guards."""
    p = populated_lib.get("Potgieter2013")  # no md on disk
    populated_lib.txt_path("Potgieter2013").write_text("PAYWALL FRAGMENT " * 90)
    assert populated_lib.txt_path("Potgieter2013").stat().st_size >= 500
    p.download_status = status
    populated_lib.save()

    out = await _read(server, "library://extract/Potgieter2013.txt")
    assert out == "", "terminal leftover txt served raw — serve-safety bypassed"


@pytest.mark.asyncio
async def test_resource_extract_txt_sub_floor_is_not_served(
        populated_lib, server):
    """F3: a near-empty (scanned-PDF noise) txt under the byte floor must NOT be
    served raw by the resource — it would impersonate full text (D6)."""
    p = populated_lib.get("Potgieter2013")
    populated_lib.txt_path("Potgieter2013").write_text("\f \n")  # scan noise
    p.download_status = "pending"
    populated_lib.save()

    out = await _read(server, "library://extract/Potgieter2013.txt")
    assert out == ""


@pytest.mark.asyncio
async def test_resource_extract_txt_pending_real_txt_is_served(
        populated_lib, server):
    """F3 guard: a real (≥floor) txt under a NON-terminal status is still served
    — the fix fail-closes terminal/thin txt, not legitimate in-progress txt."""
    p = populated_lib.get("Potgieter2013")
    populated_lib.txt_path("Potgieter2013").write_text("real extracted body " * 60)
    p.download_status = "pending"
    populated_lib.save()

    out = await _read(server, "library://extract/Potgieter2013.txt")
    assert "real extracted body" in out


# ---------------- §8 source-health derivation (Stage E) ---------------------


def test_derive_source_health_all_degraded(monkeypatch):
    """A configured backend whose EVERY (term,backend) pair degraded (== T)
    appears in sources_degraded; a partially-degraded one does not."""
    from papervault.library.mcp import server as srv
    monkeypatch.setenv("ADS_API_TOKEN", "tok")
    monkeypatch.setenv("CORE_API_KEY", "k")
    # T = 3 terms; s2 fully degraded (3/3), arxiv partially (1/3), core not at all.
    degraded = {"semantic_scholar": 3, "arxiv": 1}
    unconfigured, degraded_list = srv._derive_source_health(degraded, 3)
    assert "semantic_scholar" in degraded_list
    assert "arxiv" not in degraded_list       # 1 < 3 → not fully degraded
    assert unconfigured == []                  # both keyed backends configured


def test_derive_source_health_unconfigured_disjoint(monkeypatch):
    """A backend failing UNCONFIGURED_CHECK lands in sources_unconfigured and is
    EXCLUDED from sources_degraded (the two lists are disjoint by construction),
    even if its degraded count == T."""
    from papervault.library.mcp import server as srv
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    monkeypatch.setenv("CORE_API_KEY", "k")
    degraded = {"ads": 2, "core": 2}
    unconfigured, degraded_list = srv._derive_source_health(degraded, 2)
    assert "ads" in unconfigured               # no token → unconfigured
    assert "ads" not in degraded_list          # disjoint: unconfigured wins
    assert "core" in degraded_list             # configured + fully degraded


def test_derive_source_health_blank_core_key_is_unconfigured(monkeypatch):
    """The CORE check's .strip() is load-bearing: a whitespace-only key counts
    as unconfigured."""
    from papervault.library.mcp import server as srv
    monkeypatch.setenv("ADS_API_TOKEN", "tok")
    monkeypatch.setenv("CORE_API_KEY", "   ")
    unconfigured, _ = srv._derive_source_health({}, 1)
    assert "core" in unconfigured
    assert "ads" not in unconfigured


def test_derive_source_health_no_terms_no_degrade():
    """T == 0 → nothing is degraded (the == T test requires T > 0)."""
    from papervault.library.mcp import server as srv
    unconfigured, degraded_list = srv._derive_source_health({"arxiv": 0}, 0)
    assert degraded_list == []
