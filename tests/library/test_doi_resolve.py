"""Tests for the by-title DOI resolver (root-cause fix for no-DOI in-domain
stubs). Covers:

  * fetch.resolve_doi_by_title — the SAFE-MATCH predicate: accept an exact
    match, REJECT the wrong-paper top-hit (the live-probe failure mode),
    abstain on weak/ambiguous/short-title/year-mismatch.
  * Library.set_resolved_doi — re-indexes _by_doi, never overwrites a real
    DOI, routes a DOI collision to merge+purge (no stale index).
  * cli `audit --resolve-stub-dois` — --dry-run writes nothing (default),
    --no-dry-run writes via the store method; uses a TEMP library, never the
    real vault.

All HTTP is mocked via the `responses` lib. No live network, no live vault.
"""

from __future__ import annotations

import json

import pytest
import responses

from papervault.library import Library, cli, fetch


_CROSSREF_WORKS = "https://api.crossref.org/works"


def _crossref_items(items: list[dict]) -> dict:
    """Wrap a list of raw Crossref item dicts in the message envelope."""
    return {"message": {"total-results": len(items), "items": items}}


def _item(doi: str, title: str, family: str, year, *, given: str = "A.") -> dict:
    return {
        "DOI": doi,
        "title": [title],
        "author": [{"given": given, "family": family}],
        "issued": {"date-parts": [[year]]},
    }


# ============================ resolver: ACCEPT ==============================


@responses.activate
def test_resolve_exact_match_accepted():
    """Near-exact title + first-author surname + year within ±1 -> accept."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            _item("10.1234/correct",
                  "The transport of cosmic rays in the heliosheath",
                  "Strauss", 2012),
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["R. D. Strauss"], 2012)
    assert doi == "10.1234/correct"


@responses.activate
def test_resolve_year_off_by_one_accepted():
    """Year within ±1 is allowed (preprint↔journal lag)."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            _item("10.1234/correct",
                  "The transport of cosmic rays in the heliosheath",
                  "Strauss", 2013),
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["Strauss"], 2012)
    assert doi == "10.1234/correct"


@responses.activate
def test_resolve_surname_first_comma_author_accepted():
    """Regression: the stub stores authors surname-first, comma-separated
    ("Chiappetta, Federica"), while Crossref returns "given family". The
    surname extraction must read "Chiappetta" (before the comma), NOT the
    given-name tail "Federica", or the author gate false-rejects a true match.
    This was the root cause of the 0/218 batch yield."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            {"DOI": "10.3847/1538-4357/ae230e",
             "title": ["Evolution of the Shock Properties of the 2023 March 13 Event"],
             "author": [{"given": "Federica", "family": "Chiappetta"},
                        {"given": "Giuseppe", "family": "Nisticò"}],
             "issued": {"date-parts": [[2025]]}},
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "Evolution of the Shock Properties of the 2023 March 13 Event",
        ["Chiappetta, Federica", "Nisticò, Giuseppe"], 2025)
    assert doi == "10.3847/1538-4357/ae230e"


# ============================ resolver: REJECT =============================


@responses.activate
def test_resolve_wrong_top_hit_rejected():
    """The live-probe failure mode: Crossref's TOP-scored hit is the WRONG
    paper (different title, different first author). Score is IGNORED; the
    predicate rejects it -> abstain."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            # The real top hit from the live probe — wrong paper.
            {"DOI": "10.1142/9789814329033_0055",
             "title": ["Galactic Cosmic Rays in the Dynamic Heliosphere"],
             "author": [{"given": "M.", "family": "Potgieter"}],
             "issued": {"date-parts": [[2011]]},
             "score": 32.9},
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["Strauss"], 2012)
    assert doi is None  # wrong title + wrong author -> abstain


@responses.activate
def test_resolve_author_mismatch_rejected():
    """Title matches but the first-author surname is absent -> abstain."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            _item("10.1234/x",
                  "The transport of cosmic rays in the heliosheath",
                  "Wronglastname", 2012),
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["Strauss"], 2012)
    assert doi is None


@responses.activate
def test_resolve_year_mismatch_rejected():
    """Title + author match but year off by >1 -> abstain."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            _item("10.1234/x",
                  "The transport of cosmic rays in the heliosheath",
                  "Strauss", 2008),
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["Strauss"], 2012)
    assert doi is None


@responses.activate
def test_resolve_ambiguous_two_matches_abstains():
    """Two DISTINCT DOIs both pass the predicate -> ambiguous -> abstain."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            _item("10.1234/a",
                  "The transport of cosmic rays in the heliosheath",
                  "Strauss", 2012),
            _item("10.1234/b",
                  "The transport of cosmic rays in the heliosheath",
                  "Strauss", 2012),
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["Strauss"], 2012)
    assert doi is None


@responses.activate
def test_resolve_same_doi_twice_is_not_ambiguous():
    """The SAME DOI returned twice is NOT ambiguity -> accept."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            _item("10.1234/correct",
                  "The transport of cosmic rays in the heliosheath",
                  "Strauss", 2012),
            _item("10.1234/CORRECT",  # case-variant of the same DOI
                  "The transport of cosmic rays in the heliosheath",
                  "Strauss", 2012),
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["Strauss"], 2012)
    assert doi == "10.1234/correct"


def test_resolve_short_title_abstains_without_http():
    """A too-short / degenerate title is rejected BEFORE any HTTP call."""
    # No responses registered — if it tried to call out, responses would raise.
    with responses.RequestsMock():
        doi = fetch.resolve_doi_by_title("Cosmic rays", ["Strauss"], 2012)
    assert doi is None


@responses.activate
def test_resolve_no_year_requires_stricter_title():
    """A no-year stub must clear the stricter ratio (0.97). A 0.92-ish match
    that would pass WITH a year is rejected WITHOUT one."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            # Slightly different wording — passes the 0.92 bar, fails 0.97.
            _item("10.1234/x",
                  "The transport of the cosmic rays in heliosheath region",
                  "Strauss", 2012),
        ]),
        status=200,
    )
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["Strauss"], None)
    assert doi is None


@responses.activate
def test_resolve_transient_returns_none():
    """A 500 from Crossref -> transient -> abstain (None)."""
    responses.add(responses.GET, _CROSSREF_WORKS, status=500)
    doi = fetch.resolve_doi_by_title(
        "The transport of cosmic rays in the heliosheath",
        ["Strauss"], 2012)
    assert doi is None


# ====================== store.set_resolved_doi =============================


@pytest.fixture
def stub_lib(tmp_path):
    """Temp library with one no-DOI stub. NEVER the real vault."""
    lib = Library(tmp_path)
    lib.upsert({
        "title": "The transport of cosmic rays in the heliosheath",
        "authors": ["R. D. Strauss"], "year": 2012,
        "abstract": "We model cosmic-ray transport in the heliosheath.",
        "source": "ads",
    })
    lib.save()
    return lib


def test_set_resolved_doi_sets_and_reindexes(stub_lib):
    key = stub_lib.keys()[0]
    outcome, detail = stub_lib.set_resolved_doi(key, "10.1234/Correct")
    assert outcome == "set"
    p = stub_lib.get(key)
    assert p.doi == "10.1234/Correct"
    # _by_doi index updated (lowercased) so find() resolves it.
    assert stub_lib.find(doi="10.1234/correct") is not None
    assert stub_lib.find(doi="10.1234/correct").key == key
    assert p.resolved_doi_at is not None
    assert "doi_resolved" in p.source


def test_set_resolved_doi_never_overwrites_existing(tmp_path):
    lib = Library(tmp_path)
    lib.upsert({"title": "A paper that already has a DOI here",
                "authors": ["Smith"], "year": 2020, "doi": "10.9999/original",
                "abstract": "x"})
    lib.save()
    key = lib.keys()[0]
    outcome, detail = lib.set_resolved_doi(key, "10.1234/different")
    assert outcome == "has_doi"
    assert detail == "10.9999/original"
    assert lib.get(key).doi == "10.9999/original"  # untouched
    # The would-be new DOI is NOT in the index.
    assert lib.find(doi="10.1234/different") is None


def test_set_resolved_doi_collision_merges_and_purges(tmp_path):
    """If the resolved DOI already belongs to ANOTHER row, the stub is merged
    into the holder + purged — never a duplicate DOI in two rows."""
    lib = Library(tmp_path)
    lib.upsert({"title": "Cosmic ray transport in the outer heliosphere",
                "authors": ["Strauss"], "year": 2012, "doi": "10.1234/twin",
                "abstract": "holder abstract"})
    lib.upsert({"title": "The transport of cosmic rays in the heliosheath",
                "authors": ["R. D. Strauss"], "year": 2012,
                "abstract": "stub abstract", "venue": "JGR"})
    lib.save()
    holder_key = lib.find(doi="10.1234/twin").key
    stub_key = next(k for k in lib.keys() if k != holder_key)

    outcome, detail = lib.set_resolved_doi(stub_key, "10.1234/twin")
    assert outcome == "collision"
    assert detail == holder_key
    # Stub purged; DOI still maps to exactly the holder.
    assert lib.get(stub_key) is None
    assert lib.find(doi="10.1234/twin").key == holder_key
    # Holder absorbed the stub's blank-fill field (venue).
    assert lib.get(holder_key).venue == "JGR"


def test_set_resolved_doi_unknown_key(tmp_path):
    lib = Library(tmp_path)
    outcome, _ = lib.set_resolved_doi("NoSuchKey2020", "10.1/a")
    assert outcome == "no_paper"


# ====================== cli audit --resolve-stub-dois ======================


@pytest.fixture
def cli_stub_env(tmp_path, monkeypatch):
    """$PAPER_LIBRARY_PATH -> tmpdir with one no-DOI stub + one normal paper."""
    monkeypatch.setenv("PAPER_LIBRARY_PATH", str(tmp_path))
    lib = Library(tmp_path)
    lib.upsert({
        "title": "The transport of cosmic rays in the heliosheath",
        "authors": ["R. D. Strauss"], "year": 2012,
        "abstract": "We model cosmic-ray transport in the heliosheath.",
        "source": "ads",
    })
    # A normal paper that ALREADY has a DOI — must be ignored by the stub scan.
    lib.upsert({"title": "Some already-resolved paper title here",
                "authors": ["Smith"], "year": 2020, "doi": "10.9999/already",
                "abstract": "y"})
    lib.save()
    return tmp_path


def _mock_correct_crossref():
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            _item("10.1234/correct",
                  "The transport of cosmic rays in the heliosheath",
                  "Strauss", 2012),
        ]),
        status=200,
    )


@responses.activate
def test_cli_resolve_dry_run_writes_nothing(cli_stub_env, capsys):
    _mock_correct_crossref()
    rc = cli.main(["audit", "--resolve-stub-dois", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)["resolve_stub_dois"]
    assert payload["dry_run"] is True
    assert payload["scanned"] == 1  # only the no-DOI stub, not the DOI paper
    assert payload["resolved"] == 1
    assert payload["matches"][0]["doi"] == "10.1234/correct"
    # NOTHING written: reload from disk and confirm the stub still has no DOI.
    reloaded = Library(cli_stub_env)
    stub = next(p for p in reloaded.all_papers() if not p.doi)
    assert stub.doi == ""
    assert reloaded.find(doi="10.1234/correct") is None


@responses.activate
def test_cli_resolve_no_dry_run_writes(cli_stub_env, capsys):
    _mock_correct_crossref()
    rc = cli.main(["audit", "--resolve-stub-dois", "--no-dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)["resolve_stub_dois"]
    assert payload["dry_run"] is False
    assert payload["resolved"] == 1
    assert payload["outcomes"]["set"] == 1
    # Persisted: reload from disk and confirm the DOI + index landed.
    reloaded = Library(cli_stub_env)
    assert reloaded.find(doi="10.1234/correct") is not None
    resolved = reloaded.find(doi="10.1234/correct")
    assert resolved.resolved_doi_at is not None


@responses.activate
def test_cli_resolve_wrong_hit_abstains_dry_run(cli_stub_env, capsys):
    """The wrong-paper top hit is abstained even in a write run -> no DOI set."""
    responses.add(
        responses.GET, _CROSSREF_WORKS,
        json=_crossref_items([
            {"DOI": "10.1142/wrong",
             "title": ["Galactic Cosmic Rays in the Dynamic Heliosphere"],
             "author": [{"given": "M.", "family": "Potgieter"}],
             "issued": {"date-parts": [[2011]]}, "score": 99.0},
        ]),
        status=200,
    )
    rc = cli.main(["audit", "--resolve-stub-dois", "--no-dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)["resolve_stub_dois"]
    assert payload["resolved"] == 0
    assert payload["abstained"] == 1
    reloaded = Library(cli_stub_env)
    assert reloaded.find(doi="10.1142/wrong") is None
