"""Durable conflict evidence, bounded dispatch, and conservative adjudication."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from papervault.library import Library
from papervault.library.models import IdentityRecovery


def card(n=0, **extra):
    return {"title": f"Transport of cosmic rays in heliospheric region {n}",
            "authors": [f"Scientist{n}"], "year": 2020, "doi": f"10.1234/paper{n}",
            "source": "ads", "url": f"https://example.org/{n}", **extra}


def conflict(lib, n):
    paper, _ = lib.upsert(card(n))
    lib.upsert(card(n, title=f"Revised transport of cosmic rays in region {n}", source="crossref"))
    return paper


def test_merge_stages_both_sides_and_provenance_before_fill_blanks(tmp_path):
    from papervault.library.services.correction_ledger import CorrectionLedger
    lib = Library(tmp_path)
    paper = conflict(lib, 1)
    state = CorrectionLedger(tmp_path).read()
    record = next(iter(state["conflicts"].values()))
    assert paper.title == "Transport of cosmic rays in heliospheric region 1"
    assert record["fields"]["title"] == {
        "existing": paper.title, "incoming": "Revised transport of cosmic rays in region 1"}
    assert record["existing"]["source"] == "ads"
    assert record["incoming"]["source"] == "crossref"
    assert record["provenance"]["operation"] == "merge"
    assert record["existing"]["doi"] == record["incoming"]["doi"] == "10.1234/paper1"
    assert record["status"] == "pending"
    paper.download_status = "metadata_only"
    conflict(lib, 1)
    assert len(CorrectionLedger(tmp_path).read()["conflicts"]) == 1


def test_ten_subject_papers_emit_one_immutable_luna_manifest(tmp_path):
    from papervault.library.services.correction_ledger import CorrectionLedger
    lib = Library(tmp_path)
    for n in range(9):
        conflict(lib, n)
    ledger = CorrectionLedger(tmp_path)
    assert ledger.read()["manifests"] == {}
    lib.upsert(card(0, venue="First venue"))
    lib.upsert(card(0, venue="Other venue"))
    assert ledger.read()["manifests"] == {}  # fields are not papers
    conflict(lib, 9)
    state = ledger.read()
    assert len(state["manifests"]) == 1
    manifest = next(iter(state["manifests"].values()))
    assert manifest["executor"] == {"carrier": "codex-cli", "model": "gpt-6-luna", "effort": "max"}
    assert len(manifest["paper_keys"]) == 10
    assert 10 <= len(manifest["conflicts"]) <= 50
    assert manifest["default_disposition"] == "needs_human"
    assert manifest["dispatch_owner"] == "orchestrator"
    before = json.dumps(manifest, sort_keys=True)
    conflict(lib, 9)
    assert json.dumps(ledger.read()["manifests"][manifest["id"]], sort_keys=True) == before


def test_unadjudicable_and_omitted_dispositions_are_retained(tmp_path):
    from papervault.library.services.correction_ledger import CorrectionLedger
    lib = Library(tmp_path)
    for n in range(10):
        conflict(lib, n)
    ledger = CorrectionLedger(tmp_path)
    manifest = next(iter(ledger.read()["manifests"].values()))
    first = manifest["conflicts"][0]["id"]
    result = {"manifest_id": manifest["id"], "results": [
        {"conflict_id": first, "status": "resolved", "evidence": "DOI registry corroborates stored title",
         "decision": "retain_existing"}]}
    ledger.adjudicate(result)
    state = ledger.read()
    assert state["conflicts"][first]["status"] == "resolved"
    rest = [c for c in state["conflicts"].values() if c["id"] != first]
    assert all(c["status"] == "needs_human" and c["existing"] and c["incoming"] for c in rest)
    assert ledger.summary()["needs_human"] == 9
    snapshot = ledger.path.read_bytes()
    ledger.adjudicate(result)
    assert ledger.path.read_bytes() == snapshot
    assert lib.get(manifest["paper_keys"][0]).title.startswith("Transport")


@pytest.mark.parametrize("kind", ["foreign", "duplicate", "no_evidence", "unknown_status"])
def test_invalid_adjudication_is_atomic(tmp_path, kind):
    from papervault.library.services.correction_ledger import CorrectionLedger
    lib = Library(tmp_path)
    for n in range(10):
        conflict(lib, n)
    ledger = CorrectionLedger(tmp_path)
    manifest = next(iter(ledger.read()["manifests"].values()))
    item = {"conflict_id": manifest["conflicts"][0]["id"], "status": "resolved",
            "decision": "retain_existing", "evidence": "registry evidence"}
    items = [item]
    if kind == "foreign":
        item["conflict_id"] = "foreign"
    elif kind == "duplicate":
        items.append(item.copy())
    elif kind == "no_evidence":
        item["evidence"] = ""
    else:
        item["status"] = "discard"
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        ledger.adjudicate({"manifest_id": manifest["id"], "results": items})
    assert ledger.path.read_bytes() == before


def test_recovery_collision_keeps_both_papers_and_repeated_observations_deduplicate(tmp_path):
    from papervault.library.services.correction_ledger import CorrectionLedger
    lib = Library(tmp_path)
    stub, _ = lib.upsert(card(0, doi="", download_status="failed"))
    holder, _ = lib.upsert(card(1, doi="10.1234/collision"))
    recovery = IdentityRecovery(doi=holder.doi, route="ads_api", source_url=stub.url,
                                verified_title=stub.title, observed_at="2026-10-01T00:00:00+00:00")
    assert lib.set_recovered_identity(stub.key, recovery) == "collision"
    record = next(iter(CorrectionLedger(tmp_path).read()["conflicts"].values()))
    assert record["key"] == stub.key
    assert record["existing"]["key"] == holder.key
    assert record["incoming"]["key"] == stub.key
    assert record["incoming"]["doi"] == ""  # the original side, candidate is separate evidence
    assert record["provenance"]["recovery"]["route"] == "ads_api"
    assert stub.doi == "" and stub.download_status == "failed"
    assert lib.get(holder.key) is holder
    assert lib.set_recovered_identity(stub.key, recovery.model_copy(update={
        "observed_at": "2026-11-01T00:00:00+00:00"})) == "collision"
    assert len(CorrectionLedger(tmp_path).read()["conflicts"]) == 1
    stub.title = "A substantively revised scientific bibliography for this paper"
    assert lib.set_recovered_identity(stub.key, recovery) == "collision"
    assert len(CorrectionLedger(tmp_path).read()["conflicts"]) == 2


def test_enrichment_conflicts_are_not_discarded(tmp_path):
    from papervault.library.services.correction_ledger import CorrectionLedger
    lib = Library(tmp_path)
    paper, _ = lib.upsert(card(0, venue="Original journal"))
    assert not lib.enrich(paper.key, {"venue": "Other journal"}, enriched_at="2026-10-01")
    record = next(iter(CorrectionLedger(tmp_path).read()["conflicts"].values()))
    assert record["fields"]["venue"] == {"existing": "Original journal", "incoming": "Other journal"}
    assert paper.venue == "Original journal"


def test_concurrent_staging_preserves_every_conflict(tmp_path):
    from papervault.library.services.correction_ledger import CorrectionLedger
    def stage(n):
        return CorrectionLedger(tmp_path).stage(
            key=f"Paper{n}", fields={"title": {"existing": "old", "incoming": "new"}},
            existing={"title": "old", "source": "ads"}, incoming={"title": "new", "source": "crossref"},
            provenance={"operation": "merge"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(stage, range(20)))
    state = CorrectionLedger(tmp_path).read()
    assert len(state["conflicts"]) == 20
    assert len(state["manifests"]) == 2


def test_corrupt_ledger_fails_closed(tmp_path):
    from papervault.library.services.correction_ledger import CorrectionLedger
    ledger = CorrectionLedger(tmp_path)
    ledger.path.write_text('{"conflicts":')
    with pytest.raises(ValueError):
        ledger.read()
    assert ledger.path.read_text() == '{"conflicts":'


def test_recovery_rejected_after_existing_identifier_retains_verification_provenance(tmp_path):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(card(0))
    recovery = IdentityRecovery(doi="10.1234/other", route="url_pattern", source_url=paper.url,
        verified_title=paper.title, observed_at="2026-10-01T00:00:00+00:00")
    assert lib.set_recovered_identity(paper.key, recovery) == "has_identifier"
    record = next(iter(lib.corrections.read()["conflicts"].values()))
    assert record["provenance"]["recovery"] == recovery.model_dump()
    assert record["fields"]["doi"] == {"existing": "10.1234/paper0", "incoming": "10.1234/other"}


@pytest.mark.parametrize("source", ["core", "openalex"])
def test_verified_download_claim_conflicts_keep_known_source_provenance(tmp_path, source):
    from papervault.library.download_sources import core, openalex
    lib = Library(tmp_path)
    paper, _ = lib.upsert(card(0))
    if source == "core":
        core._commit_core_identifiers(core._CorePDF(b"pdf-fixture", "10.1234/other", ""), paper, lib)
    else:
        openalex._apply_openalex_identifiers(
            openalex._OpenAlexPDF(b"pdf-fixture", "10.1234/other", ""), paper, lib, "llm_match: same work")
    record = next(iter(lib.corrections.read()["conflicts"].values()))
    assert record["provenance"]["incoming_source"] == source
    assert record["provenance"]["verification"]
    assert paper.doi == "10.1234/paper0"


@pytest.mark.asyncio
async def test_enrich_sweep_retains_actual_crossref_lookup_provenance(tmp_path, monkeypatch):
    from papervault.library import fetch
    from papervault.library.services import reconcile
    lib = Library(tmp_path)
    paper, _ = lib.upsert(card(0, venue="Original journal"))
    monkeypatch.setattr(fetch, "crossref_enrich_lookup", lambda doi: ("ok", {"venue": "Other journal"}))
    await reconcile._enrich_sweep(lib, cap=1)
    record = next(iter(lib.corrections.read()["conflicts"].values()))
    assert record["provenance"]["incoming_source"] == "crossref"
    assert record["provenance"]["lookup_doi"] == "10.1234/paper0"
    assert record["provenance"]["checked_at"]


def test_distinct_verified_provider_evidence_is_not_deduplicated_away(tmp_path):
    from papervault.library.download_sources import core, openalex
    lib = Library(tmp_path)
    paper, _ = lib.upsert(card(0))
    core._commit_core_identifiers(core._CorePDF(b"fixture", "10.1234/other", ""), paper, lib)
    openalex._apply_openalex_identifiers(openalex._OpenAlexPDF(b"fixture", "10.1234/other", ""),
                                        paper, lib, "llm_match: same work")
    records = list(lib.corrections.read()["conflicts"].values())
    assert len(records) == 2
    assert {r["provenance"]["incoming_source"] for r in records} == {"core", "openalex"}
