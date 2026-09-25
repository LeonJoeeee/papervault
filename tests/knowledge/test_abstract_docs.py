"""Abstract-only docs (#144): a metadata-state paper with an abstract enters the graph as ONE
abstract-only doc, marked as such in its text, fingerprint, ledger status, synth label and the
query out-feed; full text replaces it; one CLI command rolls the whole class back.

Offline: fake ledger + fake LightRAG (the round-test fakes, with stored doc content), no DB.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from lightrag.base import DocStatus

import papervault.knowledge.cli as cli_mod
import papervault.knowledge.ingest.distill as distill
import papervault.knowledge.scheduler.round as rnd
from papervault.knowledge.ingest import abstract_doc as ad
from papervault.knowledge.ingest.fingerprint import META, fingerprint
from papervault.knowledge.ingest.paper_library_client import PaperRecord
from papervault.knowledge.ledger.store import VALID_STATUS, LedgerRecord, _next_attempts_status
from papervault.knowledge.query.aquery import _EMPTY, _abstract_only_papers
from papervault.knowledge.query.synth import _SYNTH_SYSTEM, _build_prompt
from tests.knowledge.test_round import FakeLedger, FakeRag

ABSTRACT = (
    "We report conjugate observations of ionospheric irregularities. The equatorial plasma "
    "bubbles appear simultaneously in both hemispheres within 5 minutes."
)


def _meta_rec(key: str = "Schwartz2022", abstract: str = ABSTRACT, **kw) -> PaperRecord:
    base = dict(
        key=key,
        title="Conjugate plasma bubbles",
        authors=["A. Schwartz", "B. Lee"],
        year=2022,
        venue="JGR Space Physics",
        doi="10.1029/2022JA000001",
        abstract=abstract,
        download_status="metadata_only",
    )
    base.update(kw)
    return PaperRecord(**base)


class ContentRag(FakeRag):
    """FakeRag that also keeps each doc's full_docs content (the insert text), so the adoption
    class check reads real content the way LightRAG's full_docs.get_by_id serves it."""

    def __init__(self, *, contents=None, **kw):
        super().__init__(**kw)
        self.contents: dict[str, str] = dict(contents or {})
        self.full_docs_ids |= set(self.contents)
        rag = self

        class _FullDocs:
            async def get_by_id(self, did):
                if did in rag.contents:
                    return {"content": rag.contents[did]}
                return {"content": "..."} if did in rag.full_docs_ids else None

        self.full_docs = _FullDocs()

    async def apipeline_enqueue_documents(self, *, input, ids, file_paths):
        before = set(self.rows)
        await super().apipeline_enqueue_documents(input=input, ids=ids, file_paths=file_paths)
        for text, did in zip(input, ids):
            if did not in before and did in self.rows:
                self.contents[did] = text

    async def adelete_by_doc_id(self, doc_id):
        r = await super().adelete_by_doc_id(doc_id)
        if r.status == "success":
            self.contents.pop(doc_id, None)
        return r


@pytest.fixture
def fake_ledger(monkeypatch):
    fl = FakeLedger()
    monkeypatch.setattr(distill, "ledger", fl)
    monkeypatch.setattr(rnd, "ledger", fl)
    rnd._stuck.clear()
    return fl


# ---------------------------------------------------------------- the doc + fingerprint


def test_abstract_doc_starts_with_the_header_and_carries_metadata_and_abstract():
    text = ad.build_abstract_doc(_meta_rec(arxiv_id="2201.00001"))
    lines = text.splitlines()
    assert lines[0] == "[ABSTRACT ONLY — full text not available]"
    assert "Title: Conjugate plasma bubbles" in lines
    assert "Authors: A. Schwartz, B. Lee" in lines
    assert "Year: 2022" in lines
    assert "Venue: JGR Space Physics" in lines
    assert "DOI: 10.1029/2022JA000001" in lines
    assert "arXiv: 2201.00001" in lines
    assert lines[-1] == "Abstract: " + ABSTRACT
    assert ad.is_abstract_text(text)
    assert not ad.is_abstract_text("# Introduction\nFull text body.")


def test_abstract_doc_omits_empty_fields_and_strips_markup():
    rec = _meta_rec(abstract="<jats:p>Bubbles   form\n at dusk.</jats:p>", venue="", doi="",
                    year=None, authors=[])
    text = ad.build_abstract_doc(rec)
    assert "Venue:" not in text and "DOI:" not in text and "Year:" not in text
    assert "Authors:" not in text and "arXiv:" not in text
    assert text.splitlines()[-1] == "Abstract: Bubbles form at dusk."


def test_no_abstract_means_no_doc():
    assert ad.build_abstract_doc(_meta_rec(abstract="")) is None
    assert ad.build_abstract_doc(_meta_rec(abstract="  <p> </p> ")) is None


def test_abstract_doc_stays_one_chunk_even_for_an_oversized_abstract():
    from lightrag.utils import TiktokenTokenizer

    from papervault.knowledge.config import CONFIG
    from papervault.knowledge.ingest.chunking import chunking_by_sentence_boundary

    huge = " ".join(
        f"Sentence {i} reports a {i}.{i % 7} nT perturbation at L={i % 9}.{i % 5} (Fig. {i})."
        for i in range(2000)
    )
    rec = _meta_rec(abstract=huge, authors=[f"Author{i} X." for i in range(300)],
                    title="T" * 5000)
    text = ad.build_abstract_doc(rec)
    tok = TiktokenTokenizer("gpt-4o-mini")
    size = CONFIG.lightrag.chunk_token_size
    assert len(tok.encode(text)) <= size
    chunks = chunking_by_sentence_boundary(
        tok, text, None, False, CONFIG.lightrag.chunk_overlap_token_size, size)
    assert len(chunks) == 1
    assert "et al." in text  # the author list is capped, not dropped


def test_fingerprint_marks_abstract_docs_with_their_own_class(monkeypatch):
    monkeypatch.delenv("KS_ABSTRACT_DOCS", raising=False)
    rec = _meta_rec()
    fp = fingerprint(rec)
    assert fp.startswith("ABSTRACT:") and fp != META
    assert fp == ad.abstract_fingerprint(ad.build_abstract_doc(rec))
    assert ad.is_abstract_fp(fp) and not ad.is_abstract_fp(META) and not ad.is_abstract_fp("ab" * 32)
    assert fingerprint(_meta_rec()) == fp                                 # deterministic
    assert fingerprint(_meta_rec(abstract=ABSTRACT + " More.")) != fp     # abstract change flips it


def test_fingerprint_without_abstract_stays_meta():
    assert fingerprint(_meta_rec(abstract="")) == META


@pytest.mark.parametrize("status", ["ok", "pending", "extract_failed", "failed", ""])
def test_only_metadata_only_papers_get_an_abstract_doc(status):
    # The class is the library's download_status=metadata_only (#144 Goal). A paper still on its way
    # to full text (PDF awaiting OCR, a pending/failed download) keeps META — no abstract build that
    # the full text would delete again days later.
    rec = _meta_rec(download_status=status)
    assert ad.build_abstract_doc(rec) is None
    assert fingerprint(rec) == META


def test_index_entry_carries_download_status():
    rec = PaperRecord.from_index_entry({"key": "K", "download_status": "metadata_only"})
    assert rec.download_status == "metadata_only"
    assert PaperRecord.from_index_entry({"key": "K"}).download_status == ""


def test_kill_switch_returns_metadata_papers_to_meta(monkeypatch):
    monkeypatch.setenv("KS_ABSTRACT_DOCS", "0")
    assert fingerprint(_meta_rec()) == META
    monkeypatch.setenv("KS_ABSTRACT_DOCS", "1")
    assert fingerprint(_meta_rec()).startswith("ABSTRACT:")


def test_done_abstract_is_a_valid_success_status():
    assert "done_abstract" in VALID_STATUS
    assert _next_attempts_status(2, "ABSTRACT:x", "done_abstract", "ABSTRACT:x", 3) == (0, "done_abstract")
    assert ad.done_status_for("ABSTRACT:x") == "done_abstract"
    assert ad.done_status_for("ab" * 32) == "done"
    assert ad.done_status_for(None) == "done"


# ---------------------------------------------------------------- distill + round


@pytest.mark.asyncio
async def test_distill_inserts_one_abstract_doc_and_reconcile_marks_done_abstract(fake_ledger, monkeypatch):
    monkeypatch.setattr(distill, "read_extract_raw", lambda *a, **k: pytest.fail("no full text read"))
    rec = _meta_rec()
    fp = ad.abstract_fingerprint(ad.build_abstract_doc(rec))
    rag = ContentRag()

    counters = await distill.distill_batch(rag, [(rec, fp)])

    assert counters["queued"] == 1 and counters["abstract"] == 1
    assert rag.contents["paper:Schwartz2022"] == ad.build_abstract_doc(rec)
    assert fake_ledger.rows["Schwartz2022"].status == "processing"
    assert fake_ledger.rows["Schwartz2022"].fingerprint == fp

    rag.set_status("paper:Schwartz2022", DocStatus.PROCESSED)
    await rnd.reconcile_terminal(rag)
    assert fake_ledger.rows["Schwartz2022"].status == "done_abstract"


@pytest.mark.asyncio
async def test_metadata_without_abstract_stays_done_meta_through_a_round(fake_ledger, monkeypatch):
    rec = _meta_rec(abstract="")
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: {rec.key: rec})
    rag = ContentRag()

    await rnd.run_round(rag)

    assert fake_ledger.rows[rec.key].status == "done_meta"
    assert fake_ledger.rows[rec.key].fingerprint == META
    assert not any(kind == "enqueue" for kind, _ in rag.calls)


@pytest.mark.asyncio
async def test_rollout_redistills_existing_done_meta_rows(fake_ledger, monkeypatch):
    # Bound 5: no bulk job — the new fingerprint alone moves a done_meta row into the graph.
    rec = _meta_rec()
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: {rec.key: rec})
    fake_ledger.rows = {rec.key: LedgerRecord("l0_probe", "paper", rec.key, META,
                                              "paper:" + rec.key, "done_meta")}
    rag = ContentRag()

    summary = await rnd.run_round(rag)

    assert summary["to_redistill"] == 1
    assert ad.is_abstract_text(rag.contents["paper:" + rec.key])
    rag.set_status("paper:" + rec.key, DocStatus.PROCESSED)
    await rnd.run_round(rag)
    assert fake_ledger.rows[rec.key].status == "done_abstract"


@pytest.mark.asyncio
async def test_full_text_arrival_deletes_and_replaces_the_abstract_doc(fake_ledger, monkeypatch, tmp_path):
    # Bound 2: the paper gains full text → fingerprint changes → the abstract doc is deleted and the
    # full-text doc inserted in its place; the ledger ends `done`, not `done_abstract`.
    rec = _meta_rec()
    abstract_text = ad.build_abstract_doc(rec)
    fp_abs = ad.abstract_fingerprint(abstract_text)
    fake_ledger.rows = {rec.key: LedgerRecord("l0_probe", "paper", rec.key, fp_abs,
                                              "paper:" + rec.key, "done_abstract")}
    rag = ContentRag(doc_statuses={"paper:" + rec.key: DocStatus.PROCESSED},
                     contents={"paper:" + rec.key: abstract_text})

    upgraded = _meta_rec(md_path="extracts/md/Schwartz2022.md")
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: {rec.key: upgraded})
    monkeypatch.setattr(rnd, "fingerprint", lambda r: "f" * 64)
    monkeypatch.setattr(distill, "read_extract_raw", lambda *a, **k: "# Methods\nFull body text.")

    await rnd.run_round(rag)

    assert ("delete", "paper:" + rec.key) in rag.calls
    assert rag.contents["paper:" + rec.key] == "# Methods\nFull body text."
    assert fake_ledger.rows[rec.key].fingerprint == "f" * 64
    rag.set_status("paper:" + rec.key, DocStatus.PROCESSED)
    await rnd.run_round(rag)
    assert fake_ledger.rows[rec.key].status == "done"


@pytest.mark.asyncio
async def test_adoption_never_keeps_an_abstract_doc_in_place_of_full_text(fake_ledger, monkeypatch):
    # The #131 adopt-existing rule: a missing ledger row + an existing processed `paper:<key>` is
    # adopted. When that doc is the ABSTRACT doc and the wanted fingerprint is full text, adopting
    # would record `done` while the graph still holds only the abstract — so it is replaced.
    rec = _meta_rec(md_path="extracts/md/Schwartz2022.md")
    rag = ContentRag(doc_statuses={"paper:Schwartz2022": DocStatus.PROCESSED},
                     contents={"paper:Schwartz2022": ad.build_abstract_doc(_meta_rec())})
    monkeypatch.setattr(distill, "read_extract_raw", lambda *a, **k: "Full body text.")

    counters = await distill.distill_batch(rag, [(rec, "f" * 64)])

    assert counters["existing"] == 0 and counters["replaced"] == 1 and counters["queued"] == 1
    assert ("delete", "paper:Schwartz2022") in rag.calls
    assert rag.contents["paper:Schwartz2022"] == "Full body text."
    assert rag.created_markers == []
    assert fake_ledger.rows["Schwartz2022"].status == "processing"


@pytest.mark.asyncio
async def test_adoption_keeps_a_doc_of_the_wanted_class(fake_ledger):
    rec = _meta_rec()
    text = ad.build_abstract_doc(rec)
    fp = ad.abstract_fingerprint(text)
    rag = ContentRag(doc_statuses={"paper:Schwartz2022": DocStatus.PROCESSED},
                     contents={"paper:Schwartz2022": text})

    counters = await distill.distill_batch(rag, [(rec, fp)])

    assert counters["existing"] == 1 and counters["replaced"] == 0
    assert not any(kind in ("delete", "enqueue") for kind, _ in rag.calls)
    assert fake_ledger.rows["Schwartz2022"].status == "done_abstract"


@pytest.mark.asyncio
async def test_adoption_replaces_a_full_text_doc_when_the_abstract_is_wanted(fake_ledger):
    rec = _meta_rec()
    fp = ad.abstract_fingerprint(ad.build_abstract_doc(rec))
    rag = ContentRag(doc_statuses={"paper:Schwartz2022": DocStatus.PROCESSED},
                     contents={"paper:Schwartz2022": "# Intro\nOld full text."})

    counters = await distill.distill_batch(rag, [(rec, fp)])

    assert counters["replaced"] == 1 and counters["queued"] == 1
    assert ad.is_abstract_text(rag.contents["paper:Schwartz2022"])


@pytest.mark.asyncio
async def test_replacement_blocked_by_a_busy_pipeline_is_not_enqueued(fake_ledger, monkeypatch):
    rec = _meta_rec(md_path="extracts/md/Schwartz2022.md")
    rag = ContentRag(doc_statuses={"paper:Schwartz2022": DocStatus.PROCESSED},
                     contents={"paper:Schwartz2022": ad.build_abstract_doc(_meta_rec())},
                     delete_results={"paper:Schwartz2022": "not_allowed"})
    monkeypatch.setattr(distill, "read_extract_raw", lambda *a, **k: "Full body text.")

    counters = await distill.distill_batch(rag, [(rec, "f" * 64)])

    assert counters["queued"] == 0 and counters["replace_failed"] == 1
    assert not any(kind == "enqueue" for kind, _ in rag.calls)
    assert fake_ledger.rows["Schwartz2022"].status == "pending_remove"


# ---------------------------------------------------------------- synthesis honesty


def test_synth_prompt_carries_exactly_one_abstract_only_rule():
    assert _SYNTH_SYSTEM.count("abstract-only") == 1
    rule = " ".join(_SYNTH_SYSTEM.split("6b.", 1)[1].split("\n\n", 1)[0].split())
    assert "credibility=abstract-only" in rule
    assert "cite it only for what its abstract states" in rule
    assert "never for methods, numbers or conclusions the abstract does not contain" in rule
    # the #122 single citation rule is untouched
    assert "[textbook:Griffiths]" in _SYNTH_SYSTEM


def test_abstract_chunk_reaches_the_prompt_labelled_abstract_only():
    text = ad.build_abstract_doc(_meta_rec())
    prompt = _build_prompt({"chunks": [
        {"file_path": "Schwartz2022", "content": text},
        {"file_path": "Reames2023", "content": "SEP onset timing."},
    ]}, "Conjugate bubbles?")
    assert "(source key: Schwartz2022 | credibility: abstract-only)" in prompt
    assert "(source key: Reames2023 | credibility: empirical)" in prompt
    assert "Abstract: " + ABSTRACT in prompt
    # the header's own square brackets never reach the prompt as a citation lookalike (#122)
    assert "[ABSTRACT ONLY" not in prompt


def test_query_outfeed_marks_cited_abstract_only_papers():
    data = {
        "chunks": [
            {"file_path": "Schwartz2022", "content": ad.build_abstract_doc(_meta_rec())},
            {"file_path": "Reames2023", "content": "SEP onset timing."},
            {"file_path": "textbook:Griffiths", "content": ad.ABSTRACT_HEADER + "\nodd"},
        ],
        "references": [{"file_path": "Schwartz2022"}, {"file_path": "Reames2023"}],
    }
    assert _abstract_only_papers(data) == ["Schwartz2022"]
    assert _EMPTY["abstract_only_papers"] == []


# ---------------------------------------------------------------- rollback


def _abstract_ledger():
    return FakeLedger({
        "A": LedgerRecord("l0_probe", "paper", "A", "ABSTRACT:a", "paper:A", "done_abstract"),
        "B": LedgerRecord("l0_probe", "paper", "B", "ABSTRACT:b", "paper:B", "processing"),
        "F": LedgerRecord("l0_probe", "paper", "F", "f" * 64, "paper:F", "done"),
        "M": LedgerRecord("l0_probe", "paper", "M", META, "paper:M", "done_meta"),
    })


def _rollback_rag():
    return ContentRag(
        doc_statuses={"paper:A": DocStatus.PROCESSED, "paper:B": DocStatus.PENDING,
                      "paper:F": DocStatus.PROCESSED},
        contents={"paper:A": ad.ABSTRACT_HEADER + "\nTitle: a",
                  "paper:B": ad.ABSTRACT_HEADER + "\nTitle: b",
                  "paper:F": "Full text."},
    )


@pytest.mark.asyncio
async def test_rollback_dry_run_counts_and_touches_nothing(monkeypatch):
    fl = _abstract_ledger()
    monkeypatch.setattr(distill, "ledger", fl)
    rag = _rollback_rag()

    res = await distill.rollback_abstract_docs(rag, apply=False)

    assert res["abstract_rows"] == 2 and res["by_status"] == {"done_abstract": 1, "processing": 1}
    assert res["reverted"] == 0 and sorted(res["keys"]) == ["A", "B"]
    assert not any(kind == "delete" for kind, _ in rag.calls)
    assert fl.rows["A"].status == "done_abstract"


@pytest.mark.asyncio
async def test_rollback_write_deletes_abstract_docs_and_returns_rows_to_done_meta(monkeypatch):
    fl = _abstract_ledger()
    monkeypatch.setattr(distill, "ledger", fl)
    rag = _rollback_rag()

    res = await distill.rollback_abstract_docs(rag, apply=True)

    assert res["reverted"] == 2 and res["failed"] == []
    for key in ("A", "B"):
        assert fl.rows[key].status == "done_meta" and fl.rows[key].fingerprint == META
        assert "paper:" + key not in rag.rows
    assert "paper:F" in rag.rows and fl.rows["F"].status == "done"      # full text untouched
    assert fl.rows["M"].status == "done_meta"


@pytest.mark.asyncio
async def test_rollback_never_deletes_a_doc_that_is_full_text(monkeypatch):
    fl = FakeLedger({"A": LedgerRecord("l0_probe", "paper", "A", "ABSTRACT:a", "paper:A", "processing")})
    monkeypatch.setattr(distill, "ledger", fl)
    rag = ContentRag(doc_statuses={"paper:A": DocStatus.PROCESSED},
                     contents={"paper:A": "Full text that replaced it."})

    res = await distill.rollback_abstract_docs(rag, apply=True)

    assert res["skipped_not_abstract"] == ["A"] and res["reverted"] == 0
    assert "paper:A" in rag.rows and fl.rows["A"].status == "processing"


def _noop_async(*a, **k):
    async def _n():
        return None
    return _n()


def test_rollback_cli_defaults_to_dry_run(monkeypatch):
    fl = _abstract_ledger()
    monkeypatch.setattr(distill, "ledger", fl)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", _noop_async)
    monkeypatch.setattr(cli_mod, "_active_service", lambda: pytest.fail("dry-run must not probe"))

    res = CliRunner().invoke(cli_mod.cli, ["rollback-abstracts", "--json"])

    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["dry_run"] is True and payload["abstract_rows"] == 2
    assert fl.rows["A"].status == "done_abstract"


def test_rollback_cli_apply_refuses_while_the_service_is_active(monkeypatch):
    monkeypatch.setattr(cli_mod, "_active_service", lambda: ["papervault.service"])
    monkeypatch.setattr("papervault.knowledge.store.graph.get_graph",
                        lambda: pytest.fail("must refuse before opening the graph"))

    res = CliRunner().invoke(cli_mod.cli, ["rollback-abstracts", "--apply"])

    assert res.exit_code == 2
    assert "papervault.service" in res.output


def test_rollback_cli_apply_writes_when_the_service_is_stopped(monkeypatch):
    fl = _abstract_ledger()
    monkeypatch.setattr(distill, "ledger", fl)
    rag = _rollback_rag()

    async def _get_graph():
        return rag

    async def _finalize():
        return None

    rag.finalize_storages = _finalize
    monkeypatch.setattr(cli_mod, "_active_service", lambda: [])
    monkeypatch.setattr("papervault.knowledge.store.graph.get_graph", _get_graph)
    monkeypatch.setattr("papervault.knowledge.store.graph.close_graph", _noop_async)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", _noop_async)

    res = CliRunner().invoke(cli_mod.cli, ["rollback-abstracts", "--apply"])

    assert res.exit_code == 0, res.output
    assert fl.rows["A"].status == "done_meta" and fl.rows["B"].status == "done_meta"
    assert "KS_ABSTRACT_DOCS=0" in res.output
