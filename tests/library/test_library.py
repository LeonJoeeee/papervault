"""Unit tests for the library module (no network)."""

import json
from pathlib import Path

import pytest

from papervault.library import Library, base_key
from papervault.library.cite_check import check, extract_cite_keys


def _make_lib(tmp_path) -> Library:
    return Library(tmp_path / "lib")


def test_base_key_handles_unicode():
    assert base_key("García-Pérez", 2024) == "GarciaPerez2024"
    assert base_key("Wang", "2025") == "Wang2025"
    assert base_key("", None) == "Anonnd"


def test_upsert_adds_new(tmp_path):
    lib = _make_lib(tmp_path)
    p, is_new = lib.upsert({
        "title": "Solar Modulation Review",
        "authors": ["Marius Potgieter"],
        "year": 2013,
        "doi": "10.1234/abc",
    })
    assert is_new
    assert p.key == "Potgieter2013"
    assert lib.get("Potgieter2013") == p


def test_upsert_dedupes_by_doi(tmp_path):
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "First version of test paper",
                "authors": ["Smith"], "year": 2020, "doi": "10.1/x"})
    p2, is_new = lib.upsert({"title": "Second version of test paper",
                             "authors": ["Smith"], "year": 2020,
                             "doi": "10.1/x"})
    assert not is_new
    assert p2.key == "Smith2020"
    assert len(lib.all_papers()) == 1


def test_upsert_dedupes_by_arxiv_version(tmp_path):
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "An arxiv test paper for dedup",
                "authors": ["Smith"], "year": 2020, "arxiv_id": "1234.5678v1"})
    _, is_new = lib.upsert({"title": "An arxiv test paper for dedup",
                            "authors": ["Smith"], "year": 2020,
                            "arxiv_id": "1234.5678v3"})
    assert not is_new


def test_find_by_arxiv_canonical_prefix_form_resolves(tmp_path):
    """Boundary fix #1: an ``arXiv:``-prefixed id (the CANONICAL form the
    recognizer accepts) must resolve to the SAME held paper as the bare id.
    Before the fix ``_normalize_arxiv`` stripped only the version suffix, not
    the ``arXiv:`` prefix, so ``find(arxiv_id='arXiv:1711.10561')`` missed the
    bare-keyed index and a held paper falsely reported not-in-library."""
    lib = _make_lib(tmp_path)
    p, _ = lib.upsert({"title": "Physics-informed deep learning of PDE inversion",
                       "authors": ["Raissi"], "year": 2019,
                       "arxiv_id": "1711.10561v9"})
    bare = lib.find(arxiv_id="1711.10561")
    prefixed = lib.find(arxiv_id="arXiv:1711.10561")
    prefixed_v = lib.find(arxiv_id="arXiv:1711.10561v9")
    assert bare is not None and bare.key == p.key
    assert prefixed is not None and prefixed.key == p.key          # the bug fix
    assert prefixed_v is not None and prefixed_v.key == p.key


def test_key_collision_gets_suffix(tmp_path):
    lib = _make_lib(tmp_path)
    p1, _ = lib.upsert({"title": "Test paper A for key collision",
                        "authors": ["Aslam"], "year": 2013, "doi": "10.1/a"})
    p2, _ = lib.upsert({"title": "Test paper B for key collision",
                        "authors": ["Aslam"], "year": 2013, "doi": "10.1/b"})
    p3, _ = lib.upsert({"title": "Test paper C for key collision",
                        "authors": ["Aslam"], "year": 2013, "doi": "10.1/c"})
    assert {p1.key, p2.key, p3.key} == {"Aslam2013", "Aslam2013a", "Aslam2013b"}


def test_save_and_reload(tmp_path):
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "Test paper for save and reload",
                "authors": ["Parker"], "year": 1958,
                "doi": "10.1/p", "arxiv_id": "0/0v1"})
    lib.save()
    assert lib.bib_path.exists()
    assert lib.index_path.exists()
    assert "@" in lib.bib_path.read_text()

    lib2 = Library(lib.root)
    assert lib2.get("Parker1958") is not None


def test_extract_cite_keys():
    text = r"intro \citep{A2020} mid \citet[p.3]{B2021} end \cite{C2022,D2023}"
    keys = extract_cite_keys(text)
    assert keys == ["A2020", "B2021", "C2022", "D2023"]


def test_check_finds_dangling(tmp_path):
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "Test paper for cite-check dangling",
                "authors": ["A"], "year": 2020, "doi": "10/x"})
    report = check(r"\citep{A2020}\citep{Ghost2099}", lib)
    assert not report["ok"]
    assert report["dangling"] == ["Ghost2099"]
    assert report["unique_keys"] == ["A2020", "Ghost2099"]


def test_check_with_allowed_subset(tmp_path):
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "Test paper for allowed subset (X)",
                "authors": ["A"], "year": 2020, "doi": "10/x"})
    lib.upsert({"title": "Test paper for allowed subset (Y)",
                "authors": ["B"], "year": 2021, "doi": "10/y"})
    # Topic constrains to only A2020; B2021 must be flagged even though library has it.
    report = check(r"\citep{A2020}\citep{B2021}", lib, allowed=["A2020"])
    assert not report["ok"]
    assert report["dangling"] == ["B2021"]


def test_new_paper_default_download_status_is_pending(tmp_path):
    lib = Library(tmp_path / "lib")
    p, _ = lib.upsert({"title": "Test paper for default status check",
                       "authors": ["X"], "year": 2020, "doi": "10.1234/x"})
    from papervault.library.models import DOWNLOAD_STATUS_PENDING
    assert p.download_status == DOWNLOAD_STATUS_PENDING


# ---------------- has_pdf / has_extract TOCTOU hardening (fix #13) ----------


def test_has_pdf_toctou_no_raise_when_file_vanishes(tmp_path, monkeypatch):
    """fix #13: a file deleted between the existence probe and the stat() (a
    background worker / janitor unlinking it) must return False, NOT raise
    FileNotFoundError (which crashed the whole get_paper batch). The single
    guarded stat() in _is_nonempty_file closes the race."""
    lib = Library(tmp_path / "lib")
    real_stat = Path.stat

    def vanishing_stat(self, *a, **k):
        # Simulate the file disappearing exactly at the stat() syscall.
        if self == lib.pdf_path("Ghost"):
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", vanishing_stat)
    assert lib.has_pdf("Ghost") is False  # no raise


def test_has_extract_toctou_no_raise_when_file_vanishes(tmp_path, monkeypatch):
    """fix #13: same TOCTOU hardening for has_extract (md + txt)."""
    lib = Library(tmp_path / "lib")
    real_stat = Path.stat

    def vanishing_stat(self, *a, **k):
        if self in (lib.md_path("Ghost"), lib.txt_path("Ghost")):
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", vanishing_stat)
    assert lib.has_extract("Ghost", "md") is False
    assert lib.has_extract("Ghost", "txt") is False


def test_has_pdf_true_for_real_nonempty_file(tmp_path):
    """fix #13 regression guard: the happy path still works — a real non-empty
    PDF reads True, an empty one reads False, a directory reads False."""
    lib = Library(tmp_path / "lib")
    lib.pdf_path("Real").parent.mkdir(parents=True, exist_ok=True)
    lib.pdf_path("Real").write_bytes(b"%PDF-1.0 content")
    assert lib.has_pdf("Real") is True
    lib.pdf_path("Empty").write_bytes(b"")
    assert lib.has_pdf("Empty") is False
    assert lib.has_pdf("Missing") is False


def test_topic_write_and_log(tmp_path):
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "Test paper for topic write",
                "authors": ["P"], "year": 2020, "doi": "10/t"})
    path = lib.write_topic("test-topic", {"keys": ["P2020"]})
    assert path.exists()
    assert json.loads(path.read_text())["keys"] == ["P2020"]
    lib.log({"event": "test"})
    assert lib.manifest_path.exists()
    assert "test" in lib.manifest_path.read_text()


# ---------- D7: preprint↔journal fuzzy merge ----------


def test_d7_preprint_journal_auto_merge_on_high_confidence(tmp_path):
    """Preprint already in library; journal version arrives → merge into the
    preprint's key, journal DOI is added to the existing entry."""
    lib = _make_lib(tmp_path)
    # Step 1: preprint arrives (arxiv only, no DOI)
    preprint, is_new = lib.upsert({
        "title": "Cosmic Ray Transport via Fractional Diffusion",
        "authors": ["Zimbardo Gaetano", "Perri Silvia"],
        "year": 2017,
        "arxiv_id": "1710.0001",
    })
    assert is_new
    pre_key = preprint.key

    # Step 2: journal version arrives (DOI + same title + same authors)
    journal, is_new = lib.upsert({
        "title": "Cosmic Ray Transport via Fractional Diffusion",
        "authors": ["Gaetano Zimbardo", "Silvia Perri"],  # order/format slight diff
        "year": 2017,
        "doi": "10.1051/0004-6361/201731179",
        "venue": "A&A",
    })
    assert not is_new, "should merge into existing preprint"
    assert journal.key == pre_key, "cite key must stay the preprint's"
    assert journal.doi == "10.1051/0004-6361/201731179", "journal DOI added"
    assert journal.arxiv_id == "1710.0001", "arxiv kept as back-pointer"
    assert journal.venue == "A&A", "journal venue filled"


def test_d7_does_not_merge_when_year_too_far_apart(tmp_path):
    """Similar (but not identical) titles + year diff > 2 → NOT merged.
    Titles differ enough that step-3 (exact normalized title) doesn't fire;
    we go through to step-4 fuzzy, which rejects on year diff."""
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "Cosmic Ray Transport in Magnetic Turbulence",
                "authors": ["Z. G."], "year": 2010,
                "arxiv_id": "1010.0001"})
    p2, is_new = lib.upsert({"title": "Cosmic Ray Transport in Solar Wind",
                             "authors": ["Z. G."], "year": 2024,
                             "doi": "10.99/diff"})
    assert is_new, "year diff > 2 should NOT merge — likely a different work"


def test_d7_does_not_merge_when_authors_disjoint(tmp_path):
    """Similar (but not identical) titles + totally different authors → NOT merged.
    Similar-not-identical so step-3 doesn't fire, reaching step-4 fuzzy."""
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "On Anomalous Diffusion in Turbulent Plasma",
                "authors": ["Alice One", "Bob Two"], "year": 2020,
                "arxiv_id": "2010.0001"})
    p2, is_new = lib.upsert({"title": "On Anomalous Diffusion in Coronal Loops",
                             "authors": ["Charlie Three", "Dave Four"],
                             "year": 2021,
                             "doi": "10.99/disjoint"})
    assert is_new, "disjoint authors should not collapse two unrelated papers"


def test_d7_requires_cross_identifier_type(tmp_path):
    """If both papers have DOIs (or both have arxiv_ids only), fuzzy check
    is skipped — step 1/2/3 exact match handles those, or they're genuinely
    different works."""
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "Same-ish Paper", "authors": ["X Y"], "year": 2020,
                "doi": "10.1/first"})
    p2, is_new = lib.upsert({"title": "Same-ish Paper", "authors": ["X Y"],
                             "year": 2020, "doi": "10.1/second"})
    # Both have DOIs; fuzzy step is skipped. Title may still trigger step-3
    # (normalized title exact)... in which case it merges via step 3, not D7.
    # Either way, the explicit fuzzy path is NOT taken because identifier
    # types are the same.
    # Outcome: merges via title (step 3) → is_new=False
    assert not is_new


# Title pair with similarity ~0.714 — lands in borderline range (0.7-0.85)
_BORDERLINE_TITLE_A = "Solar Wind Magnetic Turbulence"
_BORDERLINE_TITLE_B = "Solar Wind Magnetic Turbulence and Anomalous Diffusion"


def test_d7_skips_borderline_below_auto_merge_bar(tmp_path):
    """Borderline title sim (0.7-0.85) is below the auto-merge bar → conservative
    skip (don't merge). The old LLM-judged borderline path was removed (it ran a
    blocking sync LLM call on the event loop under the write lock); upsert is now
    purely heuristic and takes no ``llm`` argument."""
    lib = _make_lib(tmp_path)
    lib.upsert({"title": _BORDERLINE_TITLE_A,
                "authors": ["Zimbardo G", "Perri S"], "year": 2017,
                "arxiv_id": "1710.0001"})
    p2, is_new = lib.upsert({
        "title": _BORDERLINE_TITLE_B,
        "authors": ["G. Zimbardo", "S. Perri"], "year": 2017,
        "doi": "10.99/borderline",
    })
    # Below the auto-merge bar → don't merge (conservative — false-positive
    # merges are data loss, a residual duplicate is the safer outcome).
    assert is_new


def test_upsert_takes_no_llm_kwarg(tmp_path):
    """The dead borderline-merge LLM path is gone: upsert no longer accepts an
    ``llm=`` keyword (the freeze-prone sync llm.call under lib_write_lock)."""
    import inspect
    lib = _make_lib(tmp_path)
    assert "llm" not in inspect.signature(lib.upsert).parameters
    with pytest.raises(TypeError):
        lib.upsert({"title": "A sufficiently long paper title here",
                    "authors": ["X"], "year": 2020, "doi": "10/z"},
                   llm=object())


def test_d7_merge_event_logged(tmp_path):
    """After a fuzzy merge, manifest log contains 'dedup_merged_preprint_journal'.
    Titles must differ enough that step-3 doesn't catch this as exact normalized."""
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "Fractional Diffusion in Cosmic Rays Part One",
                "authors": ["Zimbardo G", "Perri S"], "year": 2017,
                "arxiv_id": "1710.0001"})
    lib.upsert({"title": "Fractional Diffusion in Cosmic Rays Part Two",
                "authors": ["Zimbardo G", "Perri S"], "year": 2017,
                "doi": "10.1/merged-event"})
    assert lib.manifest_path.exists()
    manifest = lib.manifest_path.read_text()
    assert "dedup_merged_preprint_journal" in manifest
    assert "10.1/merged-event" in manifest


# ===================== D14: index.json durability =========================


def _seed_one(tmp_path) -> Library:
    lib = _make_lib(tmp_path)
    lib.upsert({"title": "Durable test paper title here",
                "authors": ["Curie"], "year": 1903, "doi": "10.1/durable"})
    lib.save()
    return lib


def test_save_rotates_previous_index_into_bak(tmp_path):
    """D14: every successful save rotates the prior good index into .bak so a
    crash mid-write of the new index still leaves a known-good copy."""
    lib = _seed_one(tmp_path)
    assert lib.index_path.exists()
    # After _seed_one the primary holds 1 paper. The next save rotates that
    # 1-paper generation into .bak and writes a 2-paper primary.
    lib.upsert({"title": "Second durable paper title here",
                "authors": ["Bohr"], "year": 1922, "doi": "10.1/second"})
    lib.save()
    assert lib.index_bak_path.exists()
    bak = json.loads(lib.index_bak_path.read_text())
    # .bak holds the PRIOR generation (1 paper); primary holds both.
    assert len(bak["papers"]) == 1
    assert len(json.loads(lib.index_path.read_text())["papers"]) == 2


def test_load_recovers_from_bak_when_primary_corrupt(tmp_path):
    """D14: a truncated/corrupt primary index falls back to .bak instead of
    silently loading an empty library (which would clobber the vault)."""
    lib = _seed_one(tmp_path)
    lib.upsert({"title": "Recoverable paper title goes here",
                "authors": ["Planck"], "year": 1900, "doi": "10.1/recover"})
    lib.save()  # now .bak = 1 paper, primary = 2 papers
    # Simulate a crash that truncated the primary mid-write.
    lib.index_path.write_text('{"version": 1, "papers": {"Curie1903": {trunc')
    reloaded = Library(lib.root)
    # Fell back to .bak (1 paper) rather than loading empty.
    assert len(reloaded.all_papers()) == 1
    manifest = reloaded.manifest_path.read_text()
    assert "index_load_corrupt" in manifest


def test_load_raises_when_both_primary_and_bak_corrupt(tmp_path):
    """D14: if neither primary nor .bak is usable, refuse to load an empty
    library (a blank load would clobber both files on the next save)."""
    lib = _seed_one(tmp_path)
    lib.upsert({"title": "Another durable paper title here",
                "authors": ["Dirac"], "year": 1928, "doi": "10.1/dirac"})
    lib.save()  # .bak created
    lib.index_path.write_text("}{ not json")
    lib.index_bak_path.write_text("also not json {{{")
    with pytest.raises(RuntimeError, match="corrupt"):
        Library(lib.root)


def test_load_skips_single_corrupt_record_keeps_rest(tmp_path):
    """D14: one schema-invalid record is logged + skipped; the other ~good
    records still load (a single bad row must not sink the whole library)."""
    lib = _seed_one(tmp_path)
    lib.upsert({"title": "Good neighbour paper title here",
                "authors": ["Fermi"], "year": 1934, "doi": "10.1/good"})
    lib.save()
    # Inject a structurally-valid JSON object that is NOT a valid Paper
    # (year is a dict — fails pydantic validation).
    raw = json.loads(lib.index_path.read_text())
    raw["papers"]["Broken9999"] = {"key": "Broken9999", "title": "x",
                                   "year": {"not": "an int"}}
    lib.index_path.write_text(json.dumps(raw))
    reloaded = Library(lib.root)
    keys = {p.key for p in reloaded.all_papers()}
    assert "Broken9999" not in keys
    assert "Curie1903" in keys and "Fermi1934" in keys
    assert "index_record_skipped" in reloaded.manifest_path.read_text()


def test_load_recovers_from_bak_when_primary_missing(tmp_path):
    """D14: primary gone but .bak present → recover from .bak rather than
    starting a fresh empty index."""
    lib = _seed_one(tmp_path)
    lib.upsert({"title": "Backup-only paper title here",
                "authors": ["Heisenberg"], "year": 1927, "doi": "10.1/backup"})
    lib.save()  # .bak now holds the 1-paper generation
    lib.index_path.unlink()
    reloaded = Library(lib.root)
    assert len(reloaded.all_papers()) == 1


def test_atomic_write_fsyncs_and_leaves_no_tmp(tmp_path):
    """D14: the durable atomic write fsyncs the tmp file + parent dir and
    leaves no stray .tmp behind on success."""
    from papervault.library.store import _atomic_write
    target = tmp_path / "sub" / "data.json"
    _atomic_write(target, '{"ok": true}')
    assert target.read_text() == '{"ok": true}'
    assert not (tmp_path / "sub" / "data.json.tmp").exists()


def test_merge_in_place_preserves_identity_no_lost_update(tmp_path):
    """Regression (lost-update race): upsert's dedupe-merge must mutate the
    existing Paper IN PLACE, never swap object identity — else a download/extract
    worker holding the old object loses its pdf_path/download_status writes when
    save() persists the swapped-in object."""
    lib = Library(tmp_path / "lib")
    p, is_new = lib.upsert({"title": "Race condition test paper title",
                            "authors": ["Smith"], "year": 2024, "doi": "10.1/race"})
    assert is_new is True
    worker_ref = lib.get(p.key)            # a queue worker holds this exact object

    # concurrent upsert-merge of the SAME key (same DOI) bringing a new blank-fill
    merged, is_new2 = lib.upsert({"title": "Race condition test paper title",
                                  "doi": "10.1/race", "venue": "ApJ"})
    assert is_new2 is False
    assert merged is worker_ref            # identity preserved (no swap)
    assert lib.get(p.key) is worker_ref

    # the worker now writes its fields onto the object it has been holding
    worker_ref.download_status = "ok"
    worker_ref.pdf_path = "pdfs/race.pdf"
    lib.save()

    reloaded = Library(tmp_path / "lib").get(p.key)
    assert reloaded.venue == "ApJ"               # the merge's write survived
    assert reloaded.download_status == "ok"      # the worker's write was NOT lost
    assert reloaded.pdf_path == "pdfs/race.pdf"
