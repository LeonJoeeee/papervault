"""Phase 12.3 (#94): firecrawl pre-curator quality gate tests.

Verify _try_firecrawl_text_fallback runs review_extract on the firecrawl
md before flagging it as good. Bad md (paywall stub / landing page /
CAPTCHA) → ``paper.insight_invalid_reason`` set.

✦ Phase 28 (2026-05-24, route B): the original docstring referenced
the insight worker / 5-Q LLM that this gate was protecting from junk
input. That pipeline was removed; the gate is now a forward-looking
signal for the Librarian-side paper-curator and for the post-route-B
``list_undistilled`` filter (flagged papers are excluded so the
curator doesn't waste a turn on them).
"""
from __future__ import annotations

import json
import pytest
import responses

from papervault.library import Library, Paper, download


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path)


@pytest.fixture
def paper(lib):
    p, _ = lib.upsert({
        "title": "Test paper paywall stub", "authors": ["A"], "year": 2024,
        "doi": "10.1/firecrawl-test"
    })
    return p


@pytest.fixture
def firecrawl_env(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_URL", "https://api.firecrawl.dev")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")


# both fixtures must be > 1000 chars per firecrawl_too_short gate
REAL_PAPER_MD = (
    "# Introduction\n\nThis paper presents a novel approach to GCR transport "
    "modeling using neural networks for the inner heliosphere modulation "
    "regime. Section 2 introduces the model architecture, Section 3 describes "
    "our methodology in detail, Section 4 presents results, and Section 5 "
    "discusses physical implications. Prior work by Jokipii (1971) and Parker "
    "(1965) established the convection-diffusion framework; we extend these "
    "with data-driven coefficients.\n\n"
    "## Methodology\n\nWe train a 3-layer MLP on 24 SEP events from PSP, "
    "SolO, and STEREO-A. Hyperparameters: batch 64, lr 1e-4, Adam optimizer, "
    "100 epochs. Random seed 42. Validation split 0.2 stratified by event "
    "phase. Loss = MSE + 0.1 L2 reg on coefficients.\n\n"
    "## Results\n\nWe find that the model achieves 92% R² on the test set, "
    "outperforming baseline Parker-Jokipii by 14%. The mean RMSE drops from "
    "0.34 to 0.11 across all 24 SEP events. Statistical significance: "
    "Wilcoxon paired test p < 0.001. Per-event breakdown in Table 1.\n\n"
    "## Discussion\n\nOur results suggest that κ_⊥ has 5-16× polarity "
    "asymmetry, refuting prior κ_⊥ = const assumption. This has implications "
    "for SEP forecast accuracy in the next solar cycle.\n\n"
    "## References\n\n[1] Smith et al. 2020. Nature 580: 100-110.\n"
    "[2] Jones 2021. ApJ 920: 45-60.\n"
    "[3] Lee 2022. JGR Space Physics 127: 1234.\n"
)

# Stub fixture — passes anti-bot regex (no 'access denied' / CAPTCHA / etc.)
# but is substantively NOT a real paper body. This is the gap review_extract
# LLM judge fills.
PAYWALL_STUB_MD = (
    "Subscribe to read the full article — full text requires an active "
    "institutional or individual subscription to this publication.\n\n"
    "Title: Cosmic ray modulation in the inner heliosphere — a review\n"
    "Authors: J. Smith, A. Jones, B. Lee, C. Wong, D. Park, E. Liu\n"
    "Abstract: We review recent progress in CR modulation studies. (Full "
    "abstract requires login.)\n"
    "Keywords: cosmic rays, modulation, heliosphere, solar cycle\n"
    "DOI: 10.1/x\n"
    "Journal: Journal of Cosmic Rays Quarterly, vol 42, issue 7\n"
    "Published: 2024-03-15\n\n"
    "Pricing options for individual purchase:\n"
    "  - One-time: $35.00 per article (30-day reading window)\n"
    "  - Pay-per-view: $12.00 per view (24-hour window)\n"
    "  - Annual subscription: $99/year (12 month rolling)\n"
    "  - Institutional license: contact our sales team\n\n"
    "If your institution provides this journal, please use your "
    "institutional login. Forgot password? Click here. Privacy policy. "
    "Terms of service. Cookie settings. Accessibility statement.\n\n"
    "You may also be interested in:\n"
    "  - Another article on cosmic rays (Smith 2023)\n"
    "  - Yet another publication in this journal (Jones 2022)\n"
    "  - More content from this research group (Brown 2024)\n"
    "  - Related survey article (Park 2021)\n\n"
    "Recommended for you based on your reading history. Sign up for our "
    "weekly newsletter to receive the latest research in your field. "
    "Find us on Twitter, Facebook, and LinkedIn for daily updates.\n\n"
    "Citation information: Smith et al. (2024). Cosmic ray modulation. "
    "JCRQ, 42(7), 100-120.\n"
)
assert len(REAL_PAPER_MD) > 1000, f"REAL_PAPER_MD = {len(REAL_PAPER_MD)}"
assert len(PAYWALL_STUB_MD) > 1000, f"PAYWALL_STUB_MD = {len(PAYWALL_STUB_MD)}"


def _patch_gate(monkeypatch, complete: bool, reason: str = ""):
    """Mock the D5 completeness_gate so these tests stay hermetic (no real LLM)
    and fast. The gate is what now decides whether a firecrawl md survives."""
    def fake_gate(text, *, llm=None):
        return {"complete": complete, "reason": reason}
    monkeypatch.setattr(
        "papervault.library.extract.completeness_gate", fake_gate)


@responses.activate
def test_firecrawl_real_paper_md_passes_gate_no_flag(
    lib, paper, firecrawl_env, monkeypatch
):
    """Real paper md → completeness_gate complete=True → md kept on disk, served
    as text_path; the downstream pre-curator review_extract leaves
    insight_invalid_reason None."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": REAL_PAPER_MD}},
        status=200,
    )
    _patch_gate(monkeypatch, complete=True, reason="complete")

    def fake_review(text, *, llm=None):
        return {"ok": True, "issues": [], "confidence": 0.95}

    monkeypatch.setattr(
        "papervault.library.extract.review_extract", fake_review)

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert paper.md_engine == "firecrawl"
    assert lib.has_extract(paper.key, "md")  # md survives the gate
    assert paper.insight_invalid_reason is None, \
        "real paper md must NOT set insight_invalid_reason"


@responses.activate
def test_firecrawl_paywall_stub_rejected_by_gate(
    lib, paper, firecrawl_env, monkeypatch
):
    """D5: a paywall stub fails the completeness_gate → the md is DELETED (not
    left on disk as a serveable text_path) and the paper is demoted to a
    terminal status. The 18-tier cascade already missed; firecrawl is the final
    answer, so "fail the gate → no full text at all, leave nothing on disk to
    retry". Has abstract → metadata_only."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": PAYWALL_STUB_MD}},
        status=200,
    )
    paper.abstract = "A real citable abstract for this paywalled paper."
    _patch_gate(monkeypatch, complete=False, reason="paywall: 'Subscribe'")

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is False  # md removed → nothing written
    assert not lib.has_extract(paper.key, "md")  # stub md deleted
    assert paper.md_path is None
    assert paper.md_engine == ""
    assert paper.download_status == "metadata_only"  # abstract present → citable


@responses.activate
def test_firecrawl_paywall_stub_no_abstract_rejected_to_failed(
    lib, paper, firecrawl_env, monkeypatch
):
    """D5 mirror: a gate-rejected firecrawl md with NO abstract demotes to the
    'failed' terminal (a true zero — no full text and nothing to cite)."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": PAYWALL_STUB_MD}},
        status=200,
    )
    paper.abstract = ""  # no metadata to fall back on
    _patch_gate(monkeypatch, complete=False, reason="paywall stub")

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is False
    assert not lib.has_extract(paper.key, "md")
    assert paper.download_status == "failed"


@responses.activate
def test_firecrawl_review_llm_error_does_not_block(
    lib, paper, firecrawl_env, monkeypatch
):
    """review_extract throws (LLM down) → don't fail download,
    Phase 12.2 worker self-check is final defense. (Gate passes the md first.)"""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": REAL_PAPER_MD}},
        status=200,
    )
    _patch_gate(monkeypatch, complete=True, reason="complete")

    def fake_review_raises(text, *, llm=None):
        raise RuntimeError("LLM endpoint unreachable")

    monkeypatch.setattr(
        "papervault.library.extract.review_extract", fake_review_raises)

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert paper.md_engine == "firecrawl"
    # insight_invalid_reason left None — let Phase 12.2 worker self-check decide
    assert paper.insight_invalid_reason is None


@responses.activate
def test_firecrawl_gate_error_is_fail_open(
    lib, paper, firecrawl_env, monkeypatch
):
    """If the gate call itself throws (LLM down), fail-OPEN: keep the firecrawl
    md serveable rather than discarding it over a flaky API (mirrors the
    completeness_gate fail-open policy)."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": REAL_PAPER_MD}},
        status=200,
    )

    def gate_raises(text, *, llm=None):
        raise RuntimeError("gate LLM unreachable")
    monkeypatch.setattr(
        "papervault.library.extract.completeness_gate", gate_raises)

    def fake_review(text, *, llm=None):
        return {"ok": True, "issues": [], "confidence": 0.95}
    monkeypatch.setattr(
        "papervault.library.extract.review_extract", fake_review)

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert lib.has_extract(paper.key, "md")  # kept despite gate error
    assert paper.download_status == "ok"


def _logged_events(lib):
    if not lib.manifest_path.exists():
        return []
    return [json.loads(line) for line in
            lib.manifest_path.read_text().splitlines() if line]


@responses.activate
def test_firecrawl_review_failed_logs_event(
    lib, paper, firecrawl_env, monkeypatch
):
    """A gate-PASSING md whose downstream pre-curator review_extract still
    flags it logs 'firecrawl_pre_insight_review_failed' (the two gates are
    independent: completeness_gate decides serveability, review_extract feeds
    the list_undistilled exclusion signal)."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": REAL_PAPER_MD}},
        status=200,
    )
    _patch_gate(monkeypatch, complete=True, reason="complete")

    def fake_review(text, *, llm=None):
        return {"ok": False, "issues": ["paywall"], "confidence": 0.9}

    monkeypatch.setattr(
        "papervault.library.extract.review_extract", fake_review)

    download._try_firecrawl_text_fallback(paper, lib)
    events = _logged_events(lib)
    assert any(e["event"] == "firecrawl_pre_insight_review_failed"
               for e in events), \
        f"Expected firecrawl_pre_insight_review_failed event, got: {events}"


@responses.activate
def test_firecrawl_gate_reject_logs_event(
    lib, paper, firecrawl_env, monkeypatch
):
    """D5: a gate-rejected firecrawl md logs 'firecrawl_gate_reject' with the
    demotion target — the audit trail for "firecrawl had no real full text"."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": PAYWALL_STUB_MD}},
        status=200,
    )
    _patch_gate(monkeypatch, complete=False, reason="paywall stub")

    download._try_firecrawl_text_fallback(paper, lib)
    events = _logged_events(lib)
    reject = [e for e in events if e["event"] == "firecrawl_gate_reject"]
    assert reject, f"Expected firecrawl_gate_reject event, got: {events}"
    assert reject[0]["demoted_to"] in ("metadata_only", "failed")


# ---------------------------------------------------------------------------
# S3 — historical-md re-entry (the ~48 firecrawl papers the migration adopted
# as ``ok`` resolve to full-text OR no-full-text when reconcile routes them
# back through download_paper). SDD §6.3 / §4.1 / D5: on re-entry the on-disk
# firecrawl md is re-gated; the YAML frontmatter is stripped first so the gate
# judges the body only; NO "text-only:firecrawl" limbo survives a re-gate.
# ---------------------------------------------------------------------------


def _write_firecrawl_md(lib, key, body, *, url="https://doi.org/10.1/x"):
    """Place a firecrawl-sourced md on disk in the exact production shape
    (``---`` YAML frontmatter + body), as a historical/migrated paper would
    have. No completeness gate ran when it was written (it predates the gate)."""
    frontmatter = (
        "---\n"
        "source: firecrawl\n"
        f"url: {url}\n"
        "fetched_at: 2024-01-01T00:00:00+08:00\n"
        "firecrawl_endpoint: /v1/scrape\n"
        "note: PDF binary unavailable; markdown is firecrawl rendering.\n"
        "---\n\n"
    )
    md_path = lib.md_path(key)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(frontmatter + body)
    return md_path


def test_strip_frontmatter_removes_yaml_block():
    """_strip_frontmatter drops a leading ---…--- block, keeps the body, and
    is a no-op when there is no frontmatter."""
    with_fm = "---\nsource: firecrawl\nurl: x\n---\n\nReal body here.\n"
    assert download._strip_frontmatter(with_fm) == "Real body here.\n"
    plain = "No frontmatter, just body.\n"
    assert download._strip_frontmatter(plain) == plain
    # malformed (no closing fence) → returned unchanged, never truncated
    half = "---\nsource: firecrawl\nbody but no close"
    assert download._strip_frontmatter(half) == half


def test_firecrawl_reentry_historical_md_passes_gate_stays_ok(
    lib, paper, monkeypatch
):
    """A historical firecrawl md already on disk that PASSES the re-gate stays
    serveable: status ok + source firecrawl, md untouched. This is a real
    full-text paper among the 48 — the gate confirms it and it resolves to
    'full-text'. No /v1/scrape call is made (md already present)."""
    _write_firecrawl_md(lib, paper.key, REAL_PAPER_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"
    _patch_gate(monkeypatch, complete=True, reason="complete")

    # No HTTP mock registered — re-entry must short-circuit before /v1/scrape.
    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert lib.has_extract(paper.key, "md")  # kept on disk
    assert paper.download_status == "ok"
    assert paper.download_source == "firecrawl"


def test_firecrawl_reentry_historical_stub_fails_gate_is_deleted_terminal(
    lib, paper, monkeypatch
):
    """A historical firecrawl md already on disk that FAILS the re-gate (paywall
    stub) is DELETED and demoted to a terminal status — NO 'text-only:firecrawl'
    limbo. Abstract present → metadata_only. This is how a junk paper among the
    48 resolves to 'no-full-text' (SDD §6.3 / D5)."""
    _write_firecrawl_md(lib, paper.key, PAYWALL_STUB_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"
    paper.abstract = "A real citable abstract."
    _patch_gate(monkeypatch, complete=False, reason="paywall stub")

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is False
    assert not lib.has_extract(paper.key, "md")  # historical stub deleted
    assert paper.md_path is None
    assert paper.md_engine == ""
    assert paper.download_status == "metadata_only"


def test_firecrawl_reentry_historical_stub_no_abstract_to_failed(
    lib, paper, monkeypatch
):
    """Re-entry mirror: a gate-rejected historical firecrawl md with NO abstract
    demotes to the 'failed' terminal (true zero — no full text, nothing to
    cite)."""
    _write_firecrawl_md(lib, paper.key, PAYWALL_STUB_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"
    paper.abstract = ""
    _patch_gate(monkeypatch, complete=False, reason="paywall stub")

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is False
    assert not lib.has_extract(paper.key, "md")
    assert paper.download_status == "failed"


def test_firecrawl_reentry_gates_body_not_frontmatter(lib, paper, monkeypatch):
    """The re-gate must judge the BODY only — the captured text handed to
    completeness_gate must NOT contain the YAML frontmatter (source/url/...)."""
    _write_firecrawl_md(lib, paper.key, REAL_PAPER_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"

    seen = {}

    def capture_gate(text, *, llm=None):
        seen["text"] = text
        return {"complete": True, "reason": "complete"}
    monkeypatch.setattr(
        "papervault.library.extract.completeness_gate", capture_gate)

    download._try_firecrawl_text_fallback(paper, lib)
    assert "source: firecrawl" not in seen["text"]
    assert "firecrawl_endpoint" not in seen["text"]
    assert seen["text"].startswith("# Introduction")  # body, frontmatter gone


def test_firecrawl_reentry_gate_failopen_keeps_md(lib, paper, monkeypatch):
    """Re-entry fail-open: if the gate call throws (LLM down), keep the
    historical md serveable rather than discarding it over a flaky API."""
    _write_firecrawl_md(lib, paper.key, REAL_PAPER_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"

    def gate_raises(text, *, llm=None):
        raise RuntimeError("gate LLM unreachable")
    monkeypatch.setattr(
        "papervault.library.extract.completeness_gate", gate_raises)

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert lib.has_extract(paper.key, "md")  # kept despite gate error
    assert paper.download_status == "ok"


def test_firecrawl_reentry_nonfirecrawl_md_left_alone(lib, paper, monkeypatch):
    """An md on disk from a higher-fidelity engine (marker/dots) is NOT re-gated
    or touched on re-entry — only firecrawl-sourced md is subject to the D5
    re-gate (a real OCR extract already passed the gate at write time)."""
    md_path = lib.md_path(paper.key)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# Real OCR output\n\nThis md came from marker.\n")

    def gate_must_not_run(text, *, llm=None):
        raise AssertionError("completeness_gate must not run on marker md")
    monkeypatch.setattr(
        "papervault.library.extract.completeness_gate", gate_must_not_run)

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is False  # not firecrawl-sourced → leave alone
    assert lib.has_extract(paper.key, "md")  # untouched


# ---------------------------------------------------------------------------
# S3 — terminal resting state (issue #1/#2): a re-gate-PASS firecrawl paper must
# get ``firecrawl_pdf_hunt_exhausted`` so classify stops re-routing it to
# DOWNLOAD every reconcile sweep (the permanent hot-loop) and the re-gate runs
# AT MOST ONCE (a single spurious incomplete verdict can never accumulate into a
# guaranteed deletion of a genuinely-good historical md).
# ---------------------------------------------------------------------------


def test_firecrawl_reentry_pass_stamps_pdf_hunt_exhausted(lib, paper, monkeypatch):
    """On a re-entry gate PASS the paper is marked ``firecrawl_pdf_hunt_exhausted``
    — all 18 PDF tiers had already missed, so the real-PDF hunt is done; classify
    will now rest the paper instead of re-downloading + re-gating it forever."""
    _write_firecrawl_md(lib, paper.key, REAL_PAPER_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"
    assert paper.firecrawl_pdf_hunt_exhausted is False
    _patch_gate(monkeypatch, complete=True, reason="complete")

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert paper.download_status == "ok"
    assert paper.firecrawl_pdf_hunt_exhausted is True  # one-shot rest stamped


@responses.activate
def test_firecrawl_fresh_pass_stamps_pdf_hunt_exhausted(
    lib, paper, firecrawl_env, monkeypatch
):
    """A freshly-scraped firecrawl md that PASSes the gate is ALSO stamped — the
    18-tier cascade just missed, so re-hunting it every 600s is the same hot-loop
    we kill for the historical papers."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": REAL_PAPER_MD}},
        status=200,
    )
    _patch_gate(monkeypatch, complete=True, reason="complete")

    def fake_review(text, *, llm=None):
        return {"ok": True, "issues": [], "confidence": 0.95}
    monkeypatch.setattr("papervault.library.extract.review_extract", fake_review)

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert paper.firecrawl_pdf_hunt_exhausted is True


def test_firecrawl_reentry_failopen_stamps_exhausted_no_reloop(
    lib, paper, monkeypatch
):
    """Fail-open (gate LLM down) keeps the md AND stamps exhausted — otherwise a
    persistently-down gate would re-hunt + re-gate the paper every sweep forever,
    the exact hot-loop. Fail-open treats it as a (provisional) PASS, so it rests
    like any other PASS — PERMANENTLY: the exhausted bit is never re-armed (D5,
    "firecrawl tried and failed = failed"; no audit command touches it; only an
    explicit --force-refresh re-add clears it, a manual library edit)."""
    _write_firecrawl_md(lib, paper.key, REAL_PAPER_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"

    def gate_raises(text, *, llm=None):
        raise RuntimeError("gate LLM unreachable")
    monkeypatch.setattr(
        "papervault.library.extract.completeness_gate", gate_raises)

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert lib.has_extract(paper.key, "md")  # kept (fail-open)
    assert paper.firecrawl_pdf_hunt_exhausted is True  # rests, no re-loop


# ---------------------------------------------------------------------------
# S3 — issue #3: a non-firecrawl (marker/dots) md re-entry must NOT let
# download_paper's no-fulltext fallthrough clobber a status=ok paper that HAS
# full text on disk to a terminal status. The status field must keep telling the
# truth (the paper has full text). NOTE (G2 fix): classify now RESTS such a paper
# at TERMINAL (md_source != firecrawl → no DOWNLOAD re-route), so this branch is
# unreachable in steady state; the re-assert is now a purely DEFENSIVE guard
# (if the paper is ever explicitly enqueued, its ok status survives), not a
# hot-loop step. This test pins that defensive no-clobber invariant.
# ---------------------------------------------------------------------------


def test_download_paper_nonfirecrawl_md_reentry_keeps_status_ok(
    lib, paper, firecrawl_env, monkeypatch
):
    """A status=ok paper with a valid marker/dots md but no PDF, fed directly into
    download_paper and missing all 18 tiers, must come out STILL ``ok`` — not
    clobbered to metadata_only/failed. (issue #3 / §4.3: status must match the
    served md. Post-G2, classify rests this paper at TERMINAL rather than routing
    it to DOWNLOAD, so the re-assert is a defensive no-clobber guard, not a step
    in an active PDF hunt.)"""
    # Marker md on disk (no firecrawl frontmatter), no PDF.
    md_path = lib.md_path(paper.key)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# Real OCR output\n\nThis md came from marker.\n")
    paper.download_status = "ok"
    paper.download_source = "marker"
    paper.abstract = "A citable abstract that would tempt a metadata_only clobber."

    # All 18 PDF tiers miss → reach the firecrawl fallthrough.
    monkeypatch.setattr(download, "_STRATEGIES", [])

    result = download.download_paper(paper, lib)
    assert result is False  # no PDF binary obtained
    # The load-bearing assertion: status NOT clobbered to a terminal.
    assert paper.download_status == "ok"
    assert lib.has_extract(paper.key, "md")  # md still on disk, still served


# ---------------------------------------------------------------------------
# issue #1 (med) — the §4.3 "md on disk ⟺ gated" invariant for HISTORICAL
# un-gated firecrawl md. The 48 migrated firecrawl md predate the gate; the
# design re-gates them via the firecrawl re-entry, but that re-entry is only
# reached when all 18 tiers MISS. If a tier lands a real PDF FIRST, the re-gate
# is skipped → disk has PDF + un-gated md → classify rule 2 TERMINAL → serve
# hands out a never-gated stub as full text forever. Fix (Option A): re-gate the
# historical md BEFORE the tier loop, so a later tier-hit only ever co-exists
# with an ALREADY-gated firecrawl md.
# ---------------------------------------------------------------------------


def test_historical_md_regated_before_tier_pdf_pass_keeps(
    lib, paper, firecrawl_env, monkeypatch
):
    """Historical un-gated firecrawl md + a tier that lands a real PDF: the md is
    re-gated (PASS) BEFORE the PDF is fetched, so it ends up gated + stamped
    ``firecrawl_pdf_hunt_exhausted`` and co-exists with the PDF as an ALREADY
    gated md. The invariant "md on disk ⟺ gated" holds before rule-2 TERMINAL."""
    _write_firecrawl_md(lib, paper.key, REAL_PAPER_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"
    assert paper.firecrawl_pdf_hunt_exhausted is False

    gate_calls = {"n": 0}

    def fake_gate(text, *, llm=None):
        gate_calls["n"] += 1
        # The gate must see the historical BODY (frontmatter stripped).
        assert "source: firecrawl" not in text
        return {"complete": True, "reason": "complete"}
    monkeypatch.setattr("papervault.library.extract.completeness_gate", fake_gate)

    # A single tier that "lands a real PDF"; verification stubbed to accept.
    def landing_tier(p):
        return b"%PDF-1.5 fake pdf bytes"
    monkeypatch.setattr(download, "_STRATEGIES", [("test_tier", landing_tier)])
    monkeypatch.setattr(
        download, "_verify_pdf_matches_metadata", lambda data, p: (True, "ok"))

    result = download.download_paper(paper, lib)
    assert result is True  # a PDF was obtained
    assert lib.has_pdf(paper.key)
    assert lib.has_extract(paper.key, "md")  # md kept
    # The md was gated BEFORE the tier hit (gate ran on the historical body),
    # and the resting stamp is set → it is NOT a never-gated stub.
    assert gate_calls["n"] >= 1
    assert paper.firecrawl_pdf_hunt_exhausted is True

    # classify now hits rule 2 (has_pdf ∧ has_md) → TERMINAL, serving the
    # ALREADY-gated md. Confirm the invariant holds at the serve chokepoint.
    from papervault.library.services import classify as _classify
    assert _classify.classify(paper, lib) == _classify.TERMINAL


def test_historical_md_regated_before_tier_pdf_fail_deletes(
    lib, paper, firecrawl_env, monkeypatch
):
    """Historical un-gated firecrawl STUB + a tier that would land a real PDF:
    the stub is re-gated (FAIL) BEFORE the tier loop, so it is DELETED and the
    paper demoted to terminal — and download SHORT-CIRCUITS (it does NOT hunt a
    PDF under a terminal status). No un-gated stub is ever served."""
    _write_firecrawl_md(lib, paper.key, PAYWALL_STUB_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"
    paper.abstract = "A real citable abstract."
    _patch_gate(monkeypatch, complete=False, reason="paywall stub")

    tier_called = {"n": 0}

    def landing_tier(p):
        tier_called["n"] += 1
        return b"%PDF-1.5 fake pdf bytes"
    monkeypatch.setattr(download, "_STRATEGIES", [("test_tier", landing_tier)])
    monkeypatch.setattr(
        download, "_verify_pdf_matches_metadata", lambda data, p: (True, "ok"))

    result = download.download_paper(paper, lib)
    assert result is False  # short-circuited before the tier loop
    assert tier_called["n"] == 0  # never hunted a PDF under a terminal status
    assert not lib.has_extract(paper.key, "md")  # stub deleted, never served
    assert not lib.has_pdf(paper.key)
    assert paper.download_status == "metadata_only"


def test_gate_reject_unlink_failure_neutralises_md_not_served(
    lib, paper, monkeypatch
):
    """issue #1 manifestation 2: if _gate_firecrawl_md FAILs and the md unlink
    raises OSError, the rejected stub must NOT remain serveable. It is truncated
    to empty (size 0 → has_extract False), so serve-safety hands out NO
    text_path. fail-closed."""
    md_path = _write_firecrawl_md(lib, paper.key, PAYWALL_STUB_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"
    paper.abstract = "A citable abstract."
    _patch_gate(monkeypatch, complete=False, reason="paywall stub")

    # Make unlink fail (simulate a transient OSError); truncate fallback fires.
    import pathlib
    real_unlink = pathlib.Path.unlink

    def flaky_unlink(self, *a, **k):
        if self == md_path:
            raise OSError("simulated unlink failure")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(pathlib.Path, "unlink", flaky_unlink)

    body = download._strip_frontmatter(md_path.read_text())
    result = download._gate_firecrawl_md(paper, lib, body)
    assert result is False
    assert paper.download_status == "metadata_only"
    # The file may still exist on disk, but it is now EMPTY → has_extract False
    # → serve-safety treats the md as absent (fail-closed). Either deleted or
    # zero-size both satisfy the invariant.
    assert not lib.has_extract(paper.key, "md")

    # And the serve chokepoint hands out NO text_path for this terminal paper.
    from papervault.library.mcp.server import _attach_text_reference
    rec: dict = {}
    _attach_text_reference(rec, paper, lib)
    assert "text_path" not in rec
    assert rec.get("text_status") == "metadata_only"


def test_historical_md_regated_then_tiers_miss_no_double_gate(
    lib, paper, firecrawl_env, monkeypatch
):
    """Historical un-gated firecrawl md PASSes the early re-gate, then all 18
    tiers MISS so control reaches the firecrawl re-entry. The re-entry must NOT
    re-gate (the stamp is already set) — a second gate call on a borderline body
    could spuriously DELETE a previously-PASSed md (the accumulated-deletion
    hazard). So the gate runs EXACTLY ONCE across the whole download_paper call."""
    _write_firecrawl_md(lib, paper.key, REAL_PAPER_MD)
    paper.download_status = "ok"
    paper.download_source = "firecrawl"

    gate_calls = {"n": 0}

    def fake_gate(text, *, llm=None):
        gate_calls["n"] += 1
        return {"complete": True, "reason": "complete"}
    monkeypatch.setattr("papervault.library.extract.completeness_gate", fake_gate)

    # All tiers miss (no PDF anywhere).
    monkeypatch.setattr(download, "_STRATEGIES", [])

    result = download.download_paper(paper, lib)
    assert result is False  # no PDF binary; firecrawl md is the full text
    assert gate_calls["n"] == 1  # gated EXACTLY once — no re-entry double-gate
    assert lib.has_extract(paper.key, "md")  # kept
    assert paper.download_status == "ok"
    assert paper.firecrawl_pdf_hunt_exhausted is True


# ── identity gate (2026-06-03): firecrawl body must be topically consistent
#    with the paper's own abstract; a complete-but-WRONG article is rejected ──

_CR_ABSTRACT = (
    "We present a neural network approach to galactic cosmic ray transport "
    "modeling in the inner heliosphere, training on solar energetic particle "
    "events from Parker Solar Probe to predict modulation coefficients with "
    "high accuracy across the solar activity cycle."
)

# A COMPLETE, readable article body — but a different paper (molecular biology),
# sharing essentially none of the abstract's content words.
_WRONG_ARTICLE_MD = (
    "# Introduction\n\nThe HBP1 transcription factor acts as a tumor suppressor "
    "through epigenetic regulation of chromatin methylation in neuroblastoma "
    "cells. We investigate histone deacetylase recruitment and promoter "
    "silencing of downstream oncogenes during cellular differentiation.\n\n"
    "## Methods\n\nChromatin immunoprecipitation sequencing was performed on "
    "cultured neuroblastoma lineages. Methylation arrays quantified promoter "
    "hypermethylation; western blotting measured protein abundance across "
    "passages. Apoptosis assays used flow cytometry with annexin staining.\n\n"
    "## Results\n\nKnockdown of the suppressor increased proliferation and "
    "decreased differentiation markers. Promoter hypermethylation correlated "
    "with transcriptional silencing. Restoring expression induced apoptosis in "
    "the malignant lineages and reduced tumor xenograft volume in mice.\n\n"
) * 2


@responses.activate
def test_firecrawl_wrong_article_rejected_by_identity(
    lib, paper, firecrawl_env, monkeypatch
):
    """A COMPLETE but WRONG article (renders fine, would pass completeness) whose
    body is topically disjoint from the paper's abstract → identity gate rejects
    it; nothing is served as full text."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": _WRONG_ARTICLE_MD}},
        status=200,
    )
    paper.abstract = _CR_ABSTRACT
    # completeness/review would PASS if reached — proving the IDENTITY gate (run
    # first) is what rejects.
    _patch_gate(monkeypatch, complete=True, reason="complete")
    monkeypatch.setattr("papervault.library.extract.review_extract",
                        lambda text, *, llm=None: {"ok": True, "issues": [], "confidence": 0.9})

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is False
    assert not lib.has_extract(paper.key, "md"), "wrong article must NOT be served as full text"


@responses.activate
def test_firecrawl_matching_abstract_passes_identity(
    lib, paper, firecrawl_env, monkeypatch
):
    """Body topically consistent with the abstract → identity passes → the md is
    accepted (the existing completeness/review path runs)."""
    responses.add(
        responses.POST, "https://api.firecrawl.dev/v1/scrape",
        json={"success": True, "data": {"markdown": REAL_PAPER_MD}},
        status=200,
    )
    paper.abstract = _CR_ABSTRACT
    _patch_gate(monkeypatch, complete=True, reason="complete")
    monkeypatch.setattr("papervault.library.extract.review_extract",
                        lambda text, *, llm=None: {"ok": True, "issues": [], "confidence": 0.95})

    result = download._try_firecrawl_text_fallback(paper, lib)
    assert result is True
    assert lib.has_extract(paper.key, "md")
