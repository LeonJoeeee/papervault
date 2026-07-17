"""V7 — KS-side _strip_acks / clean() unit tests (SDD §6.9.7, §11 F22).

Pure-logic tests: _strip_acks and clean() are string→string, never open a DB pool,
so they run with no env (same class as test_fingerprint pure paths).

Covers the SDD's four conservative invariants:
  ① only a TRAILING (>50%) markdown heading is a cut point;
  ② markdown-heading based — an in-body 'funding'/'acknowledge' keyword is NOT a cut;
  ③ char-count guard (_MIN_KEEP_RATIO=0.5) aborts an over-aggressive strip;
  ④ a doc with none of these sections is an exact no-op.
Plus: clean() composes _strip_acks ∘ _strip_references, and pl's _strip_references
is left untouched (we only import it).
"""
from __future__ import annotations

from papervault.knowledge.ingest.distill import _MIN_KEEP_RATIO, _strip_acks, clean

# A realistic body long enough that a trailing acks section is well past the 50% mark
# and removing it stays above the _MIN_KEEP_RATIO guard.
_BODY = (
    "# Introduction\n\n"
    "Magnetic reconnection in the magnetotail drives substorm onset. "
    "We analyze MMS multi-spacecraft data across the plasma sheet boundary layer. "
    "The work was funded in part by ongoing campaigns, as discussed inline below.\n\n"
    "# Methods\n\n"
    "We compute the reconnection rate from the normalized inflow velocity. "
    "Electron-scale current sheets are resolved at sub-ion-inertial-length scales. "
    "Author contributions to the dataset pipeline are described in the methods text itself.\n\n"
    "# Results\n\n"
    "We find a reconnection rate of approximately 0.1, consistent with prior estimates. "
    "The diffusion region exhibits crescent-shaped electron distributions.\n\n"
    "# Discussion\n\n"
    "These results constrain the onset mechanism and the energy-conversion budget. "
    "Future missions will extend the statistics to the dayside magnetopause.\n"
)


def test_strip_acks_cuts_trailing_acknowledgements_and_preserves_body():
    """Doc WITH an Acknowledgements section → cut at the heading, body preserved verbatim."""
    acks = (
        "\n# Acknowledgements\n\n"
        "We thank J. Smith and the XYZ Foundation. This work was supported by "
        "grant NNX17AB12G and by the National Science Foundation award 1234567. "
        "Contact: author@example.edu.\n"
    )
    doc = _BODY + acks
    out = _strip_acks(doc)
    assert out == _BODY.rstrip()  # body preserved exactly, trailing section gone
    assert "Acknowledgements" not in out
    assert "XYZ Foundation" not in out
    assert "NNX17AB12G" not in out
    # Body landmarks all survive
    assert "Magnetic reconnection" in out
    assert "reconnection rate of approximately 0.1" in out


def test_strip_acks_cuts_consecutive_ack_sections_but_keeps_references():
    """Funding + Data Availability (consecutive ack-class) → BOTH stripped via recursion.
    References is NOT an ack-class heading → _strip_acks leaves it (pl's _strip_references
    removes refs; _strip_acks must only touch ack-class sections, SDD §6.9.7 narrow-cut)."""
    tail = (
        "\n# Funding\n\nSupported by grant ABC-123.\n\n"
        "# Data Availability\n\nData are available on request.\n\n"
        "# References\n\n[1] Someone et al., 2020.\n"
    )
    doc = _BODY + tail
    out = _strip_acks(doc)
    assert "Funding" not in out
    assert "Data Availability" not in out
    # References preserved by _strip_acks (clean() strips it via _strip_references, not here)
    assert "# References" in out
    assert "Magnetic reconnection" in out
    assert "reconnection rate of approximately 0.1" in out


def test_strip_acks_various_headings():
    for heading in (
        "# Acknowledgments",  # US spelling
        "# Acknowledgement",  # singular
        "## Author Contributions",
        "### Conflicts of Interest",
        "# Competing Interests",
        "## Declaration of Competing Interest",
        "# Data Availability Statement",
        "#### Funding Information",
    ):
        doc = _BODY + "\n" + heading + "\n\nSome trailing boilerplate to remove.\n"
        out = _strip_acks(doc)
        assert out == _BODY.rstrip(), f"heading not stripped: {heading!r}"


def test_strip_acks_no_section_is_noop():
    """Doc WITHOUT any acks-class section → returned unchanged (④ no-op)."""
    doc = _BODY
    assert _strip_acks(doc) == doc


def test_strip_acks_ignores_inbody_keyword_not_a_heading():
    """② 'funded by' / 'Author contributions' appear in the BODY prose (not as headings)
    → must NOT be treated as a cut point. The whole body is kept."""
    doc = _BODY  # body already contains 'was funded', 'Author contributions to the dataset'
    out = _strip_acks(doc)
    assert out == doc
    assert "was funded in part" in out
    assert "Author contributions to the dataset pipeline" in out


def test_strip_acks_strips_early_bounded_ack_but_keeps_body():
    """★ Relaxed position gate (drill 2026-06-02b, SDD §6.9.7 / §11 F22 b): an early ack-class
    heading that is BOUNDED by a following heading (a narrow, single-section cut) IS honored at
    any position — this is the appendix-heavy fix (Rathore2024 @35% / Chen2020 @44% had genuine
    ack/data-availability sections before the 50% mark that the old >50% gate leaked). Only the
    ack section is removed; everything after the next heading survives."""
    early = "# Funding\n\nThis grant note appears early; appendices push it before 50%.\n\n"
    doc = early + _BODY + _BODY  # the early Funding heading sits in the first half, bounded by Introduction
    out = _strip_acks(doc)
    # The bounded early Funding section is narrowly cut...
    assert "This grant note appears early" not in out
    assert out.count("# Funding") == 0
    # ...but the body that follows (everything from the next heading on) survives in full.
    assert "Magnetic reconnection" in out
    assert "reconnection rate of approximately 0.1" in out


def test_strip_acks_unbounded_early_heading_is_not_cut_to_eof():
    """① An ack heading in the FIRST half with NO following heading (an unbounded cut-to-EOF)
    is NOT honored — an early unbounded cut would nuke the whole paper. Defensive floor that
    the relaxed gate keeps only for the bounded narrow-cut case."""
    doc = _BODY + _BODY + "\n# Funding\n\nA trailing grant note with no following heading.\n"
    # Move the Funding heading into the first half by appending more body after it WITHOUT a heading:
    doc = "# Funding\n\nEarly unbounded grant note.\n\n" + _BODY.replace("#", "") + _BODY.replace("#", "")
    out = _strip_acks(doc)
    # No following heading + first-half position → not cut (gate ① rejects unbounded early cut).
    assert out == doc
    assert "Early unbounded grant note" in out


def test_strip_acks_charcount_guard_aborts_overcut():
    """③ If stripping would drop more than (1 - _MIN_KEEP_RATIO) of the doc, abort and
    return the original (guards against mis-classifying body as a trailing section)."""
    short_body = "# Title\n\nOne short sentence of actual content.\n"
    big_acks = "\n# Acknowledgements\n\n" + ("thanks " * 500)
    doc = short_body + big_acks
    # The heading IS in the trailing half, but the kept part is far below half the doc.
    kept = short_body.rstrip()
    assert len(kept) < len(doc) * _MIN_KEEP_RATIO  # precondition: guard should trip
    out = _strip_acks(doc)
    assert out == doc  # aborted → original returned, no truncation


def test_strip_acks_empty_and_none_safe():
    assert _strip_acks("") == ""


def test_clean_composes_refs_then_acks():
    """clean() = _strip_acks(_strip_references(text)) — both tails gone, body intact."""
    refs = "\n# References\n\n[1] A. Author, J. Foo, 2019.\n[2] B. Author, 2021.\n"
    acks = "\n# Acknowledgements\n\nThanks to the team. Grant XR-9.\n"
    # Acks AFTER references is the common layout; clean must remove from the references
    # heading onward (refs strip handles refs+everything-after; acks strip is then a no-op
    # on the already-trimmed text). Either order of removal yields the body.
    doc = _BODY + refs + acks
    out = clean(doc)
    assert "References" not in out
    assert "Acknowledgements" not in out
    assert "Grant XR-9" not in out
    assert "Magnetic reconnection" in out
    assert "reconnection rate of approximately 0.1" in out


def test_clean_acks_before_references_layout():
    """Some papers put Acknowledgements before References. clean() must strip both."""
    acks = "\n# Acknowledgements\n\nThanks to the team.\n"
    refs = "\n# References\n\n[1] A. Author, 2019.\n"
    doc = _BODY + acks + refs
    out = clean(doc)
    assert "Acknowledgements" not in out
    assert "References" not in out
    assert "Magnetic reconnection" in out


def test_clean_noop_when_no_tails():
    """clean() on a doc with neither references nor acks → unchanged."""
    assert clean(_BODY) == _BODY


# ---- drill fixes 2026-06-02 (SDD §6.9.7 / §11 F22) --------------------------

def test_strip_acks_preserves_appendix_after_acknowledgements():
    """★ Narrow cut: an Appendix that sits AFTER Acknowledgements must survive. Old cut-to-EOF
    swept it away (18% of test100 cut papers lost appendix body, e.g. Abdollahi2017 -33%)."""
    appendix = (
        "\n# Appendix A: Energy Reconstruction\n\n"
        "We reconstruct the absolute energy scale from the calorimeter response. "
        "Systematic uncertainties are propagated through the full detector simulation. "
        "Table A1 lists the per-bin energy resolution across the fiducial volume.\n\n"
        "# Appendix B: Data Tables\n\n"
        "The binned flux measurements and their covariance matrix are tabulated here.\n"
    )
    acks = "\n# Acknowledgements\n\nWe thank the XYZ collaboration. Grant NNX17AB12G.\n"
    doc = _BODY + acks + appendix  # acks BEFORE the appendices (common journal layout)
    out = _strip_acks(doc)
    # ack section gone...
    assert "Acknowledgements" not in out
    assert "NNX17AB12G" not in out
    # ...but BOTH appendices (substantive body) survive
    assert "Appendix A: Energy Reconstruction" in out
    assert "absolute energy scale" in out
    assert "Appendix B: Data Tables" in out
    assert "covariance matrix" in out


def test_strip_acks_numbered_and_roman_prefixed_headings():
    """★ Headings like '## 5. Acknowledgments', '## VII. ACKNOWLEDGMENTS', '## 6 Acknowledgment'
    (section number / roman numeral before the keyword) must strip. Old regex skipped them →
    junk entities leaked into the graph (Dembinski2017/Bartoli2015/Yan2024/Bindi2017)."""
    for heading in (
        "## 5. Acknowledgments",
        "## VII. ACKNOWLEDGMENTS",
        "## 6 Acknowledgment",
        "### 7) Funding",
        "## IV. Data Availability",
    ):
        doc = _BODY + "\n" + heading + "\n\nTrailing boilerplate, grant 1234, author@x.edu.\n"
        out = _strip_acks(doc)
        assert out == _BODY.rstrip(), f"numbered/roman heading not stripped: {heading!r}"


def test_strip_acks_numbered_heading_keeps_following_appendix():
    """Combine both fixes: a numbered ack heading, with an Appendix after it, strips the ack
    and keeps the appendix."""
    doc = (
        _BODY
        + "\n## 5. Acknowledgments\n\nThanks to collaborators. Grant XR-9.\n\n"
        + "## 6. Appendix\n\nDerivation of the antiproton-to-proton ratio is given here.\n"
    )
    out = _strip_acks(doc)
    assert "Acknowledgments" not in out
    assert "Grant XR-9" not in out
    assert "Appendix" in out
    assert "antiproton-to-proton ratio" in out


# ---- drill fixes 2026-06-02b: body-marker gate (SDD §6.9.7 / §11 F22 a/a2) -----------------

def test_strip_acks_eof_cut_keeps_floating_table_after_acknowledgements():
    """★ Adriani2009 shape: the ack heading is the LAST heading, but OCR dropped a headline
    data TABLE + figure caption AFTER it with no intervening heading. Old cut-to-EOF deleted
    them (-24%, the paper's headline measurement). Body-marker gate keeps the whole segment."""
    tail = (
        "\n## ACKNOWLEDGMENTS\n\n"
        "We thank D. Marinucci and the U. Chicago group. Grant NNX17AB12G, contact a@b.edu.\n\n"
        "TABLE I: Summary of positron fraction results.\n"
        "<table><tr><td>0.05</td><td>0.012</td></tr></table>\n\n"
        "FIG. 5: Positron event display.\n"
    )
    doc = _BODY + tail  # NO heading after ACKNOWLEDGMENTS → would be a cut-to-EOF
    out = _strip_acks(doc)
    # The floating table + figure (genuine results body) are protected — segment kept intact.
    assert "TABLE I: Summary of positron fraction results" in out
    assert "<table>" in out
    assert "FIG. 5: Positron event display" in out
    # Body untouched.
    assert "reconnection rate of approximately 0.1" in out


def test_strip_acks_narrow_cut_keeps_floating_table_between_ack_and_appendix():
    """★ Cholis2020 shape: a fit-results TABLE sits between '## ACKNOWLEDGMENTS' and
    '## Appendix A' with no heading of its own → the narrow cut would sweep it. Body-marker
    gate keeps the segment (so neither the table nor the appendix is lost)."""
    middle = (
        "\n## ACKNOWLEDGMENTS\n\n"
        "IC acknowledges support from NASA Grant No. NNX15AJ20H. DH supported by DE-AC02.\n\n"
        "TABLE II. Constraints on phi_0 / phi_1 / Delta-chi^2 across 19 averaging schemes.\n"
        "FIG. 7. The 2-sigma confidence contours.\n"
    )
    appendix = "\n## Appendix A\n\nDerivation of the propagation kernel is given here.\n"
    doc = _BODY + middle + appendix
    out = _strip_acks(doc)
    # The floating fit-results table + figure survive (not pure ack → segment kept)...
    assert "TABLE II. Constraints on phi_0" in out
    assert "FIG. 7. The 2-sigma confidence contours" in out
    # ...and the appendix survives too.
    assert "Appendix A" in out
    assert "propagation kernel" in out


def test_strip_acks_picks_later_clean_ack_when_earlier_has_body_marker():
    """If an earlier ack segment holds a body marker (kept), a LATER pure-ack section is still
    stripped — the gate skips the marked segment and continues, it does not give up."""
    doc = (
        _BODY
        + "\n## Acknowledgements\n\nThanks. Grant A-1.\n\n"
        + "TABLE I: results.\n<table><tr><td>1</td><td>2</td></tr></table>\n\n"
        + "## Appendix A\n\nReal appendix body about energy reconstruction.\n\n"
        + "## Funding\n\nSupported by grant B-2 only; no tables here.\n"
    )
    out = _strip_acks(doc)
    # earlier ack kept (has the table)...
    assert "Grant A-1" in out
    assert "TABLE I: results" in out
    assert "Appendix A" in out
    # ...but the later, clean Funding section (no body markers) IS stripped.
    assert "grant B-2" not in out
    assert "## Funding" not in out


def test_strip_acks_pure_ack_with_inprose_table_mention_is_still_cut():
    """The body-marker gate is line-anchored: an in-prose mention like 'see Table 2' inside an
    ack paragraph is NOT a body marker, so a genuinely pure ack section is still stripped."""
    doc = _BODY + "\n# Acknowledgements\n\nWe thank the team; data shown earlier (see Table 2).\n"
    out = _strip_acks(doc)
    assert out == _BODY.rstrip()
    assert "Acknowledgements" not in out


# ---- drill fix 2026-06-02c: declaration-class heading variants (SDD §6.9.7 / §11 F22 b3) ----

def test_strip_acks_declaration_class_heading_variants():
    """★ Two real journal styles leaked through the old fixed keyword list:
    Liu2021 '## Compliance with ethical standard Conflict' (Springer; bare 'Conflict' under a
    'Compliance...' heading) and Liu2024a '## Disclosure statement'. Plus 'Declaration of
    interests'. All sit @ trailing position with one-line boilerplate; all must now strip."""
    for heading in (
        "## Disclosure statement",                      # Liu2024a
        "## Compliance with ethical standard Conflict",  # Liu2021 Springer (bare 'Compliance...')
        "## Compliance with Ethical Standards",
        "# Disclosure",
        "## Declaration of Interests",
        "### Declaration of Competing Interests",
        "## VII. Disclosure statement",                 # numbered prefix still works
    ):
        doc = _BODY + "\n" + heading + "\n\nThe authors have no conflicts of interest to declare.\n"
        out = _strip_acks(doc)
        assert out == _BODY.rstrip(), f"declaration heading not stripped: {heading!r}"


def test_strip_acks_liu2021_compliance_and_liu2024a_disclosure_leak_closed():
    """Reproduces the two leaking shapes end-to-end (trailing @ ~99%, one-line boilerplate)."""
    liu2021 = _BODY + (
        "\n## Compliance with ethical standard Conflict\n\n"
        "The authors have no conflicts of interest to declare that are relevant to this work.\n"
    )
    liu2024a = _BODY + (
        "\n## Disclosure statement\n\n"
        "No potential conflict of interest was reported by the authors.\n"
    )
    assert _strip_acks(liu2021) == _BODY.rstrip()
    assert "conflicts of interest" not in _strip_acks(liu2021)
    assert _strip_acks(liu2024a) == _BODY.rstrip()
    assert "No potential conflict of interest" not in _strip_acks(liu2024a)


# ---- char-guard frame semantics: benign false-keep (SDD §6.9.7 ④ / §11 F22) ----------------

def test_strip_acks_charguard_topframe_is_final_vs_original():
    """The guard is evaluated per recursion frame, but recursion returns BEFORE the guard, so
    the OUTERMOST frame compares final-vs-original. Consecutive ack sections that together
    exceed half a SHORT doc are still removed as long as each single-section cut stays above
    the per-frame ratio (the real-corpus path; 0 guard trips in test100)."""
    body = (
        "# Introduction\n\nWe study antiproton propagation in the heliosphere over many lines "
        "of genuine scientific body text that comfortably dominate the document length so that "
        "removing the short trailing declarations never approaches the half-document floor.\n\n"
        "# Results\n\nThe measured ratio is consistent with secondary production models.\n"
    )
    tail = (
        "\n# Funding\n\nGrant A-1.\n\n"
        "# Disclosure\n\nNo competing interests.\n\n"
        "# Data Availability\n\nData on request.\n"
    )
    out = _strip_acks(body + tail)
    assert out == body.rstrip()  # all three trailing declaration sections gone
    assert "Funding" not in out and "Disclosure" not in out and "Data Availability" not in out


def test_strip_acks_ackdominant_short_doc_benign_false_keep():
    """★ Documented benign false-keep (SDD §6.9.7 ④): an extremely short, ack-DOMINANT doc where
    the acknowledgements is >50% of the text trips the char-guard → the cut is abandoned and the
    ack is RETAINED (conservative direction: keep a little noise rather than risk over-cut). Real
    papers are never 50%+ acknowledgements, so this never fires on the corpus — locked here so a
    future refactor that 'fixes' it does so deliberately, doc-first."""
    short_body = "# Results\n\nThe ratio is 0.1.\n"
    big_ack = "\n# Acknowledgements\n\n" + ("We gratefully thank many collaborators and funders. " * 30)
    doc = short_body + big_ack
    # precondition: the ack dominates → a clean cut would drop more than half the doc
    assert len(short_body.rstrip()) < len(doc) * _MIN_KEEP_RATIO
    out = _strip_acks(doc)
    assert out == doc  # false-keep: ack retained (benign, conservative-direction)
