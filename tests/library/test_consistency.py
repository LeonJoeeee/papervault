"""Tests for the metadata<->abstract<->extract consistency predicate."""
from __future__ import annotations

from papervault.library import consistency as c


# ----------------------------- abstract ----------------------------------

def test_abstract_presence():
    assert not c.has_usable_abstract(None)
    assert not c.has_usable_abstract("")
    assert not c.has_usable_abstract("   ")
    assert not c.has_usable_abstract("too short")            # < 40 chars
    assert c.has_usable_abstract("x" * 40)
    assert c.has_usable_abstract(
        "We present a study of cosmic-ray modulation in the heliosphere.")


# --------------------------- extract status -------------------------------

GIACALONE_HEAD = (
    "# COSMIC-RAY TRANSPORT COEFFICIENTS\n\nJOE GIACALONE\n\n*The University of "
    "Arizona*\n\n**Abstract.** A review of cosmic-ray transport coefficients ...")


def test_extract_clean_match():
    """Title topic present AND first author present -> MATCH."""
    status, _ = c.extract_status(
        "Cosmic-Ray Transport Coefficients", ["Joe Giacalone"], GIACALONE_HEAD)
    assert status == c.MATCH


def test_extract_multi_article_scan_still_matches():
    """A multi-article journal scan: the metadata paper IS in the file (title
    tokens + author present) even though other text precedes it -> MATCH."""
    head = (
        "If the level had negative parity it would be formed by p capture ...\n"
        "Cosmic-Ray Modulation by Solar Wind\nE. N. Parker\nIt is shown that the "
        "hydrodynamic outflow of gas from the sun reduces the cosmic-ray "
        "intensity in the inner solar system.")
    status, _ = c.extract_status(
        "Cosmic-Ray Modulation by Solar Wind", ["E. N. Parker"], head)
    assert status == c.MATCH


def test_extract_same_topic_wrong_paper_is_uncertain():
    """Stored text is a DIFFERENT same-topic paper: title words all overlap but
    the metadata first author is absent -> UNCERTAIN (escalate to LLM)."""
    head = (
        "# The Physics of Galactic Winds Driven by Cosmic Rays I: Diffusion\n"
        "Eliot Quataert, Todd A. Thompson, and Yan-Fei Jiang\nWe study galactic "
        "winds driven by cosmic rays with diffusion ...")
    status, _ = c.extract_status(
        "Galactic winds driven by cosmic rays", ["Ipavich, F. M."], head)
    assert status == c.UNCERTAIN


def test_extract_totally_wrong_is_uncertain():
    """Neither the title topic nor the first author at top -> UNCERTAIN (the
    deterministic pass never declares MISMATCH; the LLM does)."""
    head = (
        "LETTERS TO NATURE\nEmtage, J. L. T. & Jensen, R. E. J. Cell Biol. 122, "
        "1003 (1993). Dekker, P. J. T. et al. FEBS Lett. 331, 66 (1993).")
    status, _ = c.extract_status(
        "Delayed recombination as a major source of the soft X-ray background",
        ["Breitschwerdt, D."], head)
    assert status == c.UNCERTAIN


def test_extract_cited_author_below_top_is_not_a_match():
    """A wrong same-topic paper whose text CITES the metadata author deep in the
    body must NOT match: the author appears only beyond the top window."""
    top = ("# The Physics of Galactic Winds Driven by Cosmic Rays I: Diffusion\n"
           "Eliot Quataert, Todd A. Thompson, Yan-Fei Jiang\n"
           "We study galactic winds driven by cosmic rays with diffusion. ")
    deep = "x" * 1600 + " following Ipavich 1975 the wind is driven by ... "
    status, _ = c.extract_status(
        "Galactic winds driven by cosmic rays", ["Ipavich, F. M."], top + deep)
    assert status == c.UNCERTAIN   # 'Ipavich' is past the top window -> not a match


def test_extract_empty_head_is_uncertain():
    status, _ = c.extract_status("Anything", ["Author"], "")
    assert status == c.UNCERTAIN


# --------------------------- record verdict -------------------------------

ABS = "We present a faithful and sufficiently long abstract of the paper here."


def test_verdict_no_abstract_never_ok():
    v = c.record_verdict(abstract="", title="Cosmic-Ray Transport Coefficients",
                         authors=["Joe Giacalone"], extract_head=GIACALONE_HEAD)
    assert v["has_abstract"] is False
    assert v["ok"] is False


def test_verdict_metadata_only_with_abstract_is_ok():
    """A record with an abstract but NO stored extract is allowed."""
    v = c.record_verdict(abstract=ABS, title="Some Paper",
                         authors=["A. Author"], extract_head=None)
    assert v["extract_status"] == c.NO_EXTRACT
    assert v["ok"] is True


def test_verdict_abstract_plus_matching_extract_is_ok():
    v = c.record_verdict(abstract=ABS, title="Cosmic-Ray Transport Coefficients",
                         authors=["Joe Giacalone"], extract_head=GIACALONE_HEAD)
    assert v["ok"] is True
    assert v["needs_llm"] is False


def test_verdict_abstract_plus_unmatched_extract_not_ok():
    head = ("LETTERS TO NATURE\nEmtage & Jensen J. Cell Biol. 122, 1003 (1993).")
    v = c.record_verdict(abstract=ABS, title="Soft X-ray background paper",
                         authors=["Breitschwerdt, D."], extract_head=head)
    assert v["ok"] is False
    assert v["extract_status"] == c.UNCERTAIN
    assert v["needs_llm"] is True


def test_verdict_uncertain_extract_flags_needs_llm():
    head = ("The Physics of Galactic Winds Driven by Cosmic Rays\nQuataert, "
            "Thompson, Jiang\nWe study galactic winds driven by cosmic rays.")
    v = c.record_verdict(abstract=ABS, title="Galactic winds driven by cosmic rays",
                         authors=["Ipavich, F. M."], extract_head=head)
    assert v["needs_llm"] is True
    assert v["ok"] is False   # not ok until the uncertainty is resolved
