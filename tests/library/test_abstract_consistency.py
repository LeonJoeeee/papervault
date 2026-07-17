"""Tests for the forward abstract-consistency mechanism:
fetch_abstract_by_doi, the reconcile abstract-backfill sweep, and the janitor.
"""
from __future__ import annotations

import asyncio

import responses

from papervault.library import Library, fetch
from papervault.library.services import reconcile


def _run(c):
    return asyncio.run(c)


def _inv(text):
    inv = {}
    for i, w in enumerate(text.split()):
        inv.setdefault(w, []).append(i)
    return inv


_OA = "https://api.openalex.org/works/https://doi.org/10.1/x"


@responses.activate
def test_fetch_abstract_by_doi_reconstructs_openalex_inverted_index():
    text = ("We present a detailed study of cosmic ray modulation in the "
            "heliosphere over a full solar cycle using a transport model.")
    responses.add(responses.GET, _OA,
                  json={"abstract_inverted_index": _inv(text)}, status=200)
    assert fetch.fetch_abstract_by_doi("10.1/x") == text


@responses.activate
def test_fetch_abstract_by_doi_miss_returns_empty():
    # OpenAlex has no abstract; SS endpoint unregistered -> blocked -> "".
    responses.add(responses.GET, _OA, json={"abstract_inverted_index": {}}, status=200)
    assert fetch.fetch_abstract_by_doi("10.1/x") == ""


def _mk_extract(lib, key, title, body):
    p, _ = lib.upsert({"title": title, "authors": ["Smith"], "year": 2024,
                       "doi": f"10.1/{key}"})
    p.abstract = ""
    md = lib.md_path(p.key)
    md.write_text(body)
    p.md_path = str(md.relative_to(lib.root))
    lib.save()
    return p


def test_abstract_section_pulls_verbatim(tmp_path):
    lib = Library(tmp_path)
    body = ("# Transport of cosmic rays\nSmith\n\n**Abstract.** We model cosmic "
            "ray transport in the heliosphere and present new diffusion results "
            "in considerable detail for the inner heliosphere here.\n\n"
            "## 1. Introduction\nbody text follows.")
    p = _mk_extract(lib, "a", "Transport of cosmic rays", body)
    ab = reconcile._abstract_section(lib, p)
    assert "diffusion results" in ab.lower()


def test_opening_paragraph_takes_prose_skips_refs(tmp_path):
    lib = Library(tmp_path)
    # No labelled Abstract: a prose lede paragraph should be taken...
    prose = ("It is shown that the hydrodynamic outflow of gas from the sun "
             "reduces the cosmic ray intensity in the inner solar system, and "
             "we argue that this explains the eleven year variation that is "
             "observed at the Earth over the solar cycle in great detail.")
    p1 = _mk_extract(lib, "prose", "Solar wind modulation",
                     "# Solar wind modulation\nParker\n\n" + prose + "\n\n## Body\nmore")
    assert "hydrodynamic outflow" in reconcile._opening_paragraph(lib, p1).lower()
    # ...but a reference-list head must NOT be stored as an abstract.
    refs = ("[1] Smith, J. Astrophys. J. 580, 100 (2020). [2] Jones, A. ApJ "
            "920, 45 (2021). [3] Lee, B. JGR 127, 1234 (2022). [4] Park, C. "
            "Nature 600, 12 (2023). [5] Wong, D. Science 370, 9 (2019).")
    p2 = _mk_extract(lib, "refs", "Some letter paper",
                     "# LETTERS\n" + refs + "\n\nReferences continue")
    assert reconcile._opening_paragraph(lib, p2) == ""


def test_abstract_sweep_fills_no_abstract_row_from_its_extract(tmp_path):
    lib = Library(tmp_path)
    body = ("# Transport of cosmic rays\nSmith\n\n**Abstract.** We model cosmic "
            "ray transport in the heliosphere and present new diffusion results "
            "in considerable detail for the inner heliosphere here.\n\n"
            "## 1. Introduction\nbody.")
    p = _mk_extract(lib, "b", "Transport of cosmic rays", body)
    n = _run(reconcile._abstract_sweep(lib, cap=10))
    assert n == 1
    assert len((lib.get(p.key).abstract or "").strip()) >= 60


def test_janitor_purges_only_terminal_no_abstract_no_extract(tmp_path):
    lib = Library(tmp_path)
    # victim: terminal, no abstract, no extract
    v, _ = lib.upsert({"title": "Ghost stub paper title here", "authors": ["A"],
                       "year": 2024, "doi": "10.1/ghost"})
    v.abstract = ""
    v.download_status = "failed"
    # keep #1: terminal + no abstract but HAS an extract
    k1 = _mk_extract(lib, "withext", "Real paper with extract title",
                     "# Real paper with extract title\n\nbody text here for the paper.")
    k1.download_status = "failed"
    # keep #2: terminal, no extract, but HAS an abstract
    k2, _ = lib.upsert({"title": "Real metadata-only paper title", "authors": ["B"],
                        "year": 2024, "doi": "10.1/metaonly"})
    k2.abstract = "x" * 80
    k2.download_status = "metadata_only"
    lib.save()
    n = _run(reconcile._janitor_sweep(lib))
    assert n == 1
    assert lib.get(v.key) is None
    assert lib.get(k1.key) is not None
    assert lib.get(k2.key) is not None
