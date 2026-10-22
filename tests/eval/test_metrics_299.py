"""Regression tests for evidence identities and unfiltered citation scoring (#299)."""
import json

import pytest

from papervault.eval import _fast_backbone as fast
from papervault.eval import backbone
from papervault.eval.build_judge_prompts import fill_prompt


def test_backbone_accepts_both_paper_eras_and_explicit_identity_without_expanding_budget():
    chunks = [
        {"file_path": "Bare2017"},
        {"file_path": "paper/Old2018"},
        {"paper_key": "Explicit2019", "file_path": "opaque/path"},
        {"file_path": "textbook:Book#s1"},
        {"file_path": "Bare2017"},
        {"file_path": "Sixth2020"},
    ] + [{"file_path": "Bare2017"}] * 6 + [{"file_path": "Last2021"}]
    rec = {
        "data": {"chunks": chunks, "references": [{"file_path": "paper/Ref2022"}]},
        "retrieved_papers": ["Bare2017", "Old2018", "Explicit2019", "Sixth2020", "Last2021", "Ref2022"],
        "answer": "[Bare2017] [Old2018] [Explicit2019] [Last2021] [Ref2022]",
    }
    metrics = backbone.compute_question_metrics(rec, {"gold_keys": ["Sixth2020", "Last2021"]})
    assert backbone.retrieved_papers_from_chunks(rec["data"], top_n=5) == [
        "Bare2017", "Old2018", "Explicit2019",
    ]
    assert metrics.paper_recall_at_5 == 0
    assert metrics.paper_recall_at_12 == 0.5
    assert metrics.paper_recall_at_served_distinct == 1
    assert metrics.hallucinated_rate == 0


@pytest.mark.parametrize("candidates", [
    {"retrieved_papers": ["Seen2017"]},
    {"data": {"references": [{"file_path": "Seen2017"}]}},
])
def test_backbone_uses_saved_candidate_identity_for_citation_validation(candidates):
    metrics = backbone.compute_question_metrics(
        {**candidates, "answer": "[Seen2017] [Invented2026]"}, {"gold_keys": ["Seen2017"]},
    )
    assert metrics.hallucinated_rate == 0.5
    assert metrics.paper_recall_at_served_distinct == 1
    assert metrics.paper_recall_at_12 == 1


@pytest.mark.parametrize("chunk, label", [
    ({"paper_key": None, "source_key": "textbook:Book#s2", "content": "book"}, "textbook:Book"),
    ({"paper_key": None, "source_key": "textbook:Book", "file_path": "Book", "content": "book"}, "textbook:Book"),
    ({"paper_key": None, "file_path": "notebook:idea#s3", "content": "note"}, "notebook:idea"),
    ({"paper_key": None, "file_path": "web:nasa#s4", "content": "web"}, "web:nasa"),
])
def test_judge_prompt_preserves_operator_source_identity(chunk, label):
    prompt = fill_prompt({"data": {"chunks": [chunk]}}, {})
    assert f"source_key={label}" in prompt
    assert chunk["content"] in prompt


def _fast_report(tmp_path, monkeypatch, capsys, record):
    (tmp_path / "gold.jsonl").write_text(json.dumps({"qid": "q", "gold_keys": ["Seen2017"]}) + "\n")
    (tmp_path / "fixture.jsonl").write_text(json.dumps({"qid": "q", **record}) + "\n")
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["_fast_backbone", "fixture", str(tmp_path / "gold.jsonl")])
    fast.main()
    return json.loads(capsys.readouterr().out)


def test_fast_scores_raw_inline_papers_instead_of_filtered_cited_papers(tmp_path, monkeypatch, capsys):
    report = _fast_report(tmp_path, monkeypatch, capsys, {
        "data": {"chunks": [{"file_path": "Seen2017"}]},
        "retrieved_papers": ["Seen2017"], "cited_papers": ["Seen2017"],
        "answer": "Fact [Seen2017]; fabrication [Invented2026].",
    })
    assert report["paper_recall_at_served_distinct"] == 1
    assert report["hallucinated_rate"] == 0.5


@pytest.mark.parametrize("source_fields", [
    {"retrieved_sources": ["textbook:Book#s2", "notebook:idea", "web:nasa"]},
    {"cited_sources": ["textbook:Book", "notebook:idea#s3", "web:nasa"]},
    {"data": {"chunks": [
        {"source_key": "textbook:Book#s2"}, {"file_path": "notebook:idea#s3"}, {"file_path": "web:nasa"},
    ]}},
])
def test_fast_validates_operator_citations_and_normalizes_sections(
    tmp_path, monkeypatch, capsys, source_fields,
):
    report = _fast_report(tmp_path, monkeypatch, capsys, {
        "retrieved_papers": ["Seen2017"], "cited_papers": ["Seen2017"],
        "answer": "[textbook:Book] [notebook:idea#s4] [web:nasa] [web:invented]",
        **source_fields,
    })
    assert report["hallucinated_rate"] == 0.25
    assert report["unresolved_prose_citations"] == {"q": ["web:invented"]}


def test_fast_does_not_trust_cited_sources_when_retrieved_sources_present(tmp_path, monkeypatch, capsys):
    report = _fast_report(tmp_path, monkeypatch, capsys, {
        "retrieved_papers": ["Seen2017"], "cited_papers": ["Seen2017"],
        "retrieved_sources": [], "cited_sources": ["web:invented"],
        "data": {"chunks": [{"file_path": "web:invented"}]},
        "answer": "[web:invented]",
    })
    assert report["hallucinated_rate"] == 1
    assert report["unresolved_prose_citations"] == {"q": ["web:invented"]}


def test_fast_preserves_no_synthesis_candidate_fallback(tmp_path, monkeypatch, capsys):
    report = _fast_report(tmp_path, monkeypatch, capsys, {
        "data": {"chunks": [{"file_path": "paper/Seen2017"}]},
        "retrieved_papers": ["Seen2017"], "cited_papers": ["Seen2017"], "answer": None,
    })
    assert report["paper_recall_at_served_distinct"] == 1
    assert report["gold_citation_recall"] == 1
    assert report["hallucinated_rate"] == 0
    assert report["prose_citation_answers_unavailable"] == ["q"]


@pytest.mark.parametrize("answer", [
    "No citations, just [12] [2, 3] [B^2 = 2 mu0 p] [9.74/(4.69+VBs)].",
    "Links [NASA](https://nasa.gov), [citation][ref], and [a] enumeration.\n\n[ref]: https://example.com",
    "(KB 无相关知识)",
    "(synthesis LLM failed; see cited_papers.)",
])
def test_fast_rate_excludes_noncitations_math_and_failure_answers(tmp_path, monkeypatch, capsys, answer):
    report = _fast_report(tmp_path, monkeypatch, capsys, {
        "data": {"chunks": [{"paper_key": "Seen2017"}]}, "answer": answer,
    })
    assert report["hallucinated_rate"] is None


def test_fast_rate_counts_descriptive_stand_ins_from_raw_answer(tmp_path, monkeypatch, capsys):
    report = _fast_report(tmp_path, monkeypatch, capsys, {
        "retrieved_papers": ["Seen2017"], "cited_papers": ["Seen2017"],
        "answer": "[Seen2017] [preliminary study] [Seen2017]",
    })
    assert report["hallucinated_rate"] == 0.5
