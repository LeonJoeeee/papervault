"""FAST scoring reports bracket tokens outside the answer's available sources."""

import json

import pytest

from papervault.eval import _fast_backbone as fast


def test_unresolved_citations_reports_every_occurrence():
    answer = "[An2017] [preliminary study] [Dst Index entity] [12] [preliminary study]"
    assert fast.unresolved_citations(answer, ["An2017"]) == [
        "preliminary study", "Dst Index entity", "12", "preliminary study",
    ]


def test_unresolved_citations_accepts_papers_and_operator_sources():
    answer = "[An2017][Suvorovand] [textbook:Griffiths] [notebook:idea] [web:nasa]"
    assert fast.unresolved_citations(answer, ["An2017", "Suvorovand"]) == []


def test_unresolved_citation_breakdown_separates_classes_and_keeps_total():
    # #122 fixture: one resolving paper, one canonical operator cite, then one token per class.
    answer = ("[Reames2023] [textbook:Baumjohann2012] [Baumjohann2012, textbook] "
              "[preliminary CCE study] [9.74/(4.69+VBs)]")
    assert fast.unresolved_citation_breakdown(answer, ["Reames2023"]) == {
        "paper_stand_in": 1, "operator_shape": 1, "numeric_other": 1, "total": 3,
    }


def test_unresolved_citation_breakdown_counts_repeats_and_empty_answers():
    answer = "[Koskinen2011, textbook] [Koskinen2011, textbook] [Koskinen2011] [12]"
    assert fast.unresolved_citation_breakdown(answer, []) == {
        "paper_stand_in": 1, "operator_shape": 2, "numeric_other": 1, "total": 4,
    }
    assert fast.unresolved_citation_breakdown("", []) == {
        "paper_stand_in": 0, "operator_shape": 0, "numeric_other": 0, "total": 0,
    }


@pytest.mark.parametrize("token, expected", [
    ("Baumjohann2012, textbook", "operator_shape"),
    ("Textbook: Parks2004", "operator_shape"),
    ("Textbook:Parks2004", "operator_shape"),
    ("idea-scope; notebook", "operator_shape"),
    ("9.74/(4.69+VBs)", "numeric_other"),
    ("12", "numeric_other"),
    ("2, 3", "numeric_other"),
    ("B^2 = 2 mu0 p", "numeric_other"),
    # Strict residual: anything not clearly operator-shaped or mathematical stays a stand-in.
    ("preliminary CCE study", "paper_stand_in"),
    ("Koskinen2011", "paper_stand_in"),
    ("Smith et al. (2019)", "paper_stand_in"),
    ("Smith 2019/2020", "paper_stand_in"),
    ("web-based survey", "paper_stand_in"),
    ("preliminary web study", "paper_stand_in"),
    ("Dst Index entity", "paper_stand_in"),
])
def test_classify_unresolved_keeps_stand_ins_as_the_residual_class(token, expected):
    assert fast.classify_unresolved(token) == expected


def test_fast_arm_scores_prose_including_traps_and_distinguishes_absent_answers(
    tmp_path, monkeypatch, capsys,
):
    gold = [{"qid": qid, "gold_keys": keys} for qid, keys in [
        ("bad", ["An2017"]), ("clean", ["An2017"]), ("trap", []), ("no-synth", ["An2017"]),
    ]]
    records = [{"qid": qid, "cited_papers": ["An2017"], "answer": answer} for qid, answer in [
        ("bad", "[An2017] [preliminary study]"), ("clean", "[An2017] [web:nasa]"),
        ("trap", "[Dst Index entity]"), ("no-synth", None),
    ]]
    (tmp_path / "gold.jsonl").write_text("\n".join(map(json.dumps, gold)))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "fixture.jsonl").write_text("\n".join(map(json.dumps, records)))
    monkeypatch.setattr(fast, "EVAL", tmp_path)
    monkeypatch.setattr("sys.argv", ["_fast_backbone", "fixture", "gold.jsonl"])
    fast.main()
    report = json.loads(capsys.readouterr().out)
    assert report["prose_citation_answers_checked"] == 3
    assert report["prose_citation_answers_unavailable"] == ["no-synth"]
    assert report["unresolved_prose_citations"] == {
        "bad": ["preliminary study"], "trap": ["Dst Index entity"],
    }
    assert report["unresolved_prose_citation_classes"] == {
        "paper_stand_in": 2, "operator_shape": 0, "numeric_other": 0, "total": 2,
    }
