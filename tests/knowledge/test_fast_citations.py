"""FAST scoring reports bracket tokens outside the answer's available sources."""

import json

from papervault.eval import _fast_backbone as fast


def test_unresolved_citations_reports_every_occurrence():
    answer = "[An2017] [preliminary study] [Dst Index entity] [12] [preliminary study]"
    assert fast.unresolved_citations(answer, ["An2017"]) == [
        "preliminary study", "Dst Index entity", "12", "preliminary study",
    ]


def test_unresolved_citations_accepts_papers_and_operator_sources():
    answer = "[An2017][Suvorovand] [textbook:Griffiths] [notebook:idea] [web:nasa]"
    assert fast.unresolved_citations(answer, ["An2017", "Suvorovand"]) == []


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
