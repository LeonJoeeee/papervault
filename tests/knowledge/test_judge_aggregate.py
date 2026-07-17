"""Pure unit tests for the judge aggregation + validation (experiments/eval/judge_aggregate.py).

No judge / no LLM — canned judge JSONs. Covers: a valid JSON passes; each self-consistency
identity rejects when broken; trap correct-refusal validates; seed-averaging + spread.
"""
from __future__ import annotations

import json
import math

import pytest

from papervault.eval import judge_aggregate as J


def _good(qid="q", *, prec=1.0, rec=1.0, nug=1.0, faith=1.0, relv=90):
    """A self-consistent judge JSON: 1 entail check, 1 cited claim, 1 covered nugget."""
    return {
        "qid": qid,
        "trap_correct_refusal": False,
        "citation_checks": [{"sentence": "s", "paper_key": "A", "verdict": "entail"}],
        "citation_support_precision": prec,
        "citation_recall": rec,
        "n_substantive_claims": 1,
        "n_claims_with_citation": 1,
        "nugget_judgements": [{"nugget_idx": 0, "nugget": "n", "coverage": "covered", "score": 1.0}],
        "nugget_recall": nug,
        "faithfulness": faith,
        "faithfulness_unsupported_claims": [],
        "relevance": relv,
        "relevance_note": "ok",
    }


# ---- validation: good ------------------------------------------------------
def test_valid_json_passes():
    J.validate_judge_json(_good(), n_gold_nuggets=1)  # no raise


def test_trap_correct_refusal_with_empty_lists_validates():
    j = {
        "qid": "t", "trap_correct_refusal": True,
        "citation_checks": [], "citation_support_precision": 1.0,
        "citation_recall": 1.0, "n_substantive_claims": 0, "n_claims_with_citation": 0,
        "nugget_judgements": [], "nugget_recall": 1.0,
        "faithfulness": 1.0, "faithfulness_unsupported_claims": [],
        "relevance": 100, "relevance_note": "correct refusal",
    }
    J.validate_judge_json(j, n_gold_nuggets=0)  # 0 checks/claims/nuggets -> identities give 1.0


# ---- validation: each identity broken --------------------------------------
def test_bad_precision_rejected():
    j = _good(prec=0.5)  # 1 entail / 1 check should be 1.0
    with pytest.raises(J.JudgeValidationError):
        J.validate_judge_json(j)


def test_bad_citation_recall_rejected():
    j = _good(rec=0.5)  # 1 cited / 1 claim should be 1.0
    with pytest.raises(J.JudgeValidationError):
        J.validate_judge_json(j)


def test_bad_nugget_recall_rejected():
    j = _good(nug=0.5)  # one covered nugget -> mean should be 1.0
    with pytest.raises(J.JudgeValidationError):
        J.validate_judge_json(j)


def test_nugget_score_mismatch_rejected():
    j = _good()
    j["nugget_judgements"][0]["score"] = 0.5  # coverage=covered but score 0.5
    with pytest.raises(J.JudgeValidationError):
        J.validate_judge_json(j)


def test_bad_verdict_rejected():
    j = _good()
    j["citation_checks"][0]["verdict"] = "kinda"
    with pytest.raises(J.JudgeValidationError):
        J.validate_judge_json(j)


def test_relevance_out_of_range_rejected():
    j = _good(relv=150)
    with pytest.raises(J.JudgeValidationError):
        J.validate_judge_json(j)


def test_cited_more_than_claims_rejected():
    j = _good()
    j["n_claims_with_citation"] = 2  # > n_substantive_claims=1
    with pytest.raises(J.JudgeValidationError):
        J.validate_judge_json(j)


def test_nugget_count_mismatch_rejected_when_gold_given():
    j = _good()  # has 1 nugget judgement
    with pytest.raises(J.JudgeValidationError):
        J.validate_judge_json(j, n_gold_nuggets=2)


def test_missing_chunk_verdict_counts_against_precision():
    # 1 entail + 1 missing_chunk = precision 0.5; a JSON claiming 0.5 must validate
    j = _good(prec=0.5)
    j["citation_checks"].append({"sentence": "s2", "paper_key": "B", "verdict": "missing_chunk"})
    J.validate_judge_json(j)  # 1 entail / 2 checks = 0.5 -> consistent


# ---- seed aggregation ------------------------------------------------------
def test_aggregate_seeds_means_and_spread():
    # three seeds with relevance 80/90/100 -> mean relevance .9 (normalised), sd>0
    seeds = [_good(qid="q", relv=80), _good(qid="q", relv=90), _good(qid="q", relv=100)]
    agg = J.aggregate_seeds(seeds, n_gold_nuggets=1)
    assert agg["n_seeds"] == 3 and agg["qid"] == "q"
    assert math.isclose(agg["mean"]["relevance"], 0.9, abs_tol=1e-9)
    assert agg["mean"]["nugget_recall"] == 1.0
    assert agg["sd"]["relevance"] > 0.0
    assert agg["sd"]["nugget_recall"] == 0.0  # identical across seeds


def test_aggregate_seeds_rejects_invalid_seed():
    seeds = [_good(), _good(prec=0.5)]  # second is inconsistent
    with pytest.raises(J.JudgeValidationError):
        J.aggregate_seeds(seeds)


# ---- trap policy: traps excluded from quality tables, scored as a guardrail (tail 9) -------
def _trap_refusal(qid: str) -> dict:
    """A trap correct-refusal judge JSON: pinned 1.0 with empty check/nugget lists."""
    return {
        "qid": qid, "trap_correct_refusal": True,
        "citation_checks": [], "citation_support_precision": 1.0,
        "citation_recall": 1.0, "n_substantive_claims": 0, "n_claims_with_citation": 0,
        "nugget_judgements": [], "nugget_recall": 1.0,
        "faithfulness": 1.0, "faithfulness_unsupported_claims": [],
        "relevance": 100, "relevance_note": "correct refusal",
    }


def _write_seeds(base, qid, seed_jsons):
    base.mkdir(parents=True, exist_ok=True)
    for k, j in enumerate(seed_jsons):
        (base / f"{qid}.seed{k}.json").write_text(json.dumps(j))


def _write_gold(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def test_per_question_scalars_excludes_traps_from_quality_tables(tmp_path):
    # one answerable + one trap. The trap's pinned-1.0 row must NOT appear in any quality table
    # (it would inject an always-tied delta-0 row and tighten the CI vs the n=21 backbone).
    judge_dir = tmp_path / "judge"
    tag = "v"
    base = judge_dir / tag
    _write_seeds(base, "ans", [_good("ans"), _good("ans"), _good("ans")])
    _write_seeds(base, "trap", [_trap_refusal("trap")] * 3)

    gold = tmp_path / "gold.jsonl"
    _write_gold(gold, [
        {"qid": "ans", "intent": "i", "gold_keys": ["A"], "nuggets": ["n"],
         "per_paper_relevance": {"A": "directly-answering"}, "expected_coverage_band": "thin"},
        {"qid": "trap", "intent": "i", "gold_keys": [], "nuggets": [],
         "per_paper_relevance": {}, "expected_coverage_band": "empty"},
    ])

    tables = J.per_question_scalars(judge_dir, gold, tag=tag)
    for m in J.JUDGE_METRICS:
        assert "ans" in tables[m]      # answerable present
        assert "trap" not in tables[m]  # trap dropped from EVERY quality table


def test_trap_correct_refusal_rate_guardrail(tmp_path):
    # two traps: one judged correct-refusal (all seeds True), one not (majority False).
    judge_dir = tmp_path / "judge"
    tag = "v"
    base = judge_dir / tag
    _write_seeds(base, "trap_ok", [_trap_refusal("trap_ok")] * 3)
    bad = _good("trap_bad")          # judge says the system DID make claims on a trap
    bad["trap_correct_refusal"] = False
    _write_seeds(base, "trap_bad", [bad, bad, bad])
    # an answerable question must be ignored by the trap-rate entirely
    _write_seeds(base, "ans", [_good("ans")] * 3)

    gold = tmp_path / "gold.jsonl"
    _write_gold(gold, [
        {"qid": "trap_ok", "intent": "i", "gold_keys": [], "nuggets": [],
         "per_paper_relevance": {}, "expected_coverage_band": "empty"},
        {"qid": "trap_bad", "intent": "i", "gold_keys": [], "nuggets": [],
         "per_paper_relevance": {}, "expected_coverage_band": "empty"},
        {"qid": "ans", "intent": "i", "gold_keys": ["A"], "nuggets": ["n"],
         "per_paper_relevance": {"A": "directly-answering"}, "expected_coverage_band": "thin"},
    ])

    out = J.trap_correct_refusal_rate(judge_dir, gold, tag=tag)
    assert out["n_traps_scored"] == 2          # only the two traps, not the answerable
    assert out["trap_correct_refusal_rate"] == 0.5  # 1 of 2 traps correctly refused
    assert out["per_qid"] == {"trap_ok": True, "trap_bad": False}
