"""Pure unit tests for the reference-free metric (experiments/eval/metric_reffree.py).

No judge / no LLM — canned judge JSONs. Covers: PART 1 precision (validation: good + each broken
identity, band caps, pointer cap; per-question + run aggregation, empty-question floor) and PART 2
pairwise recall (validation: counts, verdict-grounding, importance; the cross-question tally +
weighted-balance tiebreak).
"""
from __future__ import annotations

import json
import math

import pytest

from papervault.eval import metric_reffree as M


# ================================================================================================
# PART 1 — PRECISION (per-chunk IMS)
# ================================================================================================

def _pj(qid="q", cid="A#0", *, ims=92, band="90-100", pointer=False):
    """A self-consistent precision-judge JSON."""
    return {
        "qid": qid, "chunk_id": cid, "paper_key": cid.split("#")[0],
        "counter_case": "thin", "pointer_only": pointer, "band": band, "ims": ims,
    }


# ---- validation: good ------------------------------------------------------
def test_precision_valid_passes():
    M.validate_precision_json(_pj())                       # 92 in 90-100
    M.validate_precision_json(_pj(ims=78, band="70-89"))
    M.validate_precision_json(_pj(ims=50, band="40-69"))
    M.validate_precision_json(_pj(ims=30, band="0-39"))
    M.validate_precision_json(_pj(ims=39, band="0-39", pointer=True))  # pointer at the cap


# ---- validation: each identity / cap broken --------------------------------
def test_precision_ims_out_of_range_rejected():
    with pytest.raises(M.JudgeValidationError):
        M.validate_precision_json(_pj(ims=150, band="90-100"))


def test_precision_ims_not_int_rejected():
    j = _pj(); j["ims"] = 92.5
    with pytest.raises(M.JudgeValidationError):
        M.validate_precision_json(j)


def test_precision_bool_ims_rejected():
    # bool is an int subclass — must be rejected so True/False can't masquerade as a score
    j = _pj(); j["ims"] = True
    with pytest.raises(M.JudgeValidationError):
        M.validate_precision_json(j)


def test_precision_bad_band_rejected():
    j = _pj(); j["band"] = "good"
    with pytest.raises(M.JudgeValidationError):
        M.validate_precision_json(j)


def test_precision_band_cap_40_69_rejected():
    # 40-69 band caps at 55; ims=60 violates the on-topic-no-payload cap
    with pytest.raises(M.JudgeValidationError):
        M.validate_precision_json(_pj(ims=60, band="40-69"))


def test_precision_band_cap_0_39_rejected():
    with pytest.raises(M.JudgeValidationError):
        M.validate_precision_json(_pj(ims=45, band="0-39"))


def test_precision_ims_below_band_floor_rejected():
    # ims=80 cannot sit in the 90-100 band
    with pytest.raises(M.JudgeValidationError):
        M.validate_precision_json(_pj(ims=80, band="90-100"))


def test_precision_pointer_cap_rejected():
    # pointer_only=true but ims=70 exceeds the pointer cap of 39
    j = _pj(ims=70, band="70-89"); j["pointer_only"] = True
    with pytest.raises(M.JudgeValidationError):
        M.validate_precision_json(j)


# ---- per-question + run aggregation ----------------------------------------
def test_precision_for_question_mean():
    # IMS 90 / 50 / 10 -> mean 50 -> /100 = 0.5
    js = [_pj(ims=90, band="90-100"), _pj(ims=50, band="40-69"), _pj(ims=10, band="0-39")]
    assert math.isclose(M.precision_for_question(js), 0.5, abs_tol=1e-9)


def test_precision_for_question_empty_is_zero():
    # a question that served no chunks -> 0.0 (the honest "too little material" floor)
    assert M.precision_for_question([]) == 0.0


def test_precision_for_question_validates():
    with pytest.raises(M.JudgeValidationError):
        M.precision_for_question([_pj(ims=60, band="40-69")])  # cap violation


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_precision_per_question_groups_by_qid(tmp_path):
    judge_dir = tmp_path / "judge"; tag = "v"; base = judge_dir / tag
    # q1: two chunks (ims 90, 70 -> mean 80 -> 0.80); q2: one chunk (ims 40 -> 0.40)
    _write(base / "q1.A#0.seed0.json", _pj("q1", "A#0", ims=90, band="90-100"))
    _write(base / "q1.B#1.seed0.json", _pj("q1", "B#1", ims=70, band="70-89"))
    _write(base / "q2.C#0.seed0.json", _pj("q2", "C#0", ims=40, band="40-69"))
    # a seed1 file must NOT leak into the seed0 table
    _write(base / "q1.A#0.seed1.json", _pj("q1", "A#0", ims=10, band="0-39"))

    table = M.precision_per_question(judge_dir, tag=tag)
    assert set(table) == {"q1", "q2"}
    assert math.isclose(table["q1"], 0.80, abs_tol=1e-9)
    assert math.isclose(table["q2"], 0.40, abs_tol=1e-9)
    assert math.isclose(M.run_precision(table), (0.80 + 0.40) / 2, abs_tol=1e-9)


def test_run_precision_empty_is_zero():
    assert M.run_precision({}) == 0.0


def test_chunk_id_keeps_duplicates_distinct():
    # same paper, two pieces -> distinct ids (no dedup by paper)
    assert M.chunk_id("Corti2018", 0) != M.chunk_id("Corti2018", 3)
    assert M.chunk_id(None, 0) == "none#0"


# ================================================================================================
# PART 2 — PAIRWISE RECALL GUARD
# ================================================================================================

def _gp(point="p", key="A", importance="high"):
    return {"point": point, "b_paper_key": key, "importance": importance}


def _lp(point="p", key="A", importance="high"):
    return {"point": point, "a_paper_key": key, "importance": importance}


def _pw(qid="q", *, gains=None, losses=None, verdict="tie"):
    gains = gains or []
    losses = losses or []
    return {
        "qid": qid, "intent_need": "need",
        "b_gains": gains, "b_losses": losses,
        "n_gains": len(gains), "n_losses": len(losses),
        "verdict": verdict, "justification": "because",
    }


# ---- validation ------------------------------------------------------------
def test_pairwise_valid_passes():
    M.validate_pairwise_json(_pw(gains=[_gp()], verdict="net_gain"))
    M.validate_pairwise_json(_pw(losses=[_lp()], verdict="net_loss"))
    M.validate_pairwise_json(_pw(verdict="tie"))


def test_pairwise_count_mismatch_rejected():
    j = _pw(gains=[_gp()], verdict="net_gain"); j["n_gains"] = 2
    with pytest.raises(M.JudgeValidationError):
        M.validate_pairwise_json(j)


def test_pairwise_bad_verdict_rejected():
    with pytest.raises(M.JudgeValidationError):
        M.validate_pairwise_json(_pw(verdict="better"))


def test_pairwise_bad_importance_rejected():
    with pytest.raises(M.JudgeValidationError):
        M.validate_pairwise_json(_pw(gains=[_gp(importance="huge")], verdict="net_gain"))


def test_pairwise_ungrounded_net_gain_rejected():
    # net_gain with zero gains is contradictory
    with pytest.raises(M.JudgeValidationError):
        M.validate_pairwise_json(_pw(verdict="net_gain"))


def test_pairwise_ungrounded_net_loss_rejected():
    with pytest.raises(M.JudgeValidationError):
        M.validate_pairwise_json(_pw(verdict="net_loss"))


# ---- tally -----------------------------------------------------------------
def test_pairwise_tally_counts_and_net(tmp_path):
    judge_dir = tmp_path / "judge"; tag = "v__vs__b"; base = judge_dir / tag
    base.mkdir(parents=True, exist_ok=True)
    # 2 net_gain, 1 net_loss, 1 tie  -> net_verdict net_gain
    _write(base / "q1.seed0.json", _pw("q1", gains=[_gp(importance="high")], verdict="net_gain"))
    _write(base / "q2.seed0.json", _pw("q2", gains=[_gp(importance="medium")], verdict="net_gain"))
    _write(base / "q3.seed0.json", _pw("q3", losses=[_lp(importance="low")], verdict="net_loss"))
    _write(base / "q4.seed0.json", _pw("q4", verdict="tie"))

    out = M.pairwise_tally(judge_dir, tag=tag)
    assert out["n_questions"] == 4
    assert out["n_net_gain"] == 2 and out["n_net_loss"] == 1 and out["n_tie"] == 1
    assert out["net_verdict"] == "net_gain"
    # weighted balance: gains 3+2=5, losses 1 -> +4
    assert math.isclose(out["weighted_balance"], 4.0, abs_tol=1e-9)


def test_pairwise_tally_weighted_balance_breaks_count_tie(tmp_path):
    judge_dir = tmp_path / "judge"; tag = "t"; base = judge_dir / tag
    base.mkdir(parents=True, exist_ok=True)
    # 1 net_gain (high=3) vs 1 net_loss (low=1): question counts tie 1-1, balance +2 -> net_gain
    _write(base / "q1.seed0.json", _pw("q1", gains=[_gp(importance="high")], verdict="net_gain"))
    _write(base / "q2.seed0.json", _pw("q2", losses=[_lp(importance="low")], verdict="net_loss"))
    out = M.pairwise_tally(judge_dir, tag=tag)
    assert out["n_net_gain"] == 1 and out["n_net_loss"] == 1
    assert math.isclose(out["weighted_balance"], 2.0, abs_tol=1e-9)
    assert out["net_verdict"] == "net_gain"  # count-tie broken by weighted balance


def test_pairwise_tally_validates(tmp_path):
    judge_dir = tmp_path / "judge"; tag = "bad"; base = judge_dir / tag
    base.mkdir(parents=True, exist_ok=True)
    bad = _pw("q1", verdict="net_gain")  # net_gain with no gains -> ungrounded
    _write(base / "q1.seed0.json", bad)
    with pytest.raises(M.JudgeValidationError):
        M.pairwise_tally(judge_dir, tag=tag)
