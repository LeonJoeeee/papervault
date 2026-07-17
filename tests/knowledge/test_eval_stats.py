"""Pure unit tests for the paired-difference stats layer (experiments/eval/stats.py).

No DB / no LLM. Seeds are pinned so bootstrap/permutation are deterministic. Toy inputs with
known answers (a clear all-positive delta -> improvement; a no-op delta -> not improved; a
lower-is-better reduction; an exact small-n permutation p; a zero-noise floor).
"""
from __future__ import annotations

import numpy as np
import pytest

from papervault.eval import stats as S


def _table(vals: dict[str, float]) -> dict[str, float]:
    return dict(vals)


# ---- pairing ---------------------------------------------------------------
def test_paired_deltas_only_common_qids():
    b = {"q1": 0.2, "q2": 0.4, "q3": 0.1}
    v = {"q1": 0.5, "q2": 0.4, "q9": 9.9}  # q9 only in variant, q3 only in baseline
    qids, d = S.paired_deltas(b, v)
    assert qids == ["q1", "q2"]
    assert np.allclose(d, [0.3, 0.0])


def test_paired_deltas_no_overlap_raises():
    try:
        S.paired_deltas({"a": 1.0}, {"b": 1.0})
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---- compare_metric: clear improvement -------------------------------------
def test_compare_metric_clear_improvement_higher_is_better():
    # every question improves by a solid margin -> CI low > 0 and p small -> improved
    b = {f"q{i}": 0.2 for i in range(10)}
    v = {f"q{i}": 0.6 for i in range(10)}
    res = S.compare_metric("recall", b, v, higher_is_better=True, seed=0)
    assert res.n == 10
    assert res.mean_delta > 0.39
    assert res.ci_low > 0.0
    assert res.p_value < 0.05
    assert res.improved is True


def test_compare_metric_noop_not_improved():
    # identical runs -> delta 0 everywhere -> p=1, CI=(0,0), not improved
    b = {f"q{i}": 0.3 for i in range(8)}
    v = dict(b)
    res = S.compare_metric("recall", b, v, higher_is_better=True, seed=0)
    assert res.mean_delta == 0.0
    assert res.ci_low == 0.0 and res.ci_high == 0.0
    assert res.p_value == 1.0
    assert res.improved is False


def test_compare_metric_regression_not_improved():
    # variant is worse -> mean_delta < 0, ci_low not > 0 -> not improved
    b = {f"q{i}": 0.6 for i in range(8)}
    v = {f"q{i}": 0.3 for i in range(8)}
    res = S.compare_metric("recall", b, v, higher_is_better=True, seed=0)
    assert res.mean_delta < 0
    assert res.improved is False


def test_compare_metric_lower_is_better_reduction_reported_in_original_units():
    # a poison rate that drops from 0.5 -> 0.1 everywhere is an IMPROVEMENT for lower-is-better;
    # reported mean_delta / CI must be in original (negative) direction.
    b = {f"q{i}": 0.5 for i in range(10)}
    v = {f"q{i}": 0.1 for i in range(10)}
    res = S.compare_metric("hallucinated_rate", b, v, higher_is_better=False, seed=0)
    assert res.mean_delta < 0                 # original direction: rate went down
    assert res.ci_high < 0                     # whole CI below 0 (a real reduction)
    assert res.improved is True


# ---- permutation p exactness (small n) -------------------------------------
def test_sign_flip_p_exact_small_n():
    # n=3 all-positive equal deltas: only the all-(+) flip has |mean| >= obs among 8 -> 1/8?
    # actually flips that keep all signs positive OR all negative both reach |obs|; with equal
    # magnitudes, the extreme means are +obs and -obs -> 2 of 8 assignments hit >= obs.
    d = np.array([0.4, 0.4, 0.4])
    rng = np.random.default_rng(0)
    p = S._sign_flip_p(d, n_perm=0, rng=rng)
    assert abs(p - 2 / 8) < 1e-9


def test_sign_flip_p_all_zero_is_one():
    p = S._sign_flip_p(np.zeros(5), n_perm=0, rng=np.random.default_rng(0))
    assert p == 1.0


# ---- noise floor -----------------------------------------------------------
def test_noise_floor_zero_for_identical_repeats():
    runs = [{"q1": 0.5, "q2": 0.8}, {"q1": 0.5, "q2": 0.8}, {"q1": 0.5, "q2": 0.8}]
    nf = S.noise_floor(runs)
    assert nf["n_runs"] == 3 and nf["n_questions"] == 2
    # identical repeats => zero noise floor (numpy std leaves ~1e-17 float residue, allow tol)
    assert nf["per_question_sd_mean"] == pytest.approx(0.0, abs=1e-12)
    assert nf["run_mean_sd"] == pytest.approx(0.0, abs=1e-12)
    assert nf["run_mean_range"] == pytest.approx(0.0, abs=1e-12)


def test_noise_floor_picks_up_spread():
    runs = [{"q1": 0.4}, {"q1": 0.6}, {"q1": 0.5}]
    nf = S.noise_floor(runs)
    assert nf["run_mean_range"] > 0.0
    assert nf["per_question_sd_mean"] > 0.0


def test_noise_floor_needs_two_runs():
    try:
        S.noise_floor([{"q1": 0.5}])
        assert False
    except ValueError:
        pass


# ---- determinism: same seed -> same CI/p -----------------------------------
def test_compare_metric_deterministic_under_seed():
    b = {f"q{i}": 0.2 + 0.01 * i for i in range(12)}
    v = {f"q{i}": 0.4 + 0.02 * i for i in range(12)}
    r1 = S.compare_metric("m", b, v, seed=7)
    r2 = S.compare_metric("m", b, v, seed=7)
    assert (r1.ci_low, r1.ci_high, r1.p_value) == (r2.ci_low, r2.ci_high, r2.p_value)


# --------------------------------------------------------------------------- #
# R-tail 1 (reconcile drill 2026-06-03): movable_qids + the redesigned jackknife.
# The original jackknife dropped ALL 21 qids and required each leave-one-out to re-clear the
# discrete sign-flip p<0.05 — which made the metric structurally UN-WINNABLE on its own corpus
# (18/21 Qs saturated → only ~3 movers → p floored at 0.25). These pin the fix: drop only MOVERS,
# use a continuous per-drop criterion, and let a genuine few-mover win survive.
# --------------------------------------------------------------------------- #
def test_movable_qids_excludes_saturated_ties():
    # 18 saturated (identical in both runs → delta 0) + 3 genuine movers.
    sat = {f"s{i}": 1.0 for i in range(18)}
    base = dict(sat); var = dict(sat)
    for m in ("m0", "m1", "m2"):
        base[m] = 0.3
        var[m] = 0.95
    movers = S.movable_qids(base, var)
    assert set(movers) == {"m0", "m1", "m2"}      # only the movable Qs, saturated ties dropped


def test_sign_flip_p_discrete_floor_is_set_by_mover_count():
    # The crux of R-tail 1: with the rest saturated at delta 0, the exact sign-flip p depends ONLY
    # on the number of movers, NOT the effect size — 3 movers → 0.25, 5 → 0.0625, 6 → 0.03125. So
    # <=5 movers can NEVER reach p<0.05 no matter how large the lift. This is why (i) had to drop
    # the hard p<0.05 gate. Pin the floor exactly (exact enumeration at n<=20).
    import numpy as np
    for nm, expect in [(3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125)]:
        d = np.array([0.5] * nm + [0.0] * (20 - nm))  # n=20 → exact enumeration
        p = S._sign_flip_p(d, n_perm=0, rng=np.random.default_rng(0))
        assert abs(p - expect) < 1e-9, f"{nm} movers: p={p} != {expect}"


def test_jackknife_genuine_few_mover_win_survives():
    # THE R-tail-1 regression: 18 saturated + 3 genuine movers (0.3→0.95). The OLD jackknife
    # rejected this (each leave-one-out fell to 2 movers → p never < 0.05). The redesigned one
    # drops only the 3 MOVERS and uses the continuous ci_low>0 criterion → survives.
    sat = {f"s{i}": 1.0 for i in range(18)}
    base = dict(sat); var = dict(sat)
    for m in ("m0", "m1", "m2"):
        base[m] = 0.3
        var[m] = 0.95
    res = S.jackknife_headline(base, var, n_boot=3000, n_perm=3000, seed=0)
    assert res["n_movers"] == 3
    assert set(res["per_qid"]) == {"m0", "m1", "m2"}   # only movers are dropped
    assert all(res["per_qid"].values())
    assert res["survives"] is True


def test_jackknife_single_mover_is_not_robust():
    # 1 mover among saturated ties → single-Q-driven → n_movers<2 → not robust.
    base = {f"s{i}": 1.0 for i in range(18)}
    var = dict(base)
    base["m0"] = 0.1
    var["m0"] = 0.95
    res = S.jackknife_headline(base, var, n_boot=2000, n_perm=2000, seed=0)
    assert res["n_movers"] == 1
    assert res["survives"] is False
    assert res["per_qid"] == {}


def test_jackknife_win_hinging_on_one_of_two_movers_fails():
    # 2 movers but the win HINGES on m0 (huge) — m1 is a tiny WRONG-WAY wobble. Dropping m0 leaves
    # only m1 (delta negative) → that leave-one-out is not net-positive → does not survive.
    base = {f"s{i}": 1.0 for i in range(18)}
    var = dict(base)
    base["m0"], var["m0"] = 0.1, 0.95             # big genuine mover
    base["m1"], var["m1"] = 0.9, 0.88             # small mover the WRONG way
    res = S.jackknife_headline(base, var, n_boot=2000, n_perm=2000, seed=0)
    assert res["n_movers"] == 2
    # dropping m0 leaves m1 whose delta is negative → that drop does not survive:
    assert res["per_qid"]["m0"] is False
    assert res["survives"] is False
