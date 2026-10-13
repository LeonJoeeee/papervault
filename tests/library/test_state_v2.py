"""Pure acquisition contracts from the accepted v2 spec, Appendix A1/A3."""

from dataclasses import replace
import importlib

import pytest

from papervault.library.models import Paper


@pytest.fixture
def v2():
    return importlib.import_module("papervault.library.state_v2")


def eligible(v2, **fields):
    return v2.TerminalState(retrieval=v2.Retrieval.ELIGIBLE,
                            policy=v2.Policy.CLEAR, **fields)


def test_admission_all_conditions_satisfied(v2):
    assert v2.admission(eligible(v2), identity_resolved=True, key="K") is True


@pytest.mark.parametrize("condition", [
    "quarantine", "retrieval", "alias", "alias_target", "identity", "policy", "lease", "budget",
])
def test_each_condition_independently_blocks_acquisition(v2, condition):
    state = eligible(v2)
    context = {"identity_resolved": True, "key": "K", "lease_ok": True, "budget_ok": True}
    if condition == "quarantine":
        state = replace(state, quarantine=True)
    elif condition == "retrieval":
        state = replace(state, retrieval=v2.Retrieval.EXHAUSTED)
    elif condition == "alias":
        state = replace(state, merge=v2.Merge.ALIAS, alias_key="other")
    elif condition == "alias_target":
        state = replace(state, alias_key="other")
    elif condition == "policy":
        state = replace(state, policy=v2.Policy.UNKNOWN)
    else:
        context[{"identity": "identity_resolved", "lease": "lease_ok", "budget": "budget_ok"}[condition]] = False
    assert v2.admission(state, **context) is False


@pytest.mark.parametrize("retrieval", ["ineligible", "in-progress", "exhausted", "acquired", "verified"])
def test_only_eligible_can_acquire(v2, retrieval):
    assert not v2.admission(replace(eligible(v2), retrieval=v2.Retrieval(retrieval)),
                            identity_resolved=True, key="K")


def test_policy_unknown_and_excluded_deny_even_with_grant(v2):
    grant = v2.PolicyOverride(scope=("acquire",), reasons=("off_domain",), revision=4, key="K")
    for policy in (v2.Policy.UNKNOWN, v2.Policy.EXCLUDED):
        state = eligible(v2, policy_reasons=("off_domain",), override=grant)
        assert not v2.effective_policy(replace(state, policy=policy), key="K")
    assert not v2.effective_policy(eligible(v2, policy_reasons=("off_domain",)), key="K")


@pytest.mark.parametrize("mutation", [None, "action", "reason", "revision", "key", "missing"])
def test_override_scope_revision_and_origin_bound_grant(v2, mutation):
    grant = v2.PolicyOverride(scope=("acquire",), reasons=("off_domain",), revision=4, key="K")
    state = replace(eligible(v2, policy_reasons=("off_domain",), override=grant),
                    policy=v2.Policy.OVERRIDE)
    if mutation == "action":
        state = replace(state, override=replace(grant, scope=("serve",)))
    elif mutation == "reason":
        state = replace(state, policy_reasons=("off_domain", "owner_excluded"))
    elif mutation == "revision":
        state = replace(state, override=replace(grant, revision=3))
    elif mutation == "key":
        state = replace(state, override=replace(grant, key="merged-away"))
    elif mutation == "missing":
        state = replace(state, override=None)
    assert v2.effective_policy(state, key="K", required_revision=4) is (mutation is None)
    assert v2.admission(state, key="K", identity_resolved=True,
                        required_revision=4) is (mutation is None)


@pytest.mark.parametrize("value", [None, "unexpected", {}, 0])
def test_predicate_is_total_and_fails_closed_for_unknown_records(v2, value):
    assert v2.admission(value, identity_resolved=True) is False
    assert v2.effective_policy(value) is False


@pytest.mark.parametrize("field,value", [
    ("policy_reasons", None), ("policy_reasons", 7),
    ("required_revision", None), ("required_revision", "4"),
])
def test_effective_policy_is_total_for_malformed_state(v2, field, value):
    grant = v2.PolicyOverride(scope=("acquire",), revision=4, key="K")
    state = replace(eligible(v2, override=grant), policy=v2.Policy.OVERRIDE)
    assert v2.effective_policy(replace(state, **{field: value}), key="K") is False


@pytest.mark.parametrize("field,value", [
    ("scope", None), ("scope", 7), ("reasons", 7), ("revision", None), ("revision", "4"),
])
def test_effective_policy_is_total_for_malformed_grant(v2, field, value):
    grant = v2.PolicyOverride(scope=("acquire",), revision=4, key="K")
    state = replace(eligible(v2, override=replace(grant, **{field: value})), policy=v2.Policy.OVERRIDE)
    assert v2.effective_policy(state, key="K") is False


@pytest.mark.parametrize("status,retrieval,policy", [
    ("pending", "eligible", "clear"),
    ("metadata_only", "exhausted", "unknown"),
    ("ok", "acquired", "unknown"),
    ("extract_failed", "acquired", "unknown"),
    ("failed", "exhausted", "unknown"),
    ("unexpected", "ineligible", "unknown"),
])
def test_legacy_mapping_is_conservative_and_read_only(v2, status, retrieval, policy):
    paper = Paper(key="K", download_status=status)
    before = paper.model_dump()
    state = v2.derive_legacy(paper)
    assert state.retrieval.value == retrieval
    assert state.policy.value == policy
    assert state.merge == v2.Merge.CANONICAL and state.quarantine is False
    assert paper.model_dump() == before


@pytest.mark.parametrize("status,path,engine,expected", [
    ("ok", "extracts/md/K.md", "mineru2.5-pro", "verified"),
    ("ok", None, "mineru2.5-pro", "acquired"),
    ("ok", "extracts/md/K.md", "", "acquired"),
    ("extract_failed", "extracts/md/K.md", "mineru2.5-pro", "acquired"),
])
def test_verified_requires_existing_extraction_evidence(v2, status, path, engine, expected):
    paper = Paper(key="K", download_status=status, md_path=path, md_engine=engine)
    assert v2.derive_legacy(paper).retrieval.value == expected


def test_legacy_mapping_is_total_for_missing_or_unknown_input(v2):
    for paper in (None, {}, {"download_status": None}, {"download_status": []}):
        state = v2.derive_legacy(paper)
        assert state.policy == v2.Policy.UNKNOWN
        assert state.retrieval == v2.Retrieval.INELIGIBLE
