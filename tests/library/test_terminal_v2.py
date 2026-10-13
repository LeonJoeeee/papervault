"""Flag compatibility, write-through persistence, and final dispatch rechecks."""

from dataclasses import replace
import asyncio
import hashlib
import itertools
import json

import pytest

from papervault.library.models import Paper
from papervault.library.services.classify import classify
from papervault.library import state_v2 as v2
from papervault.library.store import Library


class Facts:
    def has_pdf(self, key):
        return self.pdf

    def has_extract(self, key, fmt):
        return self.md

    def md_source(self, key):
        return self.source


def add(lib, **fields):
    paper, _ = lib.upsert({"title": "A sufficiently long acquisition state test paper",
                           "authors": ["Curie"], "year": 2026, "doi": "10.1/test", **fields})
    return paper


def test_flag_off_router_snapshot_matches_named_base(monkeypatch):
    monkeypatch.delenv("PAPERVAULT_TERMINAL_V2", raising=False)
    facts = Facts()
    results = []
    for status, pdf, md, source, hunt, attempts in itertools.product(
        ["pending", "ok", "metadata_only", "failed", "extract_failed", "", "ok:legacy", "unexpected"],
        [False, True], [False, True], ["firecrawl", "mineru2.5-pro"], [False, True], [0, 3],
    ):
        facts.pdf, facts.md, facts.source = pdf, md, source
        paper = Paper(key="K", download_status=status, firecrawl_pdf_hunt_exhausted=hunt,
                      extract_attempts=attempts)
        results.append(classify(paper, facts))
    # Captured from named base 7d8566d before any implementation changes (256 cases).
    assert hashlib.sha256(json.dumps(results).encode()).hexdigest() == (
        "555f1fde4ef4d12400f58350dbf5c5ab6c41adaa44e6b3f6814be48bd2a1b2e3")


@pytest.mark.parametrize("flag", [None, "0", "false"])
def test_flag_off_does_not_add_dimensions_on_writes(monkeypatch, tmp_path, flag):
    if flag is None:
        monkeypatch.delenv("PAPERVAULT_TERMINAL_V2", raising=False)
    else:
        monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", flag)
    lib = Library(tmp_path)
    paper = add(lib)
    paper.download_status = "failed"
    lib.save()
    assert "terminal_state" not in paper.model_dump()
    assert "terminal_state" not in json.loads(lib.index_path.read_text())["papers"][paper.key]


@pytest.mark.parametrize("status,retrieval,cause", [
    ("metadata_only", "exhausted", "retrieval_exhausted"),
    ("failed", "exhausted", "retrieval_exhausted"),
    ("ok", "acquired", "acquired"),
    ("extract_failed", "acquired", "extraction_failed"),
])
def test_new_terminal_writes_record_dimensions_and_cause(monkeypatch, tmp_path, status, retrieval, cause):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    paper.download_status = status
    lib.save()
    raw = json.loads(lib.index_path.read_text())["papers"][paper.key]["terminal_state"]
    assert (raw["retrieval"], raw["policy"], raw["merge"], raw["quarantine"], raw["cause"]) == (
        retrieval, "clear", "canonical", False, cause)
    assert Library(tmp_path).get(paper.key).acquisition_state == paper.acquisition_state


def test_terminal_ingress_also_has_explicit_dimensions(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "true")
    lib = Library(tmp_path)
    paper = add(lib, download_status="metadata_only")
    lib.save()
    assert paper.acquisition_state.retrieval == v2.Retrieval.EXHAUSTED
    assert json.loads(lib.index_path.read_text())["papers"][paper.key]["terminal_state"]["cause"]


def test_new_verified_write_requires_extraction_evidence(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    paper = Paper(key="K")
    paper.md_path, paper.md_engine = "extracts/md/K.md", "mineru2.5-pro"
    paper.download_status = "ok"
    assert paper.acquisition_state.retrieval == v2.Retrieval.VERIFIED


def test_legacy_dimensions_are_derived_and_only_touched_row_is_persisted(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    index = tmp_path / "index.json"
    rows = {key: {"key": key, "title": f"Legacy terminal paper {key}",
                  "download_status": "metadata_only"} for key in ("A", "B")}
    index.write_text(json.dumps({"version": 1, "papers": rows}))
    before = index.read_bytes()
    lib = Library(tmp_path)
    assert lib.get("A").acquisition_state.policy == v2.Policy.UNKNOWN
    assert lib.get("A").terminal_state is None
    assert index.read_bytes() == before
    lib.save()
    assert all("terminal_state" not in row for row in json.loads(index.read_text())["papers"].values())
    lib.get("A").abstract = "New observation touches one record."
    lib.save()
    persisted = json.loads(index.read_text())["papers"]
    assert persisted["A"]["terminal_state"]["policy"] == "unknown"
    assert "terminal_state" not in persisted["B"]
    # Nested edits must also count as touches, without rewriting any other record.
    lib.get("B").authors.append("Curie")
    lib.save()
    assert json.loads(index.read_text())["papers"]["B"]["terminal_state"]["policy"] == "unknown"


def test_ordinary_status_reset_never_clears_legacy_unknown_policy(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    paper = Paper(key="K", download_status="failed")
    paper.download_status = "pending"
    assert paper.acquisition_state.policy == v2.Policy.UNKNOWN
    assert paper.acquisition_state.retrieval == v2.Retrieval.ELIGIBLE


@pytest.mark.parametrize("observation", [None, {}])
def test_metadata_merge_preserves_dimensions_without_observation(monkeypatch, tmp_path, observation):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    state = v2.TerminalState(retrieval=v2.Retrieval.ACQUIRED, policy=v2.Policy.EXCLUDED,
                             policy_reasons=("owner_excluded",), merge=v2.Merge.ALIAS,
                             alias_key="canonical", quarantine=True, cause="acquired")
    paper.terminal_state = state
    incoming = {"doi": paper.doi, "abstract": "Fresh metadata"}
    if observation is not None:
        incoming["terminal_state"] = observation
    merged, _ = lib.upsert(incoming)
    assert merged is paper and merged.acquisition_state == state
    lib.save()
    assert Library(tmp_path).get(paper.key).acquisition_state == state


def test_merge_can_fill_observed_state_on_an_old_record(monkeypatch, tmp_path):
    monkeypatch.delenv("PAPERVAULT_TERMINAL_V2", raising=False)
    lib = Library(tmp_path)
    paper = add(lib)
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    observed = {"retrieval": "eligible", "policy": "excluded", "policy_reasons": ["owner_excluded"]}
    lib.upsert({"doi": paper.doi, "terminal_state": observed})
    lib.save()
    assert Library(tmp_path).get(paper.key).acquisition_state.policy == v2.Policy.EXCLUDED


@pytest.mark.parametrize("block", ["quarantine", "alias", "policy", "retrieval", "identity", "lease", "budget"])
def test_flag_on_router_uses_all_admission_inputs(monkeypatch, tmp_path, block):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    state = v2.TerminalState(retrieval=v2.Retrieval.ELIGIBLE, policy=v2.Policy.CLEAR)
    options = {}
    if block == "quarantine":
        state = replace(state, quarantine=True)
    elif block == "alias":
        state = replace(state, merge=v2.Merge.ALIAS, alias_key="canonical")
    elif block == "policy":
        state = replace(state, policy=v2.Policy.UNKNOWN)
    elif block == "retrieval":
        state = replace(state, retrieval=v2.Retrieval.ACQUIRED)
    else:
        options[f"{block}_provider"] = lambda paper: False
    paper.terminal_state = state
    before = paper.model_dump()
    assert classify(paper, lib, **options) == "TERMINAL"
    assert paper.model_dump() == before


def test_flag_on_admits_pending_and_preserves_extract_routing(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "yes")
    lib = Library(tmp_path)
    paper = add(lib)
    assert classify(paper, lib) == "DOWNLOAD"
    lib.pdf_path(paper.key).write_bytes(b"%PDF-1.4 fake")
    paper.download_status = "ok"
    assert classify(paper, lib) == "EXTRACT"


def test_download_boundary_rechecks_current_policy_and_providers(monkeypatch, tmp_path):
    from papervault.library.download import download_paper
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    assert classify(paper, lib) == "DOWNLOAD"
    # The queue's earlier decision is stale; dispatch must deny before any upstream request.
    paper.terminal_state = v2.TerminalState(retrieval=v2.Retrieval.ELIGIBLE,
                                           policy=v2.Policy.EXCLUDED,
                                           policy_reasons=("owner_excluded",))
    assert download_paper(paper, lib) is False
    assert paper.download_status == "pending" and not lib.has_pdf(paper.key)
    paper.terminal_state = replace(paper.terminal_state, policy=v2.Policy.CLEAR, policy_reasons=())
    assert download_paper(paper, lib, budget_provider=lambda paper: False) is False
    assert download_paper(paper, lib, lease_provider=lambda paper: False) is False
    assert paper.download_status == "pending"


async def test_queue_rechecks_after_waiting_for_network_slot(monkeypatch, tmp_path):
    from papervault.library import download
    from papervault.library.services import concurrency, download_queue
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    sent = []
    monkeypatch.setattr(download, "_STRATEGIES", [("sample", lambda paper: sent.append(paper.key))])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    waiting = asyncio.Event()

    class Slot(asyncio.Semaphore):
        async def acquire(self):
            waiting.set()
            return await super().acquire()

    slot = Slot(0)
    monkeypatch.setattr(concurrency, "network_sem", slot)
    queue = download_queue.DownloadQueue(lib)
    task = asyncio.create_task(queue._process_one(paper.key))
    await asyncio.wait_for(waiting.wait(), timeout=2)
    paper.terminal_state = replace(paper.acquisition_state, quarantine=True)
    slot.release()
    await asyncio.wait_for(task, timeout=2)
    assert sent == []
    assert paper.download_status == "pending"


def test_override_round_trip_retains_scope_reason_revision_and_key(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    paper.terminal_state = replace(
        paper.acquisition_state, policy=v2.Policy.OVERRIDE, policy_reasons=("off_domain",),
        required_revision=3,
        override=v2.PolicyOverride(scope=("acquire",), reasons=("off_domain",), revision=3, key=paper.key))
    lib.save()
    restored = Library(tmp_path).get(paper.key)
    assert restored.acquisition_state == paper.acquisition_state
    assert classify(restored, lib) == "DOWNLOAD"
    restored.terminal_state = replace(restored.acquisition_state, required_revision=4)
    assert classify(restored, lib) == "TERMINAL"


def test_observed_policy_exclusion_cannot_be_dropped_by_metadata_merge(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    lib.upsert({"doi": paper.doi, "terminal_state": {
        "policy": "excluded", "policy_reasons": ["owner_excluded"]}})
    assert classify(paper, lib) == "TERMINAL"
    assert paper.acquisition_state.retrieval == v2.Retrieval.ELIGIBLE
    assert paper.acquisition_state.policy_reasons == ("owner_excluded",)
    lib.upsert({"doi": paper.doi, "terminal_state": {"policy": "clear", "policy_reasons": []}})
    assert classify(paper, lib) == "TERMINAL"
    lib.save()
    assert Library(tmp_path).get(paper.key).acquisition_state.policy == v2.Policy.EXCLUDED


def test_flag_off_status_write_keeps_preexisting_dimensions_consistent(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "0")
    paper.download_status = "ok"
    lib.save()
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    restored = Library(tmp_path).get(paper.key)
    assert restored.acquisition_state.retrieval == v2.Retrieval.ACQUIRED
    assert classify(restored, lib) == "TERMINAL"


def test_merging_grants_intersects_scope_and_never_amplifies(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    paper.terminal_state = replace(paper.acquisition_state, policy=v2.Policy.OVERRIDE,
        policy_reasons=("off_domain",), required_revision=3,
        override=v2.PolicyOverride(scope=("acquire", "serve"), reasons=("off_domain",), revision=4, key=paper.key))
    lib.upsert({"doi": paper.doi, "terminal_state": {
        "override": {"scope": ["serve"], "reasons": ["off_domain"], "revision": 3, "key": paper.key}}})
    assert paper.acquisition_state.override.scope == ("serve",)
    assert paper.acquisition_state.override.revision == 3
    assert classify(paper, lib) == "TERMINAL"


def test_state_observations_never_clear_legacy_unknown_policy(monkeypatch, tmp_path):
    monkeypatch.delenv("PAPERVAULT_TERMINAL_V2", raising=False)
    lib = Library(tmp_path)
    paper = add(lib, download_status="failed")
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib.upsert({"doi": paper.doi, "terminal_state": {"policy": "clear"}})
    assert paper.acquisition_state.policy == v2.Policy.UNKNOWN


def test_merge_retains_observed_cause_without_relaxing_legacy_policy(monkeypatch, tmp_path):
    monkeypatch.delenv("PAPERVAULT_TERMINAL_V2", raising=False)
    lib = Library(tmp_path)
    paper = add(lib, download_status="failed")
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib.upsert({"doi": paper.doi, "terminal_state": {
        "retrieval": "exhausted", "policy": "clear", "cause": "retrieval_exhausted"}})
    assert paper.acquisition_state.policy == v2.Policy.UNKNOWN
    assert paper.acquisition_state.cause == "retrieval_exhausted"


@pytest.mark.parametrize("observation,expected_scope", [
    ({"scope": ["acquire"]}, ("acquire",)), ({}, ("acquire", "serve")), (None, None),
])
def test_partial_grant_observation_preserves_unobserved_fields(monkeypatch, tmp_path, observation, expected_scope):
    monkeypatch.setenv("PAPERVAULT_TERMINAL_V2", "1")
    lib = Library(tmp_path)
    paper = add(lib)
    paper.terminal_state = replace(paper.acquisition_state, policy=v2.Policy.OVERRIDE,
        policy_reasons=("owner_excluded",), required_revision=3,
        override=v2.PolicyOverride(scope=("acquire", "serve"), reasons=("owner_excluded",), revision=3, key=paper.key))
    lib.upsert({"doi": paper.doi, "terminal_state": {"override": observation}})
    grant = paper.acquisition_state.override
    if expected_scope is None:
        assert grant is None and classify(paper, lib) == "TERMINAL"
    else:
        assert (grant.scope, grant.reasons, grant.revision, grant.key) == (
            expected_scope, ("owner_excluded",), 3, paper.key)
        assert classify(paper, lib) == "DOWNLOAD"
