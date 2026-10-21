"""Operator maintenance uses the same fenced catalog reducers as normal work."""

from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
import asyncio
import time

import pytest

from papervault.store.knowledge import ClaimRejected
from test_knowledge_postgres import knowledge as knowledge, prepare_metadata, _NoBackend, _OpenBuildGate


def maintain(k, action, payload, **kwargs):
    assert find_spec("papervault.store.derivative_maintenance"), "maintenance coordinator missing"
    from papervault.store.derivative_maintenance import maintain
    return maintain(k, action, payload, **kwargs)


def window(k, seconds=120):
    stamp = datetime.now(timezone.utc)
    return k.command("begin_maintenance_window", "window", {"window_id": "window", "kind": "monthly_reboot",
        "starts_at": (stamp-timedelta(seconds=1)).isoformat(), "ends_at": (stamp+timedelta(seconds=seconds)).isoformat(),
        "operator_nonce": "test", "expected_revision": 0})


def fork_payload(k):
    return {"generation_id": "candidate", "source_generation_id": "g",
        "expected_manifest_revision": k.get_generation("g")["manifest_revision"],
        "expected_requirements_revision": k.get_authority_state()["requirements_revision"]}


def promotion_payload(k):
    return {"generation_id": "candidate",
        "expected_manifest_revision": k.get_generation("candidate")["manifest_revision"],
        "expected_requirements_revision": k.get_authority_state()["requirements_revision"],
        "expected_serving_revision": k.get_authority_state()["serving_revision"]}


def test_dry_run_is_read_only_and_apply_requires_window(knowledge, tmp_path):
    k = knowledge
    prepare_metadata(k, tmp_path)
    before = k._rows("knowledge_operations")
    plan = maintain(k, "fork", fork_payload(k), vault=tmp_path, operation_id="fork")
    assert plan["apply"] is False and plan["action"] == "fork"
    assert k.get_generation("candidate") is None
    assert k._rows("knowledge_operations") == before
    with pytest.raises(ClaimRejected, match="window"):
        maintain(k, "fork", fork_payload(k), vault=tmp_path, operation_id="fork", apply=True,
                 window_id="absent", owner="operator")
    assert k.get_generation("candidate") is None


@pytest.mark.asyncio
async def test_rebuild_isolated_then_exact_revision_promotion(knowledge, tmp_path):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler

    k = knowledge
    store, _ = prepare_metadata(k, tmp_path)
    scheduler = CatalogScheduler(k, _NoBackend(), store, breaker=_OpenBuildGate())
    await scheduler.run_round(only_keys={"A"})
    serving = k.get_serving_manifest()
    window(k)
    common = {"vault": tmp_path, "apply": True, "window_id": "window", "owner": "operator"}
    fork = maintain(k, "fork", fork_payload(k), operation_id="fork", **common)
    assert fork["result"]["state"] == "candidate"
    assert k.get_serving_manifest() == serving
    with pytest.raises(ClaimRejected, match="incomplete"):
        maintain(k, "promote", promotion_payload(k), operation_id="incomplete", **common)
    report = await scheduler.run_round(only_keys={"A"})
    assert report["published"] == 1
    assert k.get_serving_manifest() == serving
    payload = promotion_payload(k)
    with pytest.raises(ClaimRejected, match="revision"):
        maintain(k, "promote", {**payload, "expected_serving_revision": -1}, operation_id="stale", **common)
    result = maintain(k, "promote", payload, operation_id="promote", **common)
    assert result["result"]["generation_id"] == "candidate"
    assert k.get_generation("g")["state"] == "retired"
    assert k.get_serving_manifest()["generation_id"] == "candidate"
    # Lost operator response can be replayed without a second promotion.
    assert maintain(k, "promote", payload, operation_id="promote", **common)["result"] == result["result"]


@pytest.mark.parametrize("action", ["fork", "promote", "gc"])
def test_mutation_needs_explicit_owner_and_window(knowledge, tmp_path, action):
    with pytest.raises(ValueError, match="owner|window"):
        maintain(knowledge, action, {}, vault=tmp_path, operation_id="missing", apply=True)


def test_gc_requires_explicit_bounded_identities(knowledge, tmp_path):
    window(knowledge)
    for payload in ({}, {"completion_ids": []}, {"completion_ids": ["a"]*101}):
        with pytest.raises(ValueError, match="completion"):
            maintain(knowledge, "gc", payload, vault=tmp_path, operation_id="gc", apply=True,
                     window_id="window", owner="operator")


def test_committed_fork_replays_after_window_with_conflict_checks(knowledge, tmp_path):
    from papervault.store.knowledge import OperationConflict
    k = knowledge
    prepare_metadata(k, tmp_path)
    window(k, seconds=1)
    payload = fork_payload(k)
    common = {"vault": tmp_path, "apply": True, "owner": "operator", "window_id": "window", "operation_id": "fork"}
    first = maintain(k, "fork", payload, **common)
    time.sleep(1.05)
    assert maintain(k, "fork", payload, **common) == first
    with pytest.raises(OperationConflict):
        maintain(k, "fork", {**payload, "generation_id": "different"}, **common)


@pytest.mark.asyncio
async def test_namespace_mutation_is_cancelled_at_maintenance_deadline(knowledge, tmp_path):
    from test_knowledge_postgres import prepare, claim
    k = knowledge
    job = prepare(k, "candidate", candidate=True)
    window(k, seconds=1)
    execution = claim(k, job)
    output = tmp_path / "late-output"
    with pytest.raises(TimeoutError):
        async with k.namespace_access(execution):
            await asyncio.sleep(1.5)
            output.write_bytes(b"late mutation")
    assert not output.exists()
    assert not k._namespace_owners
    with pytest.raises(ClaimRejected, match="window"):
        async with k.namespace_access(execution):
            pytest.fail("expired private namespace must stay fenced")


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['asyncio', 'postgres'])
async def test_candidate_deadline_covers_lock_wait_and_releases_recovery_lock(knowledge, catalog_test_dsn, kind):
    import psycopg
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.repository import PostgresCatalogRepository
    from test_knowledge_postgres import prepare, claim
    k = knowledge
    job = prepare(k, 'candidate', candidate=True)
    window(k, seconds=1)
    execution = claim(k, job)
    namespace = execution['namespace']
    local_lock = k._namespace_locks.setdefault(namespace, asyncio.Lock())
    with psycopg.connect(catalog_test_dsn, autocommit=True) as conn:
        other = KnowledgeRepository(PostgresCatalogRepository(conn, schema=k.repository.schema))
        remote_lock = other._session_namespace_lock(namespace)
        if kind == 'asyncio':
            await local_lock.acquire()
        else:
            remote_lock.__enter__()
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                async with k.namespace_access(execution):
                    pytest.fail('namespace should still be held by competing writer')
            assert time.monotonic() - started < 3
            assert not k._namespace_owners
            recovery_key = 'papervault:' + k.repository.schema + ':recovery'
            acquired = conn.execute('SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS locked',
                (recovery_key,)).fetchone()['locked']
            assert acquired, 'timed-out operation leaked its recovery barrier lock'
            conn.execute('SELECT pg_advisory_unlock(hashtextextended(%s, 0))', (recovery_key,))
        finally:
            if kind == 'asyncio':
                local_lock.release()
            else:
                remote_lock.__exit__(None, None, None)
