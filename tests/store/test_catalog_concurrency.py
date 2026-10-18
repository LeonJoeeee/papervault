"""Two OS processes, independent working sets/connections, bounded join/deadlock checks."""
import multiprocessing
import os

import psycopg
import pytest

from test_postgres import count
from papervault.store.repository import AuditEvent, Record, PostgresCatalogRepository, PendingJob


def _writer(dsn, schema, barrier, output, number, overlap, root):
    from papervault.library.store import Library
    from papervault.store.persistence import CatalogConflict

    os.environ.update(PAPERVAULT_CATALOG_AUTHORITY="postgres", PAPERVAULT_CATALOG_DSN=dsn,
                      PAPERVAULT_CATALOG_SCHEMA=schema, PAPERVAULT_CATALOG_ACTOR=f"writer-{number}")
    library = Library(root)
    papers = library._papers
    key = "A" if overlap else ("B" if number == 1 else "C")
    papers[key].title = f"writer {number}"
    barrier.wait(timeout=15)
    try:
        library.save(force=True)
    except CatalogConflict as exc:
        output.put(("conflict", exc.keys))
    else:
        output.put(("saved", key))
    finally:
        library.close()


def _completion(dsn, schema, barrier, output, number):
    from papervault.store.repository import StaleCompletion

    with psycopg.connect(dsn, autocommit=True) as connection:
        repository = PostgresCatalogRepository(connection, schema=schema)
        barrier.wait(timeout=15)
        try:
            repository.complete_job("job", attempt=1, lease_id="lease", outcome="completed",
                expected_attempt=1, expected_lease="lease", audit=AuditEvent(str(number), "completion"))
        except StaleCompletion:
            output.put(("stale", number))
        else:
            output.put(("completed", number))


def _run_pair(repo, target, *args):
    context = multiprocessing.get_context("spawn")
    barrier, output = context.Barrier(2), context.Queue()
    processes = [context.Process(target=target, args=(os.environ["PAPERVAULT_CATALOG_TEST_DSN"],
                  repo.schema, barrier, output, number, *args)) for number in (1, 2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=25)
            assert not process.is_alive(), "two catalog writers deadlocked"
            assert process.exitcode == 0
        return [output.get(timeout=3) for _ in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        output.close()


@pytest.mark.parametrize("overlap", [False, True])
def test_two_library_writers_preserve_rows_and_report_stale_overlap(repo, overlap, tmp_path):
    for key in "ABC":
        repo.upsert(Record.from_payload(key, {"key": key, "title": "original"}),
                    expected_revision=0, audit=AuditEvent("test", "seed"))
    before = count(repo, "audit")
    results = _run_pair(repo, _writer, overlap, str(tmp_path))
    if overlap:
        assert sorted(result[0] for result in results) == ["conflict", "saved"]
        assert repo.get("A").revision == 2
        assert repo.get("A").title in {"writer 1", "writer 2"}
        assert repo.get("B").title == repo.get("C").title == "original"
        assert count(repo, "audit") == before + 1
    else:
        assert all(result[0] == "saved" for result in results)
        assert repo.get("B").title == "writer 1"
        assert repo.get("C").title == "writer 2"
        assert repo.get("A").title == "original"
        assert count(repo, "audit") == before + 2


def test_two_completers_append_one_event_and_reject_loser(repo):
    repo.upsert(Record.from_payload("A", {"key": "A"}), expected_revision=0,
        audit=AuditEvent("test", "queue", pending_jobs=(PendingJob("job", "reopen", "A",
                         state="dispatched", attempts=1, lease_id="lease"),)))
    before = count(repo, "audit")
    assert sorted(r[0] for r in _run_pair(repo, _completion)) == ["completed", "stale"]
    assert count(repo, "audit") == before + 1
