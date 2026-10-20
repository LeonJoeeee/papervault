"""Actual intake commands and interrupted operator recovery on dedicated PG."""

from types import SimpleNamespace

from psycopg import sql
import pytest

from papervault.store.repository import AuditEvent, Record


@pytest.fixture
def intake_catalog(repo, tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake
    from papervault.store.knowledge import KnowledgeRepository

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    repo.connection.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
    knowledge = KnowledgeRepository(repo)
    knowledge.command("create_generation", "active", {"generation_id": "active", "candidate": False})
    return CatalogIntake(knowledge, ImmutableInputStore(tmp_path / "inputs"), tmp_path,
        recipe={"schema_version": 1, "output_contract": {"fulltext": {"required": ["graph"], "not_applicable": []},
                "metadata": {"required": ["metadata"], "not_applicable": ["graph"]}}})


def test_record_registration_capture_and_changed_bytes_make_linked_jobs(intake_catalog, tmp_path):
    intake = intake_catalog
    repo = intake.repository.repository
    path = tmp_path / "A.md"
    path.write_text("First source evidence.")
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "A", "md_path": "A.md"}),
                expected_revision=0, audit=AuditEvent("test", "register"))
    assert intake.scan()["captured"] == 1
    first = intake.repository.list_jobs()[0]
    path.write_text("Replacement source evidence.")
    intake.scan()
    jobs = intake.repository.list_jobs()
    assert len(jobs) == 2
    second = next(job for job in jobs if job["job_id"] != first["job_id"])
    assert second["consumed_input_revision"] != first["consumed_input_revision"]
    assert second["predecessor_job_id"] == first["job_id"]
    unit = intake.repository.get_unit_for_record("A")
    inventory = intake.repository.get_current_inventory(unit["unit_id"], "active")
    assert inventory["required_units"] == [unit["unit_id"]]


def test_interrupted_force_recovers_frozen_candidate_without_active_dispatch(intake_catalog, tmp_path, monkeypatch):
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    class Words:
        def encode(self, text):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    intake = intake_catalog
    path = tmp_path / "book.md"
    path.write_text("# Original\none two three four five six seven eight nine ten")
    intake.capture_document("textbook", "textbook:Example2020", path, tokenizer=Words(), max_tokens=8)
    path.write_text("# Revised\nreplacement source words with a different evidence revision")
    command = intake._command

    def interrupted(action, payload):
        if action == "set_requirement":
            raise RuntimeError("simulated process death after durable inventory intent")
        return command(action, payload)

    monkeypatch.setattr(intake, "_command", interrupted)
    with pytest.raises(RuntimeError, match="process death"):
        intake.capture_document("textbook", "textbook:Example2020", path, tokenizer=Words(), max_tokens=8, force=True)
    monkeypatch.setattr(intake, "_command", command)
    service = CatalogService(intake.repository, intake, SimpleNamespace())
    service.recover_intake()
    candidates = intake.repository._rows("generations", state="candidate")
    assert len(candidates) == 1
    candidate_id = candidates[0]["generation_id"]
    jobs = intake.repository.list_jobs()
    candidate_jobs = [job for job in jobs if job["generation_id"] == candidate_id]
    assert candidate_jobs
    assert intake.repository.list_admission_jobs() == []  # Candidates wait for the monthly window.
    before = len(jobs)
    service.recover_intake()
    assert len(intake.repository.list_jobs()) == before
    assert intake.repository.get_active_generation()["generation_id"] == "active"


@pytest.mark.asyncio
async def test_verified_metadata_candidate_can_promote_without_claiming_fulltext(intake_catalog):
    from datetime import timedelta
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_intake import effective_distill_recipe
    from papervault.store.coverage import query_coverage

    intake = intake_catalog
    intake.recipe = effective_distill_recipe()
    k = intake.repository
    k.repository.upsert(Record.from_payload("A", {"key": "A", "title": "metadata only"}),
                        expected_revision=0, audit=AuditEvent("test", "register"))
    intake.scan()
    state = k.get_authority_state()
    k.command("fork_generation", "fork", {"generation_id": "candidate", "source_generation_id": "active",
        "expected_manifest_revision": k.get_generation("active")["manifest_revision"],
        "expected_requirements_revision": state["requirements_revision"]})
    stamp = k.connection.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
    k.command("begin_maintenance_window", "window", {"window_id": "monthly", "kind": "monthly_reboot",
        "starts_at": (stamp-timedelta(seconds=1)).isoformat(), "ends_at": (stamp+timedelta(hours=1)).isoformat(),
        "timezone": "Asia/Shanghai", "operator_nonce": "test", "expected_revision": state["maintenance_revision"]})

    class NoExternalEffects:
        async def lookup_submission(self, submission_id):
            return None

    scheduler = CatalogScheduler(k, NoExternalEffects(), intake.input_store, max_admissions=10,
                                 breaker=SimpleNamespace(allows=lambda: True))
    result = await scheduler.run_round()
    assert result["published"] == 2
    state = k.get_authority_state()
    k.command("promote_generation", "promote", {"generation_id": "candidate",
        "expected_manifest_revision": k.get_generation("candidate")["manifest_revision"],
        "expected_requirements_revision": state["requirements_revision"],
        "expected_serving_revision": state["serving_revision"]})
    assert k.get_active_generation()["generation_id"] == "candidate"
    record = query_coverage(k)["records"][0]
    assert record["tier_coverage"]["metadata"] is True
    assert record["readable"] is record["indexed"] is False
