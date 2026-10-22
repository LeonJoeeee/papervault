"""Offline operator commands: explicit catalog targets and guarded reducer inputs."""

from contextlib import contextmanager
import json

from click.testing import CliRunner
import pytest

from papervault.knowledge.cli import cli


COMPLETION = "a" * 32


@pytest.fixture
def catalog(monkeypatch):
    """Replace only catalog/connection boundaries; no configured service is used."""
    import psycopg
    import papervault.store.knowledge as knowledge
    import papervault.store.repository as repository
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    class Connection:
        @contextmanager
        def transaction(self):
            yield

    class Catalog:
        def __init__(self):
            self.calls = []
            self.connections = []
            self.repositories = []

        def command(self, action, operation_id, payload):
            self.calls.append((action, operation_id, payload))
            if action == "acquire_maintenance_lease":
                return {"resource": "extract_gc", "lease_id": "lease-1", "owner": "operator"}
            return {"action": action, "operation_id": operation_id, **payload}

        def _rows(self, table):
            assert table == "extract_retention"
            return []

        def get_authority_state(self):
            return {"authority": "catalog"}

        def get_active_maintenance_window(self):
            return None

        @staticmethod
        def _json(value):
            return value

    boundary = Catalog()
    boundary.connection = Connection()

    @contextmanager
    def connect(dsn, **kwargs):
        boundary.connections.append((dsn, kwargs))
        yield boundary.connection

    def make_repository(connection, **kwargs):
        assert connection is boundary.connection
        boundary.repositories.append(kwargs)
        return boundary

    monkeypatch.setenv("EXTRACT_TEST_DSN", "postgresql://offline/explicit_test")
    monkeypatch.setattr(psycopg, "connect", connect)
    monkeypatch.setattr(repository, "PostgresCatalogRepository", make_repository)
    monkeypatch.setattr(knowledge, "KnowledgeRepository", lambda repo: repo)
    return boundary


def target(tmp_path):
    return ["--dsn-env", "EXTRACT_TEST_DSN", "--catalog-schema", "catalog_test",
            "--vault", str(tmp_path), "--operation-id", "operator-op"]


def retention():
    return ["extract-retention", COMPLETION, "--disposable", "--expected-revision", "7",
            "--reason", "verified supersession", "--operator-nonce", "review-2026", "--actor", "leon"]


def test_retention_plan_never_writes_and_preserves_reviewed_inputs(catalog, tmp_path):
    result = CliRunner().invoke(cli, retention() + target(tmp_path))
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["apply"] is False
    assert plan["payload"] == {
        "completion_id": COMPLETION, "disposable": True, "expected_revision": 7,
        "reason": "verified supersession", "operator_nonce": "review-2026", "actor": "leon",
        "vault": str(tmp_path),
    }
    assert catalog.calls == []
    assert catalog.connections == [("postgresql://offline/explicit_test",
                                    {"autocommit": True, "connect_timeout": 5})]
    assert catalog.repositories == [{"schema": "catalog_test", "read_only": True}]


def test_retention_apply_uses_replayable_revision_guarded_command(catalog, tmp_path):
    result = CliRunner().invoke(cli, retention() + target(tmp_path) + ["--apply"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)["result"]
    assert payload["action"] == "set_extract_retention"
    assert payload["operation_id"] == "operator-op"
    assert payload["expected_revision"] == 7
    assert payload["reason"] == "verified supersession"
    assert payload["operator_nonce"] == "review-2026"
    assert payload["actor"] == "leon"
    assert catalog.repositories == [{"schema": "catalog_test", "read_only": False}]


def test_retention_can_restore_protection(catalog, tmp_path):
    args = ["--protected" if item == "--disposable" else item for item in retention()]
    result = CliRunner().invoke(cli, args + target(tmp_path) + ["--apply"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["result"]["disposable"] is False


@pytest.mark.parametrize("missing", ["--reason", "--operator-nonce", "--expected-revision", "--actor"])
def test_retention_requires_review_evidence_before_connecting(catalog, tmp_path, missing):
    args = retention()
    index = args.index(missing)
    del args[index:index + 2]
    result = CliRunner().invoke(cli, args + target(tmp_path) + ["--apply"])
    assert result.exit_code == 2
    assert missing in result.output
    assert catalog.connections == []


@pytest.mark.parametrize("kind", ["reader", "job", "recovery"])
def test_pin_preserves_owner_kind_and_bounded_reader_ttl(catalog, tmp_path, kind):
    result = CliRunner().invoke(cli, ["extract-pin", COMPLETION, "--kind", kind,
        "--owner", "reader-1", "--ttl-seconds", "600", "--actor", "leon",
        *target(tmp_path), "--apply"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)["result"]
    assert payload["action"] == "pin_extract"
    assert payload["completion_id"] == COMPLETION
    assert payload["owner"] == "reader-1"
    assert payload["kind"] == kind
    assert payload["ttl_seconds"] == 600
    assert payload["actor"] == "leon"


def test_release_pin_preserves_reviewed_disposition_and_revision(catalog, tmp_path):
    result = CliRunner().invoke(cli, ["release-extract-pin", "pin-1", "--owner", "worker-1",
        "--expected-revision", "3", "--disposition", "reviewed_recovery_disposition",
        "--reason", "consumer verified complete", "--operator-nonce", "release-review",
        "--actor", "leon", *target(tmp_path), "--apply"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["result"] == {
        "action": "release_extract_pin", "operation_id": "operator-op", "pin_id": "pin-1",
        "owner": "worker-1", "expected_revision": 3,
        "disposition": "reviewed_recovery_disposition", "reason": "consumer verified complete",
        "operator_nonce": "release-review", "actor": "leon",
    }


def test_gc_defaults_to_plan_without_acquiring_lease(catalog, tmp_path):
    result = CliRunner().invoke(cli, ["extract-gc", "--completion-id", COMPLETION,
        *target(tmp_path)])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["apply"] is False
    assert plan["payload"] == {"completion_ids": [COMPLETION]}
    assert plan["gc"]["retention_days"] == 90
    assert catalog.calls == []


def test_gc_uses_existing_leased_collector_and_releases_lease(catalog, tmp_path, monkeypatch):
    import papervault.store.derivative_maintenance as maintenance

    def collect(k, vault, completion_id, lease):
        assert k is catalog
        assert str(vault) == str(tmp_path)
        assert completion_id == COMPLETION
        assert lease == {"resource": "extract_gc", "lease_id": "lease-1", "owner": "operator"}
        return {"completion_id": completion_id, "state": "deleted"}

    monkeypatch.setattr(maintenance, "collect_extract", collect)
    result = CliRunner().invoke(cli, ["extract-gc", "--completion-id", COMPLETION,
        "--owner", "operator", "--window-id", "window-1", "--lease-seconds", "120",
        *target(tmp_path), "--apply"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["result"] == [{"completion_id": COMPLETION, "state": "deleted"}]
    assert catalog.calls[0][0] == "acquire_maintenance_lease"
    assert catalog.calls[0][2] == {"resource": "extract_gc", "owner": "operator",
                                   "window_id": "window-1", "lease_seconds": 120}
    assert catalog.calls[-1][0] == "release_maintenance_lease"


def test_gc_apply_refuses_missing_window_before_lease(catalog, tmp_path):
    result = CliRunner().invoke(cli, ["extract-gc", "--completion-id", COMPLETION,
        "--owner", "operator", *target(tmp_path), "--apply"])
    assert result.exit_code != 0
    assert "window" in str(result.exception).lower() or "window" in result.output.lower()
    assert catalog.calls == []


@pytest.mark.parametrize("missing", ["--dsn-env", "--catalog-schema", "--vault", "--operation-id"])
def test_missing_explicit_target_is_rejected_before_connection(catalog, tmp_path, missing):
    args = target(tmp_path)
    index = args.index(missing)
    del args[index:index + 2]
    result = CliRunner().invoke(cli, ["extract-gc", "--completion-id", COMPLETION, *args])
    assert result.exit_code == 2
    assert missing in result.output
    assert catalog.connections == []


def test_missing_dsn_environment_does_not_fall_back(catalog, tmp_path, monkeypatch):
    monkeypatch.delenv("EXTRACT_TEST_DSN")
    result = CliRunner().invoke(cli, retention() + target(tmp_path))
    assert result.exit_code != 0
    assert "EXTRACT_TEST_DSN" in result.output
    assert catalog.connections == []


@pytest.mark.parametrize("seconds", ["0", "7201"])
def test_reader_pin_rejects_unbounded_lifetime_before_connection(catalog, tmp_path, seconds):
    result = CliRunner().invoke(cli, ["extract-pin", COMPLETION, "--kind", "reader",
        "--owner", "reader-1", "--ttl-seconds", seconds, "--actor", "leon",
        *target(tmp_path), "--apply"])
    assert result.exit_code == 2
    assert catalog.connections == []


def test_catalog_revision_refusal_is_reported_without_success(catalog, tmp_path):
    from papervault.store.knowledge import ClaimRejected

    def refuse(action, operation_id, payload):
        raise ClaimRejected("revision changed")

    catalog.command = refuse
    result = CliRunner().invoke(cli, retention() + target(tmp_path) + ["--apply"])
    assert result.exit_code == 1
    assert "revision changed" in result.output


def test_pin_plan_does_not_acquire_a_pin(catalog, tmp_path):
    result = CliRunner().invoke(cli, ["extract-pin", COMPLETION, "--kind", "reader",
        "--owner", "reader-1", "--actor", "leon", *target(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["payload"]["ttl_seconds"] == 300
    assert catalog.calls == []
