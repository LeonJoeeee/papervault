"""Explicit census production measures scoped backends without mutating them."""
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from papervault.store.scheduler_capacity import measured_backend_bytes


def producer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("measure_derivative_usage")


def configuration(tmp_path, *, metadata_sample=False):
    vault, knowledge, retained = (tmp_path / name for name in ("vault", "knowledge", "retained"))
    for path in (vault, knowledge, retained):
        path.mkdir()
    (retained / "snapshot.bin").write_bytes(b"1234567")
    value = {"vault": str(vault), "knowledge_root": str(knowledge),
             "catalog": {"dsn_env": "EXPLICIT_TEST_DSN", "schema": "catalog_fixture"},
             "extra_derivative_roots": [str(retained)], "metadata_sample": metadata_sample}
    if not metadata_sample:
        value.update(postgres={"dsn_env": "BACKEND_TEST_DSN"}, neo4j={"uri": "bolt://explicit:7687"})
    return value


class Catalog:
    def __init__(self, counts=(0, 0)):
        from psycopg import sql
        self.counts = iter(counts)
        self._table = lambda table: sql.Identifier("catalog_fixture", table)
        self.connection = self

    def execute(self, query, parameters=None):
        return SimpleNamespace(fetchone=lambda: {"count": next(self.counts)})


class Backend:
    def __init__(self, size):
        self.size = size

    def estimated_bytes(self):
        return self.size


def test_census_counts_backend_and_retained_roots_but_excludes_catalog(tmp_path, monkeypatch):
    module = producer(monkeypatch)
    config = configuration(tmp_path)
    config["extra_derivative_roots"].append(config["extra_derivative_roots"][0])
    census = module.measure(config, catalog=Catalog(counts=()), postgres=Backend(11), neo4j=Backend(13))
    assert census["backend_bytes"] == 31
    assert census["scope"] == {"vault_root": config["vault"], "knowledge_root": config["knowledge_root"]}
    assert census["includes_catalog"] is False
    assert datetime.fromisoformat(census["measured_at"]).tzinfo == timezone.utc


@pytest.mark.parametrize("field", ["catalog", "postgres", "neo4j", "extra_derivative_roots"])
def test_incomplete_configuration_refuses_before_connecting(tmp_path, monkeypatch, field):
    module = producer(monkeypatch)
    config = configuration(tmp_path)
    config.pop(field)
    with pytest.raises(ValueError, match="explicit|complete|config|roots"):
        module.validate_configuration(config)


@pytest.mark.parametrize("fault", ["missing", "symlink", "linked-file", "source-alias"])
def test_retained_inventory_refuses_missing_roots_and_aliases(tmp_path, monkeypatch, fault):
    module = producer(monkeypatch)
    config = configuration(tmp_path)
    retained = Path(config["extra_derivative_roots"][0])
    if fault == "missing":
        config["extra_derivative_roots"] = [str(tmp_path / "absent")]
    elif fault == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(retained, target_is_directory=True)
        config["extra_derivative_roots"] = [str(alias)]
    elif fault == "linked-file":
        (retained / "alias").hardlink_to(retained / "snapshot.bin")
    else:
        config["extra_derivative_roots"] = [config["vault"]]
    with pytest.raises(ValueError):
        module.measure(config, catalog=Catalog(), postgres=Backend(1), neo4j=Backend(1))


def test_metadata_sample_can_omit_backends_only_after_catalog_proof(tmp_path, monkeypatch):
    module = producer(monkeypatch)
    config = configuration(tmp_path, metadata_sample=True)
    census = module.measure(config, catalog=Catalog())
    assert census["backend_bytes"] == 7
    with pytest.raises(ValueError, match="metadata"):
        module.measure(config, catalog=Catalog(counts=(1, 0)))
    with pytest.raises(ValueError, match="metadata"):
        module.measure(config, catalog=Catalog(counts=(0, 1)))


def test_complete_config_cannot_silently_omit_an_adapter(tmp_path, monkeypatch):
    module = producer(monkeypatch)
    config = configuration(tmp_path)
    with pytest.raises(ValueError, match="complete|adapter"):
        module.measure(config, catalog=Catalog(), postgres=Backend(11))


def test_atomic_applied_census_is_protected_and_accepted_by_runtime(tmp_path, monkeypatch):
    module = producer(monkeypatch)
    config = configuration(tmp_path)
    census = module.measure(config, catalog=Catalog(), postgres=Backend(11), neo4j=Backend(13))
    output = tmp_path / "usage.json"
    module.publish(output, census)
    assert output.stat().st_mode & 0o777 == 0o600
    assert measured_backend_bytes(output, vault_root=Path(config["vault"]),
                                  knowledge_root=Path(config["knowledge_root"])) == 31
    assert sorted(path.name for path in tmp_path.iterdir()) == ["knowledge", "retained", "usage.json", "vault"]


def test_publish_refuses_output_symlink_without_changing_target(tmp_path, monkeypatch):
    module = producer(monkeypatch)
    config = configuration(tmp_path)
    census = module.measure(config, catalog=Catalog(), postgres=Backend(11), neo4j=Backend(13))
    protected = tmp_path / "protected"
    protected.write_bytes(b"protected")
    output = tmp_path / "usage.json"
    output.symlink_to(protected)
    with pytest.raises(ValueError):
        module.publish(output, census)
    assert protected.read_bytes() == b"protected"


def test_cli_preview_measures_without_creating_output(tmp_path, monkeypatch, capsys):
    module = producer(monkeypatch)
    config = configuration(tmp_path, metadata_sample=True)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(module, "connections", lambda config, stack: {"catalog": {
        "connection": object(), "schema": "catalog_fixture"}})
    monkeypatch.setattr(module, "measure", lambda config, **kwargs: {
        "backend_bytes": 7, "scope": {"vault_root": config["vault"], "knowledge_root": config["knowledge_root"]},
        "measured_at": datetime.now(timezone.utc).isoformat(), "includes_catalog": False})
    # Repository construction and connections are isolated; the real CLI controls
    # preview/apply and the real publisher owns the actual file effects.
    from papervault.store import repository
    monkeypatch.setattr(repository, "PostgresCatalogRepository", lambda *args, **kwargs: Catalog())
    output = tmp_path / "usage.json"
    assert module.main(["--config", str(path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["apply"] is False
    assert not output.exists()
    assert module.main(["--config", str(path), "--output", str(output), "--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["apply"] is True
    assert measured_backend_bytes(output, vault_root=Path(config["vault"]),
                                  knowledge_root=Path(config["knowledge_root"])) == 7


def test_publish_never_overwrites_an_unrecognized_regular_asset(tmp_path, monkeypatch):
    module = producer(monkeypatch)
    config = configuration(tmp_path)
    census = module.measure(config, catalog=Catalog(), postgres=Backend(11), neo4j=Backend(13))
    output = tmp_path / "source.pdf"
    output.write_bytes(b"protected current source")
    with pytest.raises(ValueError):
        module.publish(output, census)
    assert output.read_bytes() == b"protected current source"
