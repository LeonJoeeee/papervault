"""Lossless, deterministic atomic export using the shadow comparison contract."""
import json

import pytest

from papervault.store.repository import AuditEvent, Record


def test_export_verified_against_catalog_and_deterministic(repo, tmp_path):
    from papervault.store.exporter import export_catalog

    for key in ("B", "A"):
        repo.upsert(Record.from_payload(key, {"key": key, "future": {"keep": "é"}}),
                    expected_revision=0, audit=AuditEvent("test", "seed"))
    output = tmp_path / "index.json"
    report = export_catalog(repo, output, verify=True)
    assert report["difference_rate"] == 0
    assert report["json_count"] == report["postgres_count"] == 2
    first = output.read_bytes()
    assert list(json.loads(first)["papers"]) == ["A", "B"]
    assert json.loads(first)["papers"]["A"]["future"] == {"keep": "é"}
    export_catalog(repo, output, verify=True)
    assert output.read_bytes() == first
    assert not list(tmp_path.glob("*.tmp"))


def test_export_invalid_catalog_refuses_before_replacing_good_file(repo, tmp_path):
    from papervault.store.exporter import export_catalog

    output = tmp_path / "index.json"
    output.write_bytes(b"last good export")
    from psycopg import sql
    repo.upsert(Record.from_payload("B", {"key": "B"}), expected_revision=0,
                audit=AuditEvent("test", "seed"))
    repo.upsert(Record.from_payload("A", {"key": "A",
        "terminal_state": {"merge": "alias", "alias_key": "B"}}),
        expected_revision=0, audit=AuditEvent("test", "alias"))
    # Simulate a corrupt projection independently of the repository contract.
    repo.connection.execute(sql.SQL("DELETE FROM {} WHERE alias_key='A'").format(
        sql.Identifier(repo.schema, "aliases")))
    with pytest.raises(ValueError):
        export_catalog(repo, output, verify=True)
    assert output.read_bytes() == b"last good export"
