"""Maintenance guards and read-only correction exports through real CLI parsing."""

import json

from papervault.library import Library, cli


def test_event_refuses_running_service_before_constructing_store(tmp_path, monkeypatch, capsys):
    from papervault import ops_guards
    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path / "absent"))
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: ["papervault.service"])
    assert cli.main(["correction-event", "credential_arrived", "--source", "ads",
                     "--cap", "2", "--apply"]) == 2
    assert "running" in capsys.readouterr().err
    assert not (tmp_path / "absent").exists()


def test_read_only_report_and_manifest_exports_create_no_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path))
    before = list(tmp_path.iterdir())
    assert cli.main(["correction-report", "--month", "2026-10"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["conflicts"]["needs_human"] == 0
    assert cli.main(["correction-report", "--kind", "manifests"]) == 0
    assert json.loads(capsys.readouterr().out) == {}
    assert list(tmp_path.iterdir()) == before


def test_empty_capped_monthly_cli_records_report_without_acquisition(tmp_path, monkeypatch, capsys):
    from papervault import ops_guards
    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path))
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: [])
    assert cli.main(["correction-sweep", "--per-cohort-cap", "1", "--total-cap", "2", "--apply"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selected"] == 0 and result["total_cap"] == 2
    assert result["report"]["aggregate"]["sweeps"] == 1
    assert Library(tmp_path).all_papers() == []


def test_identifier_notification_requires_keys_before_writes(tmp_path, monkeypatch, capsys):
    from papervault import ops_guards
    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path / "absent"))
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: [])
    assert cli.main(["correction-event", "identifiers_arrived", "--source", "ads",
                     "--cap", "1", "--apply"]) == 2
    assert "keys" in capsys.readouterr().err
    assert not (tmp_path / "absent").exists()
