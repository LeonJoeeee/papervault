"""Standalone replay CLI resolves local paths before backend preparation."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


def test_replay_cli_resolves_package_and_root_before_physical_factory(monkeypatch, tmp_path, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/rebuild_derivatives.py"
    spec = importlib.util.spec_from_file_location("rebuild_derivatives_under_test", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "package").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"physical_replay": True}))
    snapshot = ModuleType("snapshot_derivatives")
    snapshot.connections = lambda *args: {}
    runtime = ModuleType("recovery_runtime")
    prepared = []

    def factory(config, targets, package, root):
        assert package == tmp_path / "package"
        assert root == tmp_path / "output"
        prepared.append(package)
        return None, None

    runtime.physical_replay_factory = factory
    monkeypatch.setitem(sys.modules, "snapshot_derivatives", snapshot)
    monkeypatch.setitem(sys.modules, "recovery_runtime", runtime)

    async def replay(package, root, **kwargs):
        assert package == prepared[0]
        return {"apply": True}

    monkeypatch.setattr("papervault.store.recovery_replay.replay_recovery_set", replay)
    assert module.main(["--mode", "replay", "--package", "package", "--root", "output",
        "--generation", "new", "--source-generation", "old", "--config", "config.json", "--apply"]) == 0
    assert json.loads(capsys.readouterr().out) == {"apply": True}
