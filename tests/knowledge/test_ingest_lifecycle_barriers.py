"""Legacy ingest consumes only eligible records and vault-contained extracts."""
import json

import pytest

from papervault.knowledge.ingest.paper_library_client import PaperRecord, load_vault_index, read_paper_text
from papervault.knowledge.ingest.vault import read_extract_raw


@pytest.mark.parametrize("state", [
    {"quarantine": True}, {"policy": "excluded"}, {"policy": "unknown"},
    {"merge": "alias", "alias_key": "canonical"},
    {"merge": "canonical", "alias_key": "other"},
    {"policy": "clear", "policy_reasons": ["blocked"]},
    {"policy": "override"},
    {"policy": "override", "required_revision": 2, "override": {
        "key": "blocked", "scope": ["acquire"], "reasons": [], "revision": 1}},
    {"policy": "override", "override": {
        "key": "foreign", "scope": ["acquire"], "reasons": [], "revision": 2}},
    {"policy": "override", "override": {
        "key": "blocked", "scope": ["serve"], "reasons": [], "revision": 2}},
    {"policy": "override", "policy_reasons": ["blocked"], "override": {
        "key": "blocked", "scope": ["acquire"], "reasons": [], "revision": 2}},
])
def test_legacy_clean_index_excludes_terminal_barriers(tmp_path, state):
    terminal = {"policy": "clear", "merge": "canonical", "quarantine": False, **state}
    (tmp_path / "index.json").write_text(json.dumps({"version": 1, "papers": {
        "blocked": {"key": "blocked", "terminal_state": terminal},
        "clean": {"key": "clean"}}}))
    assert set(load_vault_index(tmp_path)) == {"clean"}
    assert set(load_vault_index(tmp_path, include_quarantined=True)) == {"clean", "blocked"}


@pytest.mark.parametrize("reader", [read_paper_text, read_extract_raw])
@pytest.mark.parametrize("field", ["md_path", "txt_path"])
@pytest.mark.parametrize("escape", ["traversal", "absolute", "symlink", "root"])
def test_legacy_extract_reader_rejects_paths_outside_resolved_vault(tmp_path, reader, field, escape):
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("private outside content")
    paths = {"traversal": "../secret.md", "absolute": str(outside), "symlink": "link.md", "root": "."}
    (vault / "link.md").symlink_to(outside)
    record = PaperRecord(key="paper", **{field: paths[escape]})
    with pytest.raises(ValueError, match="vault"):
        reader(record, vault)


@pytest.mark.parametrize("reader", [read_paper_text, read_extract_raw])
def test_legacy_extract_reader_accepts_internal_symlink_and_resolved_vault(tmp_path, reader):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "text.md").write_text("safe content")
    (vault / "link.md").symlink_to(vault / "text.md")
    root_link = tmp_path / "vault-link"
    root_link.symlink_to(vault, target_is_directory=True)
    assert reader(PaperRecord(key="paper", md_path="link.md"), root_link) == "safe content"


def test_legacy_clean_index_accepts_current_key_bound_scoped_override(tmp_path):
    entry = {"key": "allowed", "terminal_state": {"policy": "override", "merge": "canonical",
        "policy_reasons": ["off_domain"], "required_revision": 3, "override": {
            "key": "allowed", "scope": ["acquire"], "reasons": ["off_domain"], "revision": 3}}}
    (tmp_path / "index.json").write_text(json.dumps({"version": 1, "papers": {"allowed": entry}}))
    assert set(load_vault_index(tmp_path)) == {"allowed"}
