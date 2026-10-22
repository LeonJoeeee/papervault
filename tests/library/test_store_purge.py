"""Retired purge must preserve every research asset in standalone JSON mode."""

import pytest

from papervault.library.models import Paper
from papervault.library.store import Library
from papervault.store.repository import DeletionRefused


@pytest.mark.parametrize("target", ["Retained2026", "Missing2026"])
def test_json_purge_is_refused_without_mutating_records_indexes_or_artifacts(tmp_path, target):
    library = Library(tmp_path)
    paper = Paper(key="Retained2026", title="Retained research asset", doi="10.1234/retained",
                  arxiv_id="2601.12345")
    library._papers[paper.key] = paper
    library._reindex(paper)
    for kind in ("pdf", "md", "txt"):
        getattr(library, f"{kind}_path")(paper.key).write_bytes((kind + " retained evidence").encode())
    library.save(force=True, durable=True)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(DeletionRefused):
        library.purge(target, reason="operator request cannot authorize routine deletion")

    assert library.get(paper.key) is paper
    assert library.find(doi="10.1234/retained") is paper
    assert library.find(arxiv_id="2601.12345") is paper
    assert library.find(title="Retained research asset") is paper
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert Library(tmp_path).get(paper.key).doi == "10.1234/retained"
