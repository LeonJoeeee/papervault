"""Read-only Library facade over one captured JSON snapshot.

Library.__init__ creates directories and can recover/save indexes. The snapshot
adapter deliberately bypasses it, supplies only its in-memory paper map, and
calls its existing get/all_papers read methods. No writable Library escapes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from papervault.library.compat_v2 import check_readable, check_versions
from papervault.library.models import Paper
from papervault.library.store import Library

from .repository import Record


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_bytes(), object_pairs_hook=_unique_object)


class LibraryAdapter:
    def __init__(self, library: Library, *, payloads: dict[str, dict] | None = None):
        self._library = library
        self._payloads = payloads
        self.source: dict = {}

    @classmethod
    def from_snapshot(cls, path: str | Path) -> LibraryAdapter:
        path = Path(path)
        content = path.read_bytes()
        data = json.loads(content, object_pairs_hook=_unique_object)
        if not isinstance(data, dict) or not isinstance(data.get("papers"), dict):
            raise ValueError("snapshot must contain a papers object")
        check_readable(data.get("version", 1), surface="index")
        check_versions(data.get("acquisition_versions", {}))
        payloads = data["papers"]
        papers = {}
        for key, payload in payloads.items():
            Record.from_payload(key, payload)  # Fail closed; never silently skip a row.
            papers[key] = Paper.model_validate(payload)
        library = object.__new__(Library)
        library._papers = papers
        adapter = cls(library, payloads=payloads)
        adapter.source = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                          "index_version": data.get("version", 1)}
        return adapter

    def get(self, key: str) -> Record | None:
        paper = self._library.get(key)
        if paper is None:
            return None
        payload = self._payloads[key] if self._payloads is not None else paper.model_dump(mode="json")
        return Record.from_payload(key, payload)

    def all(self) -> Iterator[Record]:
        for paper in sorted(self._library.all_papers(include_quarantined=True), key=lambda p: p.key):
            yield self.get(paper.key)

    def all_aliases(self) -> dict[str, str]:
        return {paper.key: paper.terminal_state.alias_key
                for paper in self._library.all_papers(include_quarantined=True)
                if paper.terminal_state and paper.terminal_state.merge == "alias"}
