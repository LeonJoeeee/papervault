"""Canonical PG -> JSON export; validate the temporary file before publication."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from psycopg.pq import TransactionStatus

from papervault.library.store import _fsync_dir
from .adapter import LibraryAdapter
from .compare import compare_catalog


def export_catalog(repository, output: Path, *, verify: bool = False) -> dict:
    output = Path(output)
    idle = repository.connection.info.transaction_status == TransactionStatus.IDLE
    with repository.connection.transaction():
        if idle:
            repository.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        snapshot = repository.connection.execute(
            "SELECT pg_current_snapshot()::text AS revision").fetchone()["revision"]
        records = list(repository.all())
        data = {"version": 1, "papers": {r.key: r.payload for r in sorted(records, key=lambda r: r.key)}}
        content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        # Exclusive handoff/file lock in cutover, and a unique temporary name in
        # ordinary exports: concurrent jobs cannot share/truncate the same temp.
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".catalog-export-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                adapter = LibraryAdapter.from_snapshot(temporary)
                report = compare_catalog(adapter, repository) if verify else {"total": len(records)}
                if verify and report["difference_rate"] != 0:
                    raise ValueError("catalog export failed shadow comparison")
                os.replace(temporary, output)
                _fsync_dir(output.parent, durable=True)
            finally:
                temporary.unlink(missing_ok=True)
    report.update(output=str(output), sha256=hashlib.sha256(content).hexdigest())
    report["postgres_snapshot"] = snapshot
    if "source" in report:
        report["source"]["path"] = str(output)
    return report
