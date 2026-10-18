"""Read-only JSON/PG reconciliation over all keys and the frozen F0 R2 pools."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .repository import Record, normalized_identity


COMPARISON_FIELDS = (
    "key", "doi", "arxiv_id", "download_status", "retrieval", "policy", "merge",
    "quarantine", "title", "year", "source", "payload_version", "payload_sha256",
)


def record_differences(left: Record | None, right: Record | None) -> dict:
    if left is None or right is None:
        return {"presence": {"json": left is not None, "postgres": right is not None}}
    return {field: {"json": getattr(left, field), "postgres": getattr(right, field)}
            for field in COMPARISON_FIELDS if getattr(left, field) != getattr(right, field)}


def _pool_workload(records: dict[str, Record], path: Path) -> tuple[set[str], list[dict]]:
    by_doi, by_title = defaultdict(set), defaultdict(set)
    normalize_title = lambda title: re.sub(r"[^a-z0-9]+", "", (title or "").lower())
    for record in records.values():
        if record.doi:
            by_doi[record.doi].add(record.key)
        if title := normalize_title(record.title):
            by_title[title].add(record.key)
    keys, unmatched = set(), []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        pool = json.loads(line)
        for entry in pool["pool"]:
            matched = by_doi.get(normalized_identity(entry.get("doi"))) or by_title.get(
                normalize_title(entry.get("title")), set(),
            )
            if matched:
                keys.update(matched)
            else:
                unmatched.append({"qid": pool["qid"], "entry": entry})
    return keys, unmatched


def compare_catalog(json_records, postgres_records, *, limit: int | None = None,
                    pool_path: Path | None = None, pool_keys=()) -> dict:
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    left = {r.key: r for r in json_records.all()}
    right = {r.key: r for r in postgres_records.all()}
    canon_keys, unmatched = _pool_workload(left, Path(pool_path)) if pool_path else (set(), [])
    canon_keys.update(pool_keys)
    workload = sorted(left.keys() | right.keys())
    selected = set(workload if limit is None else workload[:limit]) | canon_keys
    differences = []
    for key in sorted(selected):
        fields = record_differences(left.get(key), right.get(key))
        if fields:
            differences.append({"key": key, "fields": fields})
    return {
        "source": getattr(json_records, "source", {}),
        "json_count": len(left), "postgres_count": len(right),
        "workload_size": len(set(workload) | canon_keys), "limit": limit,
        "compared": len(selected), "different": len(differences),
        "difference_rate": len(differences) / len(selected) if selected else 0.0,
        "differences": differences, "pool_keys": sorted(canon_keys), "pool_unmatched": unmatched,
    }
