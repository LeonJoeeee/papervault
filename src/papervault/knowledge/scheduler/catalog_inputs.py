"""Capture exact consumed bytes before dispatch; never reuse a mutable source path."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from papervault.store.repository import canonical_json, payload_hash


class InputUnavailable(RuntimeError):
    """Captured evidence is missing, corrupt, or outside its owned storage root."""


class ImmutableInputStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(self, data: bytes, *, role: str) -> dict:
        from papervault.store.fencing import writer_lease

        with writer_lease(self.root.parent, None):
            return self._capture(data, role=role)

    def _capture(self, data: bytes, *, role: str) -> dict:
        if not isinstance(data, bytes) or not role:
            raise ValueError("captured input requires bytes and a role")
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / digest
        artifact = {"artifact_id": "sha256:" + digest, "sha256": digest,
                    "byte_length": len(data), "role": role}
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            self.read(artifact)
            return artifact
        # Failed/partial writes remain unreferenced and are never silently
        # replaced on retry. Recovery reports their failed integrity check.
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return artifact

    def read(self, artifact: dict) -> bytes:
        digest = artifact.get("sha256", "")
        if (not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
                or artifact.get("artifact_id") != "sha256:" + digest
                or type(artifact.get("byte_length")) is not int):
            raise InputUnavailable("captured artifact identity is invalid")
        path = self.root / digest
        if (Path(artifact.get("locator", path)).resolve() != path
                or path.is_symlink()):
            raise InputUnavailable("captured artifact escaped its storage root")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise InputUnavailable("captured artifact is unavailable") from exc
        if len(data) != artifact["byte_length"] or hashlib.sha256(data).hexdigest() != digest:
            raise InputUnavailable("captured artifact failed hash/length verification")
        return data


def make_input_manifest(unit_id: str, metadata: dict, artifacts: list[dict], *, tier: str,
                        parent_evidence_revision: str | None = None,
                        section_definition_revision: int | None = None,
                        upstream_derivation_ids: list[str] | None = None) -> tuple[dict, str]:
    if tier not in {"fulltext", "abstract", "metadata"}:
        raise ValueError("unknown input tier")
    metadata_revision = "metadata:" + payload_hash(metadata)
    manifest = {"schema_version": 1, "unit_id": unit_id, "tier": tier,
                "artifacts": artifacts, "metadata_revision": metadata_revision,
                "metadata": metadata, "upstream_derivation_ids": upstream_derivation_ids or [],
                "parent_evidence_revision": parent_evidence_revision,
                "section_definition_revision": section_definition_revision}
    return manifest, "input:" + hashlib.sha256(canonical_json(manifest).encode()).hexdigest()


_METADATA_FIELDS = ("title", "authors", "year", "venue", "volume", "issue", "pages",
                    "abstract", "doi", "arxiv_id", "source", "source_identities", "url",
                    "publication_types")


def capture_paper_input(record, unit_id: str, store: ImmutableInputStore, vault: Path,
                        *, extraction_receipt=None, extraction_origin=None) -> dict:
    from papervault.knowledge.ingest.abstract_doc import abstract_docs_enabled, build_abstract_doc
    from papervault.knowledge.ingest.distill import clean
    from papervault.knowledge.ingest.paper_library_client import PaperRecord

    vault = Path(vault).resolve()
    metadata = {key: record.payload.get(key) for key in _METADATA_FIELDS}
    text, artifacts, tier = None, [], "metadata"
    completion_id = record.payload.get("extraction_completion_id")
    upstream = []
    if completion_id and record.payload.get("md_path"):
        if not extraction_receipt:
            raise InputUnavailable("extraction registration is pending")
        from papervault.store.extraction import ExtractionStore
        completion = extraction_receipt["receipt"]
        origin = extraction_origin or {"record_key": record.key, "unit_id": unit_id}
        if (completion["completion_id"] != completion_id or completion["record_key"] != origin["record_key"]
                or extraction_receipt["unit_id"] != origin["unit_id"]):
            raise InputUnavailable("extraction registration identity mismatch")
        try:
            evidence = ExtractionStore(vault)
            evidence.verify(completion)
            output = completion["output_inventory"][0]
            if record.payload["md_path"] != output["path"]:
                raise ValueError("selected extraction path differs from receipt")
            data = evidence.read_artifact(output)
        except (OSError, ValueError) as exc:
            raise InputUnavailable("registered extraction output is unavailable or corrupt") from exc
        artifacts = [store.capture(data, role="fulltext_evidence")]
        text, tier = clean(data.decode("utf-8", errors="replace")), "fulltext"
        upstream = ["extraction-derivation:" + completion_id]
    for field in ("md_path", "txt_path"):
        if text is not None:
            break
        relative = record.payload.get(field)
        if not relative:
            continue
        path = (vault / relative).resolve()
        if not path.is_relative_to(vault):
            raise InputUnavailable("paper evidence path escaped the configured vault")
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue
        if not data.strip():
            continue
        artifacts = [store.capture(data, role="fulltext_evidence")]
        text, tier = clean(data.decode("utf-8", errors="replace")), "fulltext"
        break
    if text is None and abstract_docs_enabled():
        text = build_abstract_doc(PaperRecord.from_index_entry(record.payload))
        if text:
            tier = "abstract"
    # Metadata/abstract are themselves exact consumed bytes, captured independently
    # of the catalog commit/optimistic revision. They are never full-text evidence.
    if tier != "fulltext":
        content = text.encode() if text is not None else canonical_json(metadata).encode()
        artifacts = [store.capture(content, role="abstract_metadata" if text else "metadata")]
    manifest, revision = make_input_manifest(unit_id, metadata, artifacts, tier=tier,
        parent_evidence_revision=artifacts[0]["artifact_id"], upstream_derivation_ids=upstream)
    manifest["extraction_provenance"] = "registered" if upstream else "legacy_unknown" if tier == "fulltext" else "not_applicable"
    revision = "input:" + payload_hash(manifest)
    return {"manifest": manifest, "consumed_input_revision": revision, "tier": tier,
            "text": text if text is not None else canonical_json(metadata)}
