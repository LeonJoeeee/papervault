"""Durable extraction evidence, independent of catalog availability.

A completed directory is published only after its bytes and evidence are fsynced.
Its pending disposition never changes: catalog registration is a separate fact.
Compatibility markdown files are projections and are not derivation outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .knowledge import canonical_recipe
from .repository import canonical_json, payload_hash


_log = logging.getLogger(__name__)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_directory(path: Path) -> None:
    if not path.exists():
        durable_directory(path.parent)
        path.mkdir(exist_ok=True)
        fsync_directory(path.parent)
    if path.is_symlink():
        raise ValueError("extraction storage cannot use symlink directories")


def write_once(path: Path, data: bytes) -> None:
    """Atomic, no-replacement installation, including concurrent identical delivery."""
    durable_directory(path.parent)
    temporary = path.parent / ("." + path.name + "." + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError("immutable extraction evidence differs")
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class ExtractionStore:
    def __init__(self, vault):
        self.vault = Path(vault).resolve()
        self.versions = self.vault / "extracts" / "versions"

    def path(self, relative: str) -> Path:
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("invalid extraction evidence path")
        path = self.vault / relative
        if (any(part.is_symlink() for part in (path, *path.parents) if part != self.vault)
                or not path.resolve().is_relative_to(self.vault)):
            raise ValueError("extraction evidence escaped its vault")
        return path

    def read_artifact(self, artifact: dict) -> bytes:
        data = self.path(artifact["path"]).read_bytes()
        if (len(data) != artifact["byte_length"]
                or hashlib.sha256(data).hexdigest() != artifact["sha256"]):
            raise ValueError("extraction artifact hash/length integrity failure")
        return data

    def capture(self, *, record_key, metadata, source: bytes, source_role, recipe,
                guard, unit_evidence, source_catalog_revision=None, authority_epoch=None) -> dict:
        spec, digest = canonical_recipe(recipe)
        source_hash = hashlib.sha256(source).hexdigest()
        relative = "extracts/inputs/" + source_hash
        write_once(self.path(relative), source)
        artifact = {"artifact_id": "sha256:" + source_hash, "sha256": source_hash,
                    "byte_length": len(source), "path": relative, "role": source_role}
        manifest = {"schema_version": 1, "metadata": metadata,
                    "metadata_revision": "metadata:" + payload_hash(metadata),
                    "artifacts": [artifact], "upstream_derivation_ids": []}
        return {"schema_version": 1, "completion_id": uuid4().hex,
                "record_key": record_key, "unit_evidence": unit_evidence,
                "source_catalog_revision": source_catalog_revision,
                "authority_epoch": authority_epoch, "guard": guard,
                "input_manifest": manifest, "consumed_input_revision": "input:" + payload_hash(manifest),
                "recipe": spec, "recipe_digest": digest}

    def finalize(self, request: dict, output: bytes, *, provider_observations=None) -> dict:
        if not output.strip():
            raise ValueError("extraction output is empty")
        completion_id = request["completion_id"]
        relative = f"extracts/versions/{completion_id}/output.md"
        generation = "extraction:" + completion_id
        stamp = datetime.now(timezone.utc).isoformat()
        inventory = [{"role": "markdown", "output_id": generation + ":markdown",
                      "namespace": generation, "store": "filesystem", "path": relative,
                      "sha256": hashlib.sha256(output).hexdigest(), "byte_length": len(output),
                      "verified_at": stamp, "verification": {"state": "present",
                          "method": "sha256-length-readback", "evidence": {"verified": True}}}]
        completion = {**request, "generation_id": generation, "achieved_tier": "fulltext",
                      "registration": "pending", "sealed": True, "durable": True,
                      "output_inventory": inventory,
                      "provider_observations": provider_observations or {"actual_model": "unknown"}}
        completion["evidence_digest"] = payload_hash(completion)
        durable_directory(self.versions)
        stage = self.versions / ("." + completion_id)
        stage.mkdir()
        write_once(stage / "output.md", output)
        if (stage / "output.md").read_bytes() != output:
            raise ValueError("extraction output failed readback")
        write_once(stage / "completion.json", canonical_json(completion).encode())
        fsync_directory(stage)
        os.rename(stage, self.versions / completion_id)
        fsync_directory(self.versions)
        # Library may have created an ancestor without syncing its entry.
        fsync_directory(self.versions.parent)
        fsync_directory(self.vault)
        return completion

    @staticmethod
    def _validate_evidence(completion: dict) -> list[dict]:
        """Validate receipt metadata separately from its retained artifact bytes."""
        if not isinstance(completion, dict):
            raise ValueError("extraction completion must be an object")
        identity = completion.get("completion_id")
        if not isinstance(identity, str) or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("invalid extraction completion identity")
        generation = "extraction:" + identity
        if (completion.get("schema_version") != 1 or completion.get("generation_id") != generation
                or completion.get("sealed") is not True or completion.get("durable") is not True
                or completion.get("achieved_tier") != "fulltext" or completion.get("registration") != "pending"):
            raise ValueError("invalid sealed extraction completion")
        if completion.get("evidence_digest") != payload_hash({
                k: v for k, v in completion.items() if k != "evidence_digest"}):
            raise ValueError("extraction completion integrity failure")
        guard = completion["guard"]
        if (not isinstance(completion["record_key"], str) or not completion["record_key"]
                or not isinstance(guard["identity"], dict)
                or guard["previous_completion_id"] is not None
                and not isinstance(guard["previous_completion_id"], str)):
            raise ValueError("invalid extraction record/projection guard")
        manifest = completion["input_manifest"]
        if (completion["consumed_input_revision"] != "input:" + payload_hash(manifest)
                or canonical_recipe(completion["recipe"]) != (completion["recipe"], completion["recipe_digest"])):
            raise ValueError("extraction input/recipe identity differs")
        unit = completion["unit_evidence"]
        if unit["kind"] != "paper" or unit["record_key"] != completion["record_key"]:
            raise ValueError("extraction unit identity differs")
        inputs, outputs = manifest["artifacts"], completion["output_inventory"]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("extraction inventory differs from output contract")
        source, output = inputs[0], outputs[0]
        if (source["path"] != "extracts/inputs/" + source["sha256"]
                or source["artifact_id"] != "sha256:" + source["sha256"]
                or source["role"] not in {"pdf", "provider_markdown"}
                or output["path"] != f"extracts/versions/{identity}/output.md"
                or output["namespace"] != generation or output["output_id"] != generation + ":markdown"
                or output["role"] != "markdown" or output["store"] != "filesystem"
                or output["verification"] != {"state": "present", "method": "sha256-length-readback",
                                              "evidence": {"verified": True}}):
            raise ValueError("extraction inventory is not immutable versioned evidence")
        return inputs + outputs

    def verify(self, completion: dict) -> None:
        for artifact in self._validate_evidence(completion):
            self.read_artifact(artifact)
        retained = self.path(f"extracts/versions/{completion['completion_id']}/completion.json")
        if json.loads(retained.read_bytes()) != completion:
            raise ValueError("extraction receipt differs from retained completion")

    def completions(self, *, key=None):
        if not self.versions.exists():
            return
        for path in sorted(self.versions.glob("[!.]*/completion.json")):
            try:
                completion = json.loads(path.read_bytes())
                self._validate_evidence(completion)
                if path.parent.name != completion["completion_id"]:
                    raise ValueError("extraction completion directory identity differs")
            except (OSError, ValueError, KeyError, TypeError):
                _log.warning("Skipping malformed extraction completion: %s", path)
                continue
            if key is None or completion["record_key"] == key:
                yield completion

    def completion(self, completion_id):
        if not isinstance(completion_id, str) or len(completion_id) != 32 or any(
                c not in "0123456789abcdef" for c in completion_id):
            raise ValueError("invalid extraction completion identity")
        value = json.loads((self.versions / completion_id / "completion.json").read_bytes())
        self.verify(value)
        return value

    def retain_from(self, source: ExtractionStore, completion: dict) -> None:
        """Transfer staged evidence before its disposable execution directory closes."""
        source.verify(completion)
        for artifact in completion["input_manifest"]["artifacts"] + completion["output_inventory"]:
            write_once(self.path(artifact["path"]), source.read_artifact(artifact))
        write_once(self.versions / completion["completion_id"] / "completion.json",
                   canonical_json(completion).encode())
        fsync_directory(self.versions)
        fsync_directory(self.versions.parent)
        fsync_directory(self.vault)


def register_completion(repository, vault, completion):
    """Persist one delivery identity so an unknown commit can be replayed exactly."""
    store = ExtractionStore(vault)
    store.verify(completion)
    path = store.versions / completion["completion_id"] / "registration-request.json"
    if path.exists():
        payload = json.loads(path.read_bytes())
        if payload["completion"] != completion or payload["vault"] != str(store.vault):
            raise ValueError("extraction registration request differs from retained evidence")
    else:
        runtime = repository.get_authority_state()
        record = repository.get_record(completion["record_key"])
        if record is None:
            raise ValueError("extraction record is not registered")
        unit = repository.get_unit_for_record(record["key"])
        head = repository.get_extraction_head(unit["unit_id"]) if unit else None
        payload = {"kind": "extraction", "record_key": record["key"], "completion": completion,
                   "authority_epoch": runtime["authority_epoch"],
                   "expected_record_revision": (completion["source_catalog_revision"]
                       if completion["authority_epoch"] is not None else record["revision"]),
                   "expected_head_revision": head["revision"] if head else 0, "vault": str(store.vault)}
        write_once(path, canonical_json(payload).encode())
    return repository.command("publish_derivation", "publish-extraction:" + completion["completion_id"], payload)
