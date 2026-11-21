"""Durable extraction evidence, independent of catalog availability.

A completed directory is published only after its bytes and evidence are fsynced.
Its pending disposition never changes: catalog registration is a separate fact.
Compatibility markdown files are projections and are not derivation outputs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .knowledge import canonical_recipe
from .repository import canonical_json, payload_hash


_log = logging.getLogger(__name__)
# Linux FICLONE ioctl for zero-copy CoW reflinks on Btrfs/XFS.
_FICLONE = 0x40049409
_COMPLETIONS_CACHE_LOCK = threading.RLock()
_COMPLETIONS_INDEX_CACHE: dict[
    str, tuple[tuple[int, int], dict[str, tuple[Path, ...]], tuple[Path, ...], tuple[Path, ...]]
] = {}
_COMPLETIONS_CACHE_MAX = 64


def _invalidate_completions_cache(versions_dir: Path) -> None:
    with _COMPLETIONS_CACHE_LOCK:
        _COMPLETIONS_INDEX_CACHE.pop(str(versions_dir), None)


def _try_reflink(source_path: Path, dest_path: Path) -> bool:
    """Attempt a Linux CoW reflink copy into ``dest_path`` via atomic temp+link."""
    durable_directory(dest_path.parent)
    temporary = dest_path.parent / ("." + dest_path.name + "." + uuid4().hex)
    try:
        with source_path.open("rb") as src, temporary.open("xb") as dst:
            fcntl.ioctl(dst.fileno(), _FICLONE, src.fileno())
            os.fsync(dst.fileno())
        try:
            os.link(temporary, dest_path)
        except FileExistsError:
            if dest_path.is_symlink():
                raise ValueError("immutable extraction evidence differs")
            return True
        fsync_directory(dest_path.parent)
        return True
    except OSError:
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _capture_source_bytes(vault: Path, dest: Path, source: bytes, source_hash: str,
                          *, record_key: str, source_role: str) -> None:
    """Persist immutable input bytes, preferring zero-copy reflink or hardlink for PDFs.

    If ``dest`` already exists with matching content, skip writing completely.
    For ``source_role == "pdf"`` when ``vault/pdfs/<record_key>.pdf`` has identical
    bytes on the same device:
      1. Try a CoW reflink (``FICLONE``) — zero disk duplication and fully isolated
         even against non-atomic in-place writes.
      2. Otherwise, for multi-block PDFs (default >= 64 KiB, configurable via
         ``PAPERVAULT_EXTRACTION_HARDLINK_MIN_BYTES``), hardlink (``os.link``) to
         avoid duplicating multi-MB PDFs on ext4 (production downloads always write
         PDFs via atomic tmp+rename). Tiny synthetic inputs fall through to
         ``write_once`` so in-place ``write_bytes`` test fixtures stay isolated.
    """
    if dest.exists():
        if dest.is_symlink() or dest.read_bytes() != source:
            raise ValueError("immutable extraction evidence differs")
        return
    if source_role == "pdf" and record_key:
        candidate = vault / "pdfs" / f"{record_key}.pdf"
        try:
            if not candidate.is_symlink():
                cand_st = candidate.stat()
                if (cand_st.st_size == len(source)
                        and hashlib.sha256(candidate.read_bytes()).hexdigest() == source_hash):
                    if _try_reflink(candidate, dest):
                        if dest.read_bytes() != source:
                            raise ValueError("immutable extraction evidence differs")
                        return
                    try:
                        min_bytes = int(os.environ.get("PAPERVAULT_EXTRACTION_HARDLINK_MIN_BYTES", "65536"))
                    except ValueError:
                        min_bytes = 65536
                    if len(source) >= min_bytes:
                        durable_directory(dest.parent)
                        if cand_st.st_dev == dest.parent.stat().st_dev:
                            try:
                                os.link(candidate, dest)
                                fsync_directory(dest.parent)
                                return
                            except FileExistsError:
                                if dest.is_symlink() or dest.read_bytes() != source:
                                    raise ValueError("immutable extraction evidence differs")
                                return
        except OSError:
            pass
    write_once(dest, source)


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
        _capture_source_bytes(
            self.vault, self.path(relative), source, source_hash,
            record_key=record_key, source_role=source_role,
        )
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
        _invalidate_completions_cache(self.versions)
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

    def _completions_index(self) -> tuple[dict[str, tuple[Path, ...]], tuple[Path, ...], tuple[Path, ...]]:
        try:
            st = self.versions.stat()
        except OSError:
            return {}, (), ()
        cache_key = str(self.versions)
        stamp = (st.st_mtime_ns, st.st_nlink)
        with _COMPLETIONS_CACHE_LOCK:
            cached = _COMPLETIONS_INDEX_CACHE.get(cache_key)
            if cached is not None and cached[0] == stamp:
                return cached[1], cached[2], cached[3]
        by_key_lists: dict[str, list[Path]] = {}
        all_paths: list[Path] = []
        malformed_paths: list[Path] = []
        cacheable = True
        try:
            entries = sorted(p for p in self.versions.iterdir() if not p.name.startswith(".") and p.is_dir())
        except OSError:
            return {}, (), ()
        for entry in entries:
            path = entry / "completion.json"
            if not path.is_file():
                cacheable = False
                continue
            all_paths.append(path)
            try:
                raw = json.loads(path.read_bytes())
                rk = raw.get("record_key") if isinstance(raw, dict) else None
                if isinstance(rk, str) and rk:
                    by_key_lists.setdefault(rk, []).append(path)
                else:
                    malformed_paths.append(path)
            except (OSError, ValueError, KeyError, TypeError):
                malformed_paths.append(path)
        by_key = {k: tuple(v) for k, v in by_key_lists.items()}
        all_tuple = tuple(all_paths)
        malformed_tuple = tuple(malformed_paths)
        if cacheable:
            with _COMPLETIONS_CACHE_LOCK:
                if len(_COMPLETIONS_INDEX_CACHE) >= _COMPLETIONS_CACHE_MAX and cache_key not in _COMPLETIONS_INDEX_CACHE:
                    _COMPLETIONS_INDEX_CACHE.pop(next(iter(_COMPLETIONS_INDEX_CACHE)))
                _COMPLETIONS_INDEX_CACHE[cache_key] = (stamp, by_key, all_tuple, malformed_tuple)
        return by_key, all_tuple, malformed_tuple

    def completions(self, *, key=None):
        if not self.versions.exists():
            return
        by_key, all_paths, malformed_paths = self._completions_index()
        if key is not None:
            for bad_path in malformed_paths:
                _log.warning("Skipping malformed extraction completion: %s", bad_path)
            candidates = by_key.get(key, ())
        else:
            candidates = all_paths
        for path in candidates:
            try:
                completion = json.loads(path.read_bytes())
                if (not isinstance(completion, dict)
                        or not isinstance(completion.get("record_key"), str)
                        or not completion["record_key"]):
                    raise ValueError("invalid extraction record key")
                if key is not None and completion["record_key"] != key:
                    continue
                self._validate_evidence(completion)
                if path.parent.name != completion["completion_id"]:
                    raise ValueError("extraction completion directory identity differs")
            except (OSError, ValueError, KeyError, TypeError):
                _log.warning("Skipping malformed extraction completion: %s", path)
                continue
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
        _invalidate_completions_cache(self.versions)


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
