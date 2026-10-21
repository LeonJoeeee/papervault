"""Coordinated catalog, vault and knowledge-store recovery capture adapters."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import ExitStack, contextmanager
from importlib.metadata import version
from pathlib import Path

from .extraction import write_once
from .fencing import exclusive_fence
from .recovery import ALERT_FRACTION, DERIVATIVE_SOFT_CAP, REQUIRED_STORES, checked_path, child, digest_file
from .recovery_postgres import catalog_barrier, export_catalog_state
from .repository import AuditEvent, canonical_json


_LOCK_FILES = {".catalog-writers.lock", ".write.lock", ".extraction-write.lock"}
_MUTABLE_VAULT_STATE = {"index.json", "index.json.bak", "library.bib", "manifest.log",
                       "correction-ledger.json", "reopen-budget.jsonl", ".catalog-handoff.json"}


def _files(root):
    root = checked_path(root)
    if not root.is_dir():
        raise ValueError("recovery source root must already exist")
    for path in sorted(root.rglob("*")):
        checked_path(path)
        if path.name in _LOCK_FILES:
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("source inventory contains a nonregular object")
        if path.name == ".env" or path.name.startswith(".env.") or path.name == "llm_keys.json":
            raise ValueError("source inventory contains protected runtime credentials")
        yield path


class CoordinatedSource:
    """Explicit roots and connections only; no runtime CONFIG or dotenv discovery.

    Full sets require both physical backend adapters. The separate metadata sample
    scope may omit their materializations only when the catalog proves none are
    referenced; it can never enter complete production retention.
    """
    def __init__(self, repository, *, vault, knowledge_root, postgres=None, neo4j=None,
                 metadata_sample=False):
        self.repository = repository
        self.vault = checked_path(vault)
        self.knowledge_root = checked_path(knowledge_root)
        if (self.vault == self.knowledge_root or self.vault.is_relative_to(self.knowledge_root)
                or self.knowledge_root.is_relative_to(self.vault)):
            raise ValueError("vault and knowledge recovery roots must be disjoint")
        self.postgres, self.neo4j = postgres, neo4j
        self.metadata_sample = metadata_sample

    def plan(self):
        if not self.metadata_sample and (self.postgres is None or self.neo4j is None):
            raise ValueError("complete snapshot requires PostgreSQL and Neo4j backend adapters")
        size = sum(path.stat().st_size for root in (self.vault, self.knowledge_root) for path in _files(root))
        backend = sum(adapter.estimated_bytes() for adapter in (self.postgres, self.neo4j) if adapter is not None)
        catalog_bytes = self.repository.connection.execute(
            "SELECT COALESCE(sum(pg_total_relation_size(c.oid)),0) AS bytes FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND c.relkind='r'",
            (self.repository.schema,)).fetchone()["bytes"]
        derivatives = backend + sum(path.stat().st_size for root in (self.vault / "extracts", self.knowledge_root)
                                    if root.exists() for path in _files(root))
        return {"required_stores": sorted(REQUIRED_STORES), "estimated_bytes": (size + backend + int(catalog_bytes)) * 4 + 1024 * 1024,
                "derivative_bytes": derivatives,
                "source_roots": [str(self.vault), str(self.knowledge_root)],
                "metadata_sample": self.metadata_sample,
                "capacity_policy": {"derivative_soft_cap_bytes": DERIVATIVE_SOFT_CAP,
                    "alert_at_bytes": int(DERIVATIVE_SOFT_CAP * ALERT_FRACTION),
                    "soft_cap_action": "alert; never delete protected/current assets"}}

    @contextmanager
    def barrier(self, snapshot_id):
        with ExitStack() as stack:
            for root in sorted((self.vault, self.knowledge_root)):
                stack.enter_context(exclusive_fence(root))
            checkpoint = stack.enter_context(catalog_barrier(self.repository, snapshot_id))
            for adapter in (self.postgres, self.neo4j):
                if adapter is not None:
                    stack.enter_context(adapter.barrier())
            yield checkpoint

    def export(self, destination, checkpoint):
        destination = checked_path(destination)
        if any(destination.is_relative_to(root) or root.is_relative_to(destination)
               for root in (self.vault, self.knowledge_root)):
            raise ValueError("recovery output must be isolated from every source root")
        catalog = export_catalog_state(self.repository, checkpoint)
        source_inventory = {str(path): digest_file(path) for root in (self.vault, self.knowledge_root)
                            for path in _files(root)}
        if self.metadata_sample:
            for row in catalog["tables"]["input_manifests"]:
                if row["manifest"].get("tier") != "metadata":
                    raise ValueError("metadata sample cannot omit backend stores for nonmetadata inputs")
            for row in catalog["tables"]["knowledge_receipts"]:
                receipt = row["receipt"]
                if any(item["store"] not in {"catalog", "catalog_inputs"} for item in receipt["output_inventory"]):
                    raise ValueError("metadata sample cannot omit referenced physical backend outputs")
        components = {}
        protected = []

        def data(name, value, format_name="json-records-v1"):
            path = name + "/data.json"
            write_once(destination / path, canonical_json(value).encode())
            components[name] = {"format": format_name, "checkpoint": checkpoint, "paths": [path]}

        data("catalog", catalog, "papervault-catalog-v1")
        data("recipes", {"recipes": catalog["tables"]["recipes"],
                         "extraction_receipts": catalog["tables"]["extraction_receipts"]})
        inventories = {name: [] for name in ("sources", "extracts", "inputs")}
        for root in (self.vault, self.knowledge_root):
            for path in _files(root):
                relative = path.relative_to(root)
                if root == self.vault:
                    name = "extracts" if relative.parts[0] == "extracts" else "sources"
                    locator = Path("extracts/vault", *relative.parts[1:]) if name == "extracts" else Path(name, relative)
                else:
                    name = "inputs"
                    locator = Path(name, relative)
                target = destination / locator
                target.parent.mkdir(parents=True, exist_ok=True)
                before = digest_file(path)
                shutil.copyfile(path, target, follow_symlinks=False)
                if digest_file(target) != before or digest_file(path) != before:
                    raise ValueError("source bytes changed during coordinated recovery export")
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                inventories[name].append(locator.as_posix())
                # Mutable projections/caches are backed up but must not pin a
                # different weekly package forever. Retained input/source bytes
                # still require a newer verified recovery copy before retirement.
                if ((root == self.vault and relative.as_posix() not in _MUTABLE_VAULT_STATE)
                        or (root == self.knowledge_root and relative.parts[0] == "catalog_inputs")):
                    protected.append(locator.as_posix())
        for name, paths in inventories.items():
            index = name + "/inventory.json"
            write_once(destination / index, canonical_json({"files": paths, "verified_empty": not paths}).encode())
            components[name] = {"format": "filesystem-inventory-v1", "checkpoint": checkpoint,
                                "paths": [*paths, index]}
        verify_file_dependencies(destination, catalog)
        for name, adapter in (("postgres", self.postgres), ("neo4j", self.neo4j)):
            if adapter is None:
                data(name, {"scope": "metadata_sample", "verified_no_referenced_backend_outputs": True,
                            "catalog_sha256": digest_file(destination / "catalog/data.json")["sha256"]},
                     "metadata-sample-not-applicable-v1")
            else:
                data(name, adapter.export(), adapter.format_name)
        after = {str(path): digest_file(path) for root in (self.vault, self.knowledge_root) for path in _files(root)}
        if after != source_inventory:
            raise ValueError("source inventory changed during coordinated export")
        parents = []
        units = {row["record_key"]: row for row in catalog["tables"]["units"]
                 if row["kind"] == "paper"}
        for row in catalog["tables"]["records"]:
            unit = units.get(row["key"])
            parents.append({"record_key": row["key"], "unit_id": unit["unit_id"] if unit else None,
                            "record_revision": row["revision"], "kind": "paper",
                            "metadata_sha256": hashlib.sha256(canonical_json(row["payload"]).encode()).hexdigest()})
        for row in catalog["tables"]["units"]:
            if row["kind"] == "operator_document":
                parents.append({"unit_id": row["unit_id"], "kind": row["kind"], "identity": row["identity"]})
        return {"components": components, "parent_inventory": parents, "protected_paths": protected,
                "source_roots": [str(self.vault), str(self.knowledge_root)],
                "tool_identities": {"papervault": version("papervault"), "lightrag-hku": version("lightrag-hku"),
                    "recovery_source_sha256": digest_file(Path(__file__))["sha256"]},
                "limitations": ["Same-host material does not provide host-loss recovery."] + (
                    ["Metadata-only sample; no graph/vector or OCR equivalence claim."] if self.metadata_sample else [])}

    def register(self, manifest, path):
        audit = AuditEvent("recovery", "recovery_snapshot_registered", {
            "snapshot_id": manifest["snapshot_id"], "checkpoint": manifest["checkpoint"],
            "manifest_sha256": digest_file(path / "manifest.json")["sha256"],
            "root": str(path), "sample": manifest["sample"]})
        with self.repository._write(audit):
            self.repository._finish(None, audit)


def verify_file_dependencies(package, catalog):
    """Verify retained bytes against catalog lineage, not just exporter file lists."""
    from .knowledge import canonical_recipe
    from .extraction import ExtractionStore

    def artifact(relative, expected):
        try:
            actual = digest_file(child(package, relative))
        except OSError as exc:
            raise ValueError("required input/artifact is absent from recovery package") from exc
        if actual != {key: expected[key] for key in ("sha256", "byte_length")}:
            raise ValueError("required input/artifact checksum mismatch")

    for row in catalog["tables"]["input_manifests"]:
        for item in row["manifest"]["artifacts"]:
            if item.get("path", "").startswith("extracts/"):
                artifact("extracts/vault/" + item["path"].removeprefix("extracts/"), item)
            else:
                artifact("inputs/catalog_inputs/" + item["sha256"], item)
    for row in catalog["tables"]["extraction_receipts"]:
        receipt = row["receipt"]
        for item in ExtractionStore._validate_evidence(receipt):
            relative = item["path"]
            if not relative.startswith("extracts/"):
                raise ValueError("extraction artifact is outside its declared evidence root")
            artifact("extracts/vault/" + relative.removeprefix("extracts/"), item)
        retained = child(package, "extracts/vault/versions/" + receipt["completion_id"] + "/completion.json")
        if json.loads(retained.read_bytes()) != receipt:
            raise ValueError("catalog extraction receipt differs from retained completion")
    # Legacy authority and interrupted reopen work can hold sealed completions
    # that have never reached the catalog. Their dependency closure is equally
    # necessary for recovery; a copied output alone is insufficient.
    roots = [child(package, "extracts/vault")]
    roots.extend(path for path in child(package, "sources").rglob("extracts") if path.is_dir())
    for root in roots:
        versions = root / "versions"
        if not versions.is_dir():
            continue
        for directory in versions.iterdir():
            if directory.name.startswith("."):
                continue  # Retain unpublished partial bytes without credit.
            completion = json.loads(child(directory, "completion.json").read_bytes())
            if completion.get("completion_id") != directory.name:
                raise ValueError("extraction completion directory identity differs")
            for item in ExtractionStore._validate_evidence(completion):
                relative = root.relative_to(package).as_posix() + "/" + item["path"].removeprefix("extracts/")
                artifact(relative, item)
    for row in catalog["tables"]["recipes"]:
        if canonical_recipe(row["spec"]) != (row["spec"], row["recipe_digest"]):
            raise ValueError("retained recipe identity differs from catalog")
    for row in catalog["tables"]["records"]:
        for field in ("pdf_path", "md_path", "txt_path"):
            relative = row["payload"].get(field)
            if relative:
                prefix, rest = ("extracts/vault/", relative.removeprefix("extracts/")) if relative.startswith("extracts/") else ("sources/", relative)
                if not child(package, prefix + rest).is_file():
                    raise ValueError("selected source/artifact path is missing from recovery package")
