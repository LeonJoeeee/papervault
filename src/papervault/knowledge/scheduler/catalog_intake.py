"""Separately admitted capture of registered records into exact distillation jobs."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from papervault.store.repository import payload_hash

from .catalog_inputs import ImmutableInputStore, InputUnavailable, capture_paper_input, make_input_manifest


BACKEND_ROLES = ("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
                "relation_chunks", "llm_response_cache", "doc_status", "chunks_vdb", "entities_vdb",
                "relationships_vdb", "graph")
OUTPUT_ROLES = (*BACKEND_ROLES, "tombstones")


def effective_embedding_contract() -> dict:
    from papervault.knowledge.config import CONFIG

    return {"schema_version": 1, "model": CONFIG.bge_m3.model_path,
            "dimension": 1024, "max_length": 8192, "use_fp16": True,
            "representation": "BGEM3FlagModel.encode:dense_vecs:cosine",
            "implementation_sha256": hashlib.sha256(
                Path(__file__).parents[1].joinpath("store/lightrag_init.py").read_bytes()).hexdigest(),
            "packages": {name: metadata.version(name) for name in ("FlagEmbedding", "transformers", "torch")}}


def effective_distill_recipe() -> dict:
    from lightrag.prompt import PROMPTS

    from papervault.domain import get_domain
    from papervault.knowledge.config import CONFIG
    from papervault.knowledge.store.extraction_prompt import apply_ks_extraction_prompt
    from papervault.knowledge.store.kw_prompt_compat import apply_kw_prompt_compat
    from papervault.llm_routing import route

    apply_ks_extraction_prompt(examples=True, exclusions=True)
    apply_kw_prompt_compat()
    package_root = Path(__file__).resolve().parents[1]
    sources = ("ingest/distill.py", "ingest/chunking.py", "ingest/abstract_doc.py",
               "store/build_pipeline.py", "store/extraction_prompt.py", "store/graph.py",
               "store/kw_prompt_compat.py", "store/catalog_storage.py", "store/catalog_adapter.py")
    return {"schema_version": 1, "name": "papervault-distill",
            "packages": {name: metadata.version(name) for name in ("papervault", "lightrag-hku", "torch")},
            "code_sha256": {name: hashlib.sha256((package_root / name).read_bytes()).hexdigest() for name in sources},
            "configured_models": {"build": route("build"), "keyword": route("keyword"),
                                  "embedding": CONFIG.bge_m3.model_path},
            "embedding_contract": effective_embedding_contract(),
            "prompts": {key: value for key, value in PROMPTS.items() if isinstance(value, (str, list, dict))},
            "ontology": get_domain().entity_types_guidance,
            "chunking": {"engine": "sentence_boundary", "token_size": CONFIG.lightrag.chunk_token_size,
                         "overlap": CONFIG.lightrag.chunk_overlap_token_size},
            "execution_settings": {
                "llm_max_async": int(os.getenv("KS_LLM_MAX_ASYNC", "32")),
                "max_parallel_insert": int(os.getenv("KS_MAX_PARALLEL_INSERT", "16")),
                "embedding_max_async": int(os.getenv("KS_EMBED_MAX_ASYNC", "16")),
                "llm_timeout": int(os.getenv("KS_LLM_TIMEOUT", "240")),
                "embedding_dim": 1024, "embedding_max_token_size": 8192,
                "entity_extract_cache": True},
            "extraction": {"max_gleaning": 1, "force_llm_summary_on_merge": 8, "language": "English"},
            "output_contract": {
                "fulltext": {"required": list(OUTPUT_ROLES), "not_applicable": ["metadata"]},
                "abstract": {"required": list(OUTPUT_ROLES), "not_applicable": ["metadata", "fulltext_evidence"]},
                "metadata": {"required": ["metadata", "tombstones"], "not_applicable": [*BACKEND_ROLES, "fulltext_evidence"]}}}


class CatalogIntake:
    def __init__(self, repository, input_store: ImmutableInputStore, vault: Path, *, recipe=None):
        self.repository = repository
        self.input_store = input_store
        self.vault = Path(vault)
        self.recipe = recipe if recipe is not None else effective_distill_recipe()
        self.after = None

    def _command(self, action, payload):
        return self.repository.command(action, "intake:" + action + ":" + payload_hash(payload), payload)

    def capture_record(self, record) -> dict:
        if record.merge == "alias":
            return {"record_key": record.key, "excluded": True, "reason": "alias"}
        unit = self.repository.get_unit_for_record(record.key)
        if unit is None:
            identity = str(uuid5(NAMESPACE_URL, "papervault:catalog-record:" + record.key))
            unit = self._command("register_unit", {
                "unit_id": "paper:" + identity, "kind": "paper", "record_key": record.key,
                "identity": {"record_id": identity},
                "legacy_keys": [{"source": "catalog_record", "legacy_key": record.key},
                                {"source": "lightrag_doc", "legacy_key": "paper:" + record.key}]})
        captured = capture_paper_input(record, unit["unit_id"], self.input_store, self.vault)
        manifest, input_revision = captured["manifest"], captured["consumed_input_revision"]
        self._command("register_input", {"unit_id": unit["unit_id"],
                      "consumed_input_revision": input_revision, "manifest": manifest})
        recipe_digest = self._command("register_recipe", {"recipe": self.recipe})["recipe_digest"]
        generation = self.repository.get_active_generation()
        if generation is None:
            raise RuntimeError("knowledge handoff has no active generation")
        generation_id = generation["generation_id"]
        inventory = {"parent_unit_id": unit["unit_id"], "mode": "whole",
                     "parent_evidence_revision": manifest["artifacts"][0]["artifact_id"],
                     "parent_consumed_input_revision": input_revision,
                     "required_units": [unit["unit_id"]], "optional_units": [], "section_mapping_revisions": {}}
        inventory_revision = "inventory:" + payload_hash(inventory)
        self._command("register_inventory", {**inventory, "inventory_revision": inventory_revision})
        selector = self.repository.get_current_inventory(unit["unit_id"], generation_id)
        if (selector is None or selector.get("invalidated")
                or selector.get("record_revision", record.revision) != record.revision
                or selector["inventory_revision"] != inventory_revision):
            self._command("set_inventory", {"parent_unit_id": unit["unit_id"], "generation_id": generation_id,
                          "inventory_revision": inventory_revision, "expected_record_revision": record.revision,
                          "expected_revision": selector["revision"] if selector else 0})
        previous = self.repository.get_requirement(unit["unit_id"], generation_id)
        excluded = bool(record.quarantine or record.policy not in {"clear", "override"})
        desired = {"unit_id": unit["unit_id"], "consumed_input_revision": input_revision,
                   "recipe_digest": recipe_digest, "generation_id": generation_id,
                   "inventory_revision": inventory_revision, "excluded": excluded}
        predecessor = None
        if previous:
            predecessor = self.repository.get_logical_job(unit["unit_id"], previous["consumed_input_revision"],
                                                         previous["recipe_digest"], generation_id)
        if (previous is None or previous.get("invalidated")
                or previous.get("record_revision", record.revision) != record.revision
                or any(previous.get(key) != value for key, value in desired.items())):
            self._command("set_requirement", {**desired,
                          "expected_revision": previous["revision"] if previous else 0,
                          "expected_record_revision": record.revision})
        self.observe_input(unit["unit_id"], input_revision, manifest, generation_id, recipe_digest)
        if excluded:
            return {"unit_id": unit["unit_id"], "excluded": True, "reason": "catalog_policy_or_quarantine"}
        existing = self.repository.get_logical_job(unit["unit_id"], input_revision, recipe_digest, generation_id)
        if existing:
            return existing
        return self._command("enqueue", {"unit_id": unit["unit_id"], "consumed_input_revision": input_revision,
                             "recipe_digest": recipe_digest, "generation_id": generation_id,
                             "predecessor_job_id": (predecessor.get("job_id", predecessor.get("id"))
                                                    if predecessor else None)})

    def observe_input(self, unit_id, input_revision, manifest, generation_id, recipe_digest=None):
        for artifact in manifest["artifacts"]:
            data = self.input_store.read(artifact)
            observation = {"unit_id": unit_id, "consumed_input_revision": input_revision,
                           "generation_id": generation_id, "recipe_digest": recipe_digest,
                           "role": "input_evidence", "store": "library", "output_id": artifact["artifact_id"],
                           "state": "present", "observed_at": datetime.now(timezone.utc).isoformat(),
                           "detail": {"artifact_id": artifact["artifact_id"],
                                      "sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data)}}
            observation["observation_id"] = "input-observation:" + payload_hash(observation)
            self.repository.command("observe", observation["observation_id"], observation)

    def capture_document(self, kind, key, path, *, tokenizer=None, max_tokens=None,
                         read_text=None, force=False):
        from papervault.knowledge.ingest.operator_docs import (
            AlreadyIngestedError, DEFAULT_MAX_TOKENS, build_sections, check_source_enabled, validate_key,
        )

        check_source_enabled(kind)
        validate_key(kind, key)
        path = Path(path)
        raw = path.read_bytes() if read_text is None else read_text(path).encode()
        text = raw.decode("utf-8", errors="replace")
        parent_id = "operator_document:" + str(uuid5(NAMESPACE_URL, "papervault:operator-document:" + key))
        parent_artifact = self.input_store.capture(raw, role="fulltext_evidence")
        parent_revision = parent_artifact["artifact_id"]
        parent_manifest, parent_input = make_input_manifest(
            parent_id, {"source_kind": kind, "source_key": key}, [parent_artifact], tier="fulltext",
            parent_evidence_revision=parent_revision)
        generation = self.repository.get_active_generation()
        if generation is None:
            raise RuntimeError("knowledge handoff has no active generation")
        generation_id = generation["generation_id"]
        previous = self.repository.get_current_inventory(parent_id, generation_id)
        definition = {"schema_version": 1, "algorithm": "heading_aware_then_sentence_boundary",
                      "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
                      "is_markdown": path.suffix.lower() in {".md", ".markdown"},
                      "tokenizer": (type(tokenizer).__module__ + "." + type(tokenizer).__qualname__
                                    if tokenizer is not None else "tiktoken:cl100k_base"),
                      "code_sha256": hashlib.sha256(Path(__file__).parents[1].joinpath("ingest/operator_docs.py").read_bytes()).hexdigest()}
        definition_digest = payload_hash(definition)
        if previous and not force:
            old_ids = previous["required_units"]
            old_mapping = self.repository.get_section(old_ids[0]) if old_ids else None
            if (previous.get("parent_consumed_input_revision") != parent_input
                    or not old_mapping or old_mapping["definition"].get("segmentation_digest") != definition_digest):
                raise AlreadyIngestedError(f"{key} already has a different source/segmentation revision; use force to queue a new generation")
        sections = build_sections(kind, key, text, tokenizer=tokenizer, max_tokens=max_tokens,
                                  is_markdown=definition["is_markdown"])
        if not sections:
            raise ValueError("operator document has no content after segmentation")
        self._command("register_unit", {"unit_id": parent_id, "kind": "operator_document",
                      "identity": {"document_id": parent_id.split(":", 1)[1]},
                      "legacy_keys": [{"source": "operator_document", "legacy_key": key}]})
        self._command("register_input", {"unit_id": parent_id, "consumed_input_revision": parent_input,
                      "manifest": parent_manifest})
        recipe_digest = self._command("register_recipe", {"recipe": self.recipe})["recipe_digest"]
        section_inputs, revisions = [], {}
        for index, section in enumerate(sections):
            unit_id = "section:" + str(uuid5(NAMESPACE_URL, parent_id + ":section:" + str(index)))
            self._command("register_unit", {"unit_id": unit_id, "kind": "section",
                          "identity": {"section_id": unit_id.split(":", 1)[1]},
                          "legacy_keys": [{"source": "operator_section", "legacy_key": section.doc_id}]})
            old = self.repository.get_section(unit_id)
            locator = {**definition, "segmentation_digest": definition_digest, "locator": {"index": index}}
            unchanged = old and old["parent_evidence_revision"] == parent_revision and old["definition"] == locator
            mapping_revision = old["mapping_revision"] if unchanged else old["mapping_revision"] + 1 if old else 1
            self._command("register_section", {"unit_id": unit_id, "mapping_revision": mapping_revision,
                          "parent_unit_id": parent_id, "parent_evidence_revision": parent_revision,
                          "definition": locator})
            artifact = self.input_store.capture(section.text.encode(), role="fulltext_evidence")
            manifest, _ = make_input_manifest(unit_id, {"source_kind": kind, "source_key": section.file_path},
                [artifact], tier="fulltext", parent_evidence_revision=parent_revision,
                section_definition_revision=mapping_revision)
            manifest["parent_consumed_input_revision"] = parent_input
            input_revision = "input:" + payload_hash(manifest)
            self._command("register_input", {"unit_id": unit_id, "consumed_input_revision": input_revision,
                          "manifest": manifest})
            section_inputs.append((unit_id, input_revision, manifest))
            revisions[unit_id] = mapping_revision
        inventory = {"parent_unit_id": parent_id, "parent_evidence_revision": parent_revision,
                     "parent_consumed_input_revision": parent_input, "mode": "sections",
                     "required_units": [unit_id for unit_id, _, _ in section_inputs], "optional_units": [],
                     "section_mapping_revisions": revisions}
        inventory_revision = "inventory:" + payload_hash(inventory)
        self._command("register_inventory", {**inventory, "inventory_revision": inventory_revision})
        mode = "candidate" if force else previous.get("enqueue_mode", "active") if previous else "active"
        request_id = uuid4().hex if force else previous.get("request_id") if previous else None
        request_id = request_id or "operator:" + payload_hash({"parent": parent_id, "inventory": inventory_revision})
        if force or previous is None or previous["inventory_revision"] != inventory_revision:
            self._command("set_inventory", {"parent_unit_id": parent_id, "generation_id": generation_id,
                          "inventory_revision": inventory_revision, "expected_revision": previous["revision"] if previous else 0,
                          "enqueue_mode": mode, "request_id": request_id, "desired_recipe_digest": recipe_digest})
        for unit_id, input_revision, manifest in section_inputs:
            desired = self.repository.get_requirement(unit_id, generation_id)
            if (desired is None or desired.get("invalidated") or desired.get("excluded")
                    or desired["consumed_input_revision"] != input_revision
                    or desired["recipe_digest"] != recipe_digest or desired.get("inventory_revision") != inventory_revision
                    or desired.get("enqueue_mode", "active") != mode):
                self._command("set_requirement", {"unit_id": unit_id, "generation_id": generation_id,
                              "consumed_input_revision": input_revision, "recipe_digest": recipe_digest,
                              "inventory_revision": inventory_revision, "expected_revision": desired["revision"] if desired else 0})
            self.observe_input(unit_id, input_revision, manifest, generation_id, recipe_digest)
            if mode == "active":
                self._command("enqueue", {"unit_id": unit_id, "generation_id": generation_id,
                              "consumed_input_revision": input_revision, "recipe_digest": recipe_digest})
        self.observe_input(parent_id, parent_input, parent_manifest, generation_id)
        candidate = None
        if mode == "candidate":
            operation_id = "operator-fork:" + request_id
            candidate = self.repository.get_operation(operation_id)
            if candidate is None:
                current = self.repository.get_generation(generation_id)
                runtime = self.repository.get_authority_state()
                candidate = self.repository.command("fork_generation", operation_id,
                    {"generation_id": "generation:" + payload_hash({"request_id": request_id}),
                     "source_generation_id": generation_id, "expected_manifest_revision": current["manifest_revision"],
                     "expected_requirements_revision": runtime["requirements_revision"]})
            generation_id = candidate["generation_id"]
        return {"catalog_queued": True, "queued": len(section_inputs), "done": 0, "error": 0,
                "sections": len(section_inputs), "generation_id": generation_id, "inventory_revision": inventory_revision,
                "rebuild_units": candidate["queued_jobs"] if candidate else None}

    def scan(self, *, limit=100, only_keys=None) -> dict:
        if only_keys is not None:
            records = [self.repository.get_record(key) for key in sorted(set(only_keys))[:limit]]
            records = [record for record in records if record is not None]
            return self._capture_rows(records)
        pending = self.repository.list_pending_input_records(limit=limit)
        records = list(pending)
        remaining = max(0, limit - len(records))
        if remaining:
            page = self.repository.list_registered_records(limit=remaining, after=self.after)
            self.after = page[-1]["key"] if len(page) == remaining else None
            known = {record["key"] for record in records}
            records.extend(record for record in page if record["key"] not in known)
        return self._capture_rows(records)

    def _capture_rows(self, records):
        from papervault.store.repository import Record

        counts = {"captured": 0, "excluded": 0, "errors": 0}
        for record in records:
            if isinstance(record, dict):
                record = replace(Record.from_payload(record["key"], record["payload"]), revision=record["revision"])
            try:
                result = self.capture_record(record)
            except (InputUnavailable, OSError) as exc:
                evidence = {"record_key": record.key, "record_revision": record.revision,
                            "error_type": type(exc).__name__}
                discrepancy_id = "input-capture:" + payload_hash(evidence)
                unit = self.repository.get_unit_for_record(record.key)
                self.repository.command("record_discrepancy", discrepancy_id,
                    {"discrepancy_id": discrepancy_id, "unit_id": unit["unit_id"] if unit else None,
                     "reason": "input_capture_unavailable", "evidence": evidence})
                counts["errors"] += 1
                continue
            counts["excluded" if result.get("excluded") else "captured"] += 1
        return counts
