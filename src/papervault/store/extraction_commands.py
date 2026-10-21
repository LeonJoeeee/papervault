"""Artifact branch of publish_derivation, inside its existing writer transaction."""

import time
from uuid import NAMESPACE_URL, uuid5

from psycopg import sql
from psycopg.types.json import Jsonb

from .extraction import ExtractionStore
from .knowledge import canonical_recipe, validate_input_manifest
from .knowledge_commands import dispatch, immutable, now, outbox, require, revision, state
from .repository import payload_hash


def publish_extraction(k, operation_id, payload):
    completion = payload["completion"]
    ExtractionStore(payload["vault"]).verify(completion)
    completion_id = completion["completion_id"]
    result = {"registration": "registered", "completion_id": completion_id,
              "receipt_id": "extraction-receipt:" + completion_id,
              "derivation_id": "extraction-derivation:" + completion_id}
    old = k.get_extraction_receipt(completion_id)
    if old:
        require(old["receipt"] == completion, "immutable extraction completion differs")
        return result
    runtime = state(k)
    require(runtime["authority"] == "catalog" and runtime["authority_epoch"] == payload["authority_epoch"],
            "extraction authority epoch changed")
    require(completion["authority_epoch"] in {None, runtime["authority_epoch"]}, "stale extraction epoch")
    record = require(k.repository.get(payload["record_key"]), "extraction record absent")
    require(record.key == completion["record_key"], "extraction record identity mismatch")
    require(record.revision == payload["expected_record_revision"], "extraction record revision changed")
    if completion["authority_epoch"] is not None:
        require(completion["source_catalog_revision"] == record.revision, "captured record revision changed")
    selected = record
    if record.merge == "alias" and completion["authority_epoch"] is None:
        selected = require(k.repository.get(k.repository.resolve_alias(record.key)), "canonical record absent")
        require(selected.key != record.key
                and selected.payload.get("extraction_completion_id") == completion_id
                and selected.payload.get("md_path") == completion["output_inventory"][0]["path"],
                "pending origin extraction is not selected by canonical record")
    require(not selected.quarantine and selected.merge == "canonical" and selected.policy in {"clear", "override"},
            "record policy prohibits extraction registration")
    # Compare the captured typed identity, not a later guessed historical recipe.
    from papervault.library.models import Paper
    from papervault.library.state_v2 import effective_policy
    require(effective_policy(Paper.model_validate(selected.payload).acquisition_state, key=selected.key),
            "effective policy prohibits extraction registration")
    paper = Paper.model_validate(record.payload)
    current = paper.model_dump(mode="json")
    identity = completion["unit_evidence"]["identity"]
    staged = completion.get("staged_claim")
    if staged:
        claim = current.get("reopen_claim")
        require(claim and claim["lease_id"] == staged["lease_id"]
                and claim["generation"] == staged["generation"], "staged extraction claim differs")
        if claim["phase"] == "completed":
            require(current.get("extraction_completion_id") == completion_id, "staged extraction was not selected")
            identity = completion["guard"]["identity"]
        else:
            token = getattr(k.repository, "extraction_completion_token", None)
            require(token and token.lease_id == staged["lease_id"] and token.generation == staged["generation"]
                    and claim["phase"] == "dispatched" and claim["lease_expiry"] > time.time(),
                    "staged extraction requires guarded writeback")
    require(all(current.get(key) == value for key, value in identity.items()),
            "extraction unit identity changed")
    unit = k.get_unit_for_record(record.key)
    if unit is None:
        identity = str(uuid5(NAMESPACE_URL, "papervault:catalog-record:" + record.key))
        unit = dispatch(k, "register_unit", operation_id, {
            "unit_id": "paper:" + identity, "kind": "paper", "record_key": record.key,
            "identity": {"record_id": identity}, "legacy_keys": [
                {"source": "catalog_record", "legacy_key": record.key},
                {"source": "lightrag_doc", "legacy_key": "paper:" + record.key}]})
    require(unit["kind"] == "paper" and not unit["quarantine"], "extraction unit ineligible")
    head = k.get_extraction_head(unit["unit_id"])
    revision(head, payload, "expected_head_revision")
    guard = completion["guard"]
    if head or "catalog_previous_completion_id" in guard:
        predecessor = guard.get("catalog_previous_completion_id", guard["previous_completion_id"])
        require((head["completion_id"] if head else None) == predecessor, "stale extraction head")
    manifest = completion["input_manifest"]
    validate_input_manifest(manifest)
    require(completion["consumed_input_revision"] == "input:" + payload_hash(manifest), "extraction input digest differs")
    recipe, digest = canonical_recipe(completion["recipe"])
    require(recipe == completion["recipe"] and digest == completion["recipe_digest"], "extraction recipe differs")
    immutable(k, "recipes", {"recipe_digest": digest}, {"spec": recipe})
    immutable(k, "input_manifests", {"consumed_input_revision": completion["consumed_input_revision"]},
              {"unit_id": unit["unit_id"], "manifest": manifest, "manifest_digest": payload_hash(manifest)})
    immutable(k, "extraction_receipts", {"completion_id": completion_id}, {
        "receipt_id": result["receipt_id"], "unit_id": unit["unit_id"],
        "generation_id": completion["generation_id"], "receipt": completion,
        "digest": completion["evidence_digest"]})
    audit = k.connection.execute(sql.SQL("INSERT INTO {}(actor,event,key,detail) "
        "VALUES('extraction','extraction_registration',%s,%s) RETURNING id").format(k.repository._table("audit")),
        (record.key, Jsonb({"operation_id": operation_id, "completion_id": completion_id}))).fetchone()
    k._insert("derivations", {"derivation_id": result["derivation_id"], "key": record.key,
        "unit_id": unit["unit_id"], "artifact_inputs": manifest, "catalog_revision": audit["id"],
        "recipe": recipe, "published_at": now(k), "consumed_input_revision": completion["consumed_input_revision"],
        "recipe_digest": digest, "extraction_generation_id": completion["generation_id"],
        "extraction_receipt_id": result["receipt_id"], "achieved_tier": completion["achieved_tier"],
        "origin_lineage": {"source_catalog_revision": completion["source_catalog_revision"],
                           "provider_observations": completion["provider_observations"]}})
    if head:
        k._update("extraction_heads", {"completion_id": completion_id, "revision": head["revision"] + 1},
                  unit_id=unit["unit_id"])
    else:
        k._insert("extraction_heads", {"unit_id": unit["unit_id"], "completion_id": completion_id})
    outbox(k, "extraction_completed", {**result, "record_key": selected.key, "unit_id": unit["unit_id"],
        "output_inventory": completion["output_inventory"]}, stable_id="distill-extraction:" + completion_id)
    return result
