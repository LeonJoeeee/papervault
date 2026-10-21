"""Read-only, version-qualified coverage derived exclusively from catalog evidence.

The repository supplies a consistent catalog snapshot via ``coverage_rows``. No
artifact existence check or LightRAG status is allowed to manufacture publication.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import re

from .repository import payload_hash

VERSION_FIELDS = ("unit_id", "consumed_input_revision", "recipe_digest", "generation_id")
EXECUTION_FIELDS = ("attempt_id", "submission_id", "authority_epoch", "fencing_token")
IDENTITY_FIELDS = (*VERSION_FIELDS, *EXECUTION_FIELDS)


def _time(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return value.astimezone(timezone.utc)


def _identity(value):
    return {key: value.get(key) for key in IDENTITY_FIELDS}


def _matches(value, expected, fields=IDENTITY_FIELDS):
    return all(expected.get(key) is not None and value.get(key) == expected[key] for key in fields)


def _role(role, required, outputs, observations, selected, now, limit):
    expected = [item for item in outputs if item.get("role") == role]
    result = {"role": role, "required": required, "identity": _identity(selected),
              "observed_at": None, "ingested_at": None, "age_seconds": None,
              "max_observation_age": limit, "state": "unknown", "reason": "no_matching_observation"}
    if not required:
        result.update(state="not_applicable", reason="recipe_not_applicable")
        return result
    if not expected:
        result["reason"] = "receipt_missing_role"
        return result
    evaluated = []
    for output in expected:
        evidence = dict(result, store=output.get("store"), output_id=output.get("output_id"))
        matching = [obs for obs in observations if _matches(obs, selected)
                    and obs.get("role") == role and obs.get("store") == output.get("store")
                    and obs.get("output_id") == output.get("output_id")
                    and output.get("namespace") is not None
                    and output["namespace"] == selected.get("namespace") == obs.get("namespace")
                    and selected.get("receipt_id") is not None
                    and obs.get("receipt_id") == selected["receipt_id"]]
        dated = [(obs, _time(obs.get("observed_at"))) for obs in matching]
        dated = [(obs, stamp) for obs, stamp in dated if stamp is not None]
        if dated:
            obs, stamp = max(dated, key=lambda pair: pair[1])
            age = (now - stamp).total_seconds()
            evidence.update(observed_at=stamp.isoformat(), ingested_at=obs.get("ingested_at"), age_seconds=age)
            if age < 0:
                evidence.update(state="unknown", reason="observation_in_future")
            elif age > limit:
                evidence.update(state="stale", reason="observation_expired")
            elif obs.get("state") in {"present", "verified_empty"}:
                verification = output.get("verification") or {}
                empty = verification.get("state") == "verified_empty"
                verified = verification.get("evidence") or {}
                counts = [value for key, value in verified.items() if key == "count" or key.endswith("_count")]
                valid_empty = (empty and bool(verification.get("method")) and bool(counts)
                               and all(type(value) is int and value == 0 for value in counts))
                if (output.get("sha256") is not None or output.get("byte_length") is not None) and any(
                        (obs.get("detail") or {}).get(key) != output.get(key) for key in ("sha256", "byte_length")):
                    evidence.update(state="unknown", reason="output_bytes_not_reverified")
                elif obs.get("state") == "verified_empty" or empty:
                    evidence.update(state="present" if valid_empty else "unknown",
                                    reason="verified_empty_output" if valid_empty else "unverified_empty_output")
                elif verification.get("state") == "present":
                    evidence.update(state="present", reason="fresh_exact_output")
                else:
                    evidence.update(state="unknown", reason="receipt_output_unverified")
            elif obs.get("state") == "missing":
                evidence.update(state="missing", reason="collector_verified_absence")
            else:
                evidence.update(state="unknown", reason="required_role_" + str(obs.get("state", "unknown")))
        evaluated.append(evidence)
    # Every output of a required role must be present; do not cherry-pick one positive.
    priority = {"missing": 0, "unknown": 1, "stale": 2, "present": 3}
    result = min(evaluated, key=lambda item: priority[item["state"]]).copy()
    result["outputs"] = evaluated
    return result


def _aggregate_evidence(row, receipt, selected, now, limit, contexts=None):
    manifest_row = row.get("generation_manifest") or {}
    manifest = manifest_row.get("manifest", manifest_row)
    if manifest.get("aggregate_layers") and contexts is not None:
        # A generation has ONE logical aggregate. Verify its physical layers
        # once, then refer to the shared context instead of expanding it per unit.
        key = id(manifest_row)
        if key not in contexts:
            layers = manifest["aggregate_layers"]
            inherited = row.get("aggregate_receipts_by_namespace")
            if inherited is None:
                inherited = {(item.get("receipt", item)).get("namespace"): item
                             for item in row.get("aggregate_receipts") or []}
            inherited = dict(inherited)
            if receipt.get("namespace"):
                inherited.setdefault(receipt["namespace"], {"receipt": receipt, "recipe": row.get("recipe")})
            indexed_observations = row.get("aggregate_observations_by_receipt")
            if indexed_observations is None:
                indexed_observations = defaultdict(list)
                for observation in row.get("aggregate_observations") or []:
                    indexed_observations[observation.get("receipt_id")].append(observation)
                for observation in row.get("observations") or []:
                    indexed_observations[observation.get("receipt_id")].append(observation)
            roles, reasons = [], []
            for namespace in layers:
                item = inherited.get(namespace)
                if item is None:
                    reasons.append("aggregate_layer_receipt_missing")
                    roles.append({"namespace": namespace, "required": True, "state": "missing",
                                  "reason": "aggregate_layer_receipt_missing"})
                    continue
                ancestor = item.get("receipt", item)
                recipe = item.get("recipe") or {}
                if ancestor.get("sealed") is not True or ancestor.get("durable") is not True:
                    reasons.append("aggregate_layer_not_sealed_durable")
                contract = (recipe.get("output_contract") or {}).get(ancestor.get("achieved_tier")) or {}
                required = contract.get("required")
                if not isinstance(required, list) or not required or set(required) & set(contract.get("not_applicable", [])):
                    reasons.append("aggregate_layer_contract_unknown")
                    continue
                for role in required:
                    evidence = _role(role, True, ancestor.get("output_inventory") or [],
                        indexed_observations.get(ancestor.get("receipt_id"), []), ancestor, now, limit)
                    evidence.update(namespace=namespace, receipt_id=ancestor.get("receipt_id"))
                    roles.append(evidence)
            context_id = "aggregate:" + str(manifest_row.get("generation_id", selected.get("generation_id"))) + ":" + str(manifest_row.get("manifest_revision", "current"))
            context = {"context_id": context_id, "roles": roles, "reasons": list(dict.fromkeys(reasons)),
                       "complete": not reasons and all(role["state"] == "present" for role in roles)}
            failures = [role for role in roles if role["state"] != "present"]
            priority = {"missing": 0, "unknown": 1, "stale": 2}
            state = min((role["state"] for role in failures), key=lambda value: priority[value]) if failures else "unknown"
            summary = [] if context["complete"] else [{"role": "aggregate", "required": True, "state": state,
                "reason": "aggregate_required_evidence_unavailable", "aggregate_context_id": context_id}]
            contexts[key] = (context, frozenset(layers), summary)
        context, namespace_set, failures = contexts[key]
        reasons = list(context["reasons"])
        namespace = receipt.get("namespace")
        if not namespace or namespace not in namespace_set:
            reasons.append("aggregate_own_layer_missing")
        if manifest.get("selected_unit_receipts", {}).get(selected.get("unit_id")) != receipt.get("receipt_id"):
            reasons.append("aggregate_manifest_selection_mismatch")
        # Full evidence lives once in aggregate_contexts. Keep failures local for
        # existing repair/section consumers and an exact context reference.
        return failures, reasons, context
    layers = list(dict.fromkeys([*receipt.get("aggregate_layers", []), *manifest.get("aggregate_layers", [])]))
    if not layers:
        return [], [], None
    reasons, roles = [], []
    own = receipt.get("namespace")
    if not own or own not in layers:
        reasons.append("aggregate_own_layer_missing")
    selected_map = receipt.get("selected_unit_receipts")
    if selected_map is not None and (selected_map.get(selected.get("unit_id")) != receipt.get("receipt_id")
            or receipt.get("selected_unit_receipts_digest") != payload_hash(selected_map)):
        reasons.append("aggregate_selection_digest_mismatch")
    if manifest.get("selected_unit_receipts") is not None and manifest["selected_unit_receipts"].get(selected.get("unit_id")) != receipt.get("receipt_id"):
        reasons.append("aggregate_manifest_selection_mismatch")
    inherited = {}
    for item in row.get("aggregate_receipts") or []:
        ancestor = item.get("receipt", item)
        inherited[ancestor.get("namespace")] = (ancestor, item.get("recipe"))
    for namespace in layers:
        if namespace == own:
            continue
        entry = inherited.get(namespace)
        if entry is None:
            reasons.append("aggregate_layer_receipt_missing")
            roles.append({"namespace": namespace, "required": True, "state": "missing", "reason": "aggregate_layer_receipt_missing"})
            continue
        ancestor, recipe = entry
        if ancestor.get("sealed") is not True or ancestor.get("durable") is not True:
            reasons.append("aggregate_layer_not_sealed_durable")
        contract = ((recipe or {}).get("output_contract") or {}).get(ancestor.get("achieved_tier")) or {}
        required = contract.get("required")
        if not isinstance(required, list) or not required or set(required) & set(contract.get("not_applicable", [])):
            reasons.append("aggregate_layer_contract_unknown")
            continue
        for role in required:
            evidence = _role(role, True, ancestor.get("output_inventory") or [],
                             row.get("aggregate_observations") or [], ancestor, now, limit)
            evidence.update(namespace=namespace, receipt_id=ancestor.get("receipt_id"))
            roles.append(evidence)
    return roles, reasons, None


def _input_role(artifact, observations, requested, now, limit):
    matching = [obs for obs in observations if obs.get("role") == "input_evidence"
                and obs.get("store") == "library" and obs.get("output_id") == artifact["artifact_id"]
                and _matches(obs, requested, ("unit_id", "consumed_input_revision"))]
    dated = [(obs, _time(obs.get("observed_at"))) for obs in matching]
    dated = [(obs, stamp) for obs, stamp in dated if stamp is not None]
    result = {"role": "input_evidence", "artifact_id": artifact["artifact_id"], "state": "unknown",
              "reason": "no_matching_input_observation", "observed_at": None, "ingested_at": None,
              "age_seconds": None, "max_observation_age": limit, "identity": dict(requested)}
    if not dated:
        return result
    obs, stamp = max(dated, key=lambda pair: pair[1])
    age = (now - stamp).total_seconds()
    result.update(observed_at=stamp.isoformat(), ingested_at=obs.get("ingested_at"), age_seconds=age)
    detail = obs.get("detail") or {}
    if age < 0:
        result.update(reason="observation_in_future")
    elif age > limit:
        result.update(state="stale", reason="input_observation_expired")
    elif obs.get("state") == "missing":
        result.update(state="missing", reason="input_absent")
    elif obs.get("state") == "present" and all(detail.get(key) == artifact.get(key)
                                               for key in ("artifact_id", "sha256", "byte_length")):
        result.update(state="present", reason="fresh_exact_input_bytes")
    else:
        result.update(reason="input_unavailable_or_bytes_mismatch")
    return result


def _unit(row, selectors, now, age, contexts=None):
    unit = row.get("unit") or {}
    requirement = row.get("requirement") or {}
    selected = dict(row.get("selected_derivation") or {})
    receipt = row.get("receipt") or {}
    # Execution identity belongs to the immutable receipt, joined to its selected derivation.
    if selected.get("receipt_id") == receipt.get("receipt_id") and _matches(receipt, selected, VERSION_FIELDS):
        for field in EXECUTION_FIELDS:
            selected.setdefault(field, receipt.get(field))
        selected.setdefault("namespace", receipt.get("namespace"))
    requested = {"unit_id": unit.get("unit_id")}
    for key, value in selectors.items():
        requested[key] = requirement.get(key) if value == "current" else value
    reasons = []
    for key in VERSION_FIELDS:
        if requested.get(key) is None or selected.get(key) != requested[key]:
            reasons.append("selected_" + key + "_mismatch")
    if requirement.get("invalidated"):
        reasons.append("requirement_invalidated")
    excluded = bool(requirement.get("excluded"))
    quarantine = unit.get("quarantine") not in (None, False, "", "none", "clear")
    record = row.get("record") or {}
    if record.get("quarantine") is True:
        quarantine = True
    if record.get("policy") in {"excluded", "unknown"}:
        reasons.append("record_policy_" + record["policy"])
    if record.get("merge") == "alias":
        reasons.append("record_is_alias")
    if record.get("revision") is not None and requirement.get("record_revision") != record["revision"]:
        reasons.append("record_revision_changed")
    if quarantine:
        reasons.append("unit_quarantined")
    if excluded:
        reasons.append("unit_excluded")
    receipt_valid = (row.get("receipt_validated") is True and bool(receipt) and receipt.get("sealed") is True and receipt.get("durable") is True
                     and receipt.get("receipt_id") == selected.get("receipt_id")
                     and _matches(receipt, selected)
                     and receipt.get("achieved_tier") == selected.get("achieved_tier"))
    if not receipt_valid:
        reasons.append("receipt_not_exact_sealed_durable")
    tier = selected.get("achieved_tier")
    recipe = row.get("recipe") or {}
    contract = (recipe.get("output_contract") or {}).get(tier)
    roles = []
    valid_contract = (isinstance(contract, dict) and isinstance(contract.get("required"), list)
                      and isinstance(contract.get("not_applicable"), list))
    if valid_contract:
        required, nonapplicable = contract["required"], contract["not_applicable"]
        if not required or set(required) & set(nonapplicable):
            valid_contract = False
        for role in dict.fromkeys(required + nonapplicable):
            roles.append(_role(role, role in required, receipt.get("output_inventory") or [],
                               row.get("observations") or [], selected, now, age))
    if not valid_contract:
        reasons.append("missing_or_invalid_recipe_output_contract")
    manifest = row.get("input_manifest") or {}
    manifest_matches = True
    if "manifest" in manifest:
        manifest_matches = _matches(manifest, requested, ("unit_id", "consumed_input_revision"))
        manifest = manifest["manifest"]
    fulltext = [item for item in manifest.get("artifacts", [])
                if item.get("kind", item.get("role")) in {"fulltext_evidence", "fulltext", "full_text", "text", "section"}
                and isinstance(item.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                and isinstance(item.get("byte_length"), int) and item["byte_length"] > 0]
    input_roles = [_input_role(artifact, row.get("observations") or [], requested, now, age) for artifact in fulltext]
    aggregate_roles, aggregate_reasons, aggregate_context = _aggregate_evidence(row, receipt, selected, now, age, contexts)
    reasons.extend(aggregate_reasons)
    compatible = not reasons
    readable = (bool(fulltext) and all(role["state"] == "present" for role in input_roles)
                and manifest_matches and all(requested.get(key) == requirement.get(key)
                    for key in ("unit_id", "consumed_input_revision")))
    published = compatible
    covered = (published and valid_contract and all(item["state"] == "present" for item in roles if item["required"])
               and (aggregate_context["complete"] if aggregate_context else all(item["state"] == "present" for item in aggregate_roles)))
    return {"unit_id": unit.get("unit_id"), "kind": unit.get("kind"), "record_key": unit.get("record_key"),
            "registered": bool(row.get("record")) or (bool(unit.get("unit_id")) and unit.get("kind") == "operator_document"), "requested_identity": requested,
            "selected_identity": _identity(selected), "derivation_id": selected.get("derivation_id"),
            "receipt_id": selected.get("receipt_id"), "receipt": receipt,
            "achieved_tier": tier, "execution": row.get("jobs") or [], "quarantine": quarantine,
            "excluded": excluded, "readable": readable, "published": published,
            "policy_eligible": not quarantine and (not record or (
                record.get("policy") in {"clear", "override"} and record.get("merge") == "canonical")),
            "source_eligible": not quarantine and not excluded and not requirement.get("invalidated")
                and (not record or (record.get("policy") in {"clear", "override"}
                                    and record.get("merge") == "canonical"))
                and (record.get("revision") is None or requirement.get("record_revision") == record["revision"]),
            "tier_covered": covered, "indexed": covered and tier == "fulltext" and readable,
            "roles": roles, "aggregate_roles": aggregate_roles,
            "aggregate_context_id": aggregate_context["context_id"] if aggregate_context else None,
            "reasons": reasons, "publication_history": row.get("publication_history") or [],
            "input_evidence": manifest, "input_roles": input_roles, "requirement": requirement}


def _inventory_context(parent, expected):
    raw = parent[0] if parent else {}
    selector = raw.get("current_inventory") or {}
    revision = (expected if expected is not None else selector.get("inventory_revision",
                (raw.get("requirement") or {}).get("inventory_revision")))
    inventories = list(raw.get("inventories") or [])
    if raw.get("inventory"):
        inventories.append(raw["inventory"])
    candidates = {payload_hash(item): item for item in inventories
                  if item.get("inventory_revision", item.get("revision")) == revision}
    inventory = next(iter(candidates.values())) if len(candidates) == 1 else None
    unit = parent[1] if parent else {}
    parent_revision = (unit.get("input_evidence") or {}).get("parent_evidence_revision")
    if parent_revision is None:
        parent_revision = (unit.get("requested_identity") or {}).get("consumed_input_revision")
    return inventory, revision, parent_revision


def _apply_inventory_eligibility(parent, members, expected):
    """Apply typed-parent policy and exact mapping before a contribution can serve.

    Missing sibling outputs affect completeness separately. They must not hide a
    parent's revocation or let an old mapping survive a new inventory selection.
    """
    inventory, revision, parent_revision = _inventory_context(parent, expected)
    raw, parent_unit = parent if parent else ({}, {})
    selector = raw.get("current_inventory") or {}
    valid = bool(inventory and not selector.get("invalidated")
        and inventory.get("parent_evidence_revision", inventory.get("parent_input_revision")) == parent_revision
        and (not inventory.get("parent_consumed_input_revision") or inventory["parent_consumed_input_revision"] ==
             parent_unit.get("requested_identity", {}).get("consumed_input_revision")))
    parent_allowed = bool(parent_unit.get("policy_eligible"))
    included = set((inventory or {}).get("required_units", (inventory or {}).get("section_unit_ids", [])))
    included.update((inventory or {}).get("optional_units", []))
    for _, unit in members:
        if unit["kind"] == "section":
            allowed = bool(valid and parent_allowed and parent_unit.get("readable")
                and inventory.get("mode", "sections") == "sections" and unit["unit_id"] in included
                and unit["requirement"].get("inventory_revision") == revision
                and _mapping_compatible(inventory, unit, raw, members, unit["unit_id"], parent_revision))
        elif unit.get("requirement", {}).get("inventory_parent_selector"):
            continue  # This row supplies parent evidence, never a selected contribution.
        else:
            allowed = bool(valid and inventory.get("mode") == "whole"
                and inventory.get("required_units") == [unit["unit_id"]]
                and not inventory.get("optional_units")
                and unit.get("requirement", {}).get("inventory_revision") == revision)
        if not allowed:
            unit["source_eligible"] = False
            unit["reasons"].append("current_parent_inventory_or_policy_ineligible")


def _uncovered_state(member):
    evidence = [*member.get("roles", []), *member.get("input_roles", []), *member.get("aggregate_roles", [])]
    for state in ("missing", "stale"):
        if any(role.get("state") == state for role in evidence):
            return state, state + "_required_evidence"
    return "unknown", "required_coverage_unknown"


def _sections(parent, members, by_id, expected, qualifying_tiers=None, *, with_outcomes=False):
    counts = dict.fromkeys(("required", "complete", "missing", "stale", "unknown", "excluded"), 0)
    raw, parent_unit = parent if parent else ({}, {})
    inventory, revision, parent_revision = _inventory_context(parent, expected)
    section_mode = any(unit["kind"] == "section" for _, unit in members) or revision is not None
    if not section_mode:
        return (counts, False, []) if with_outcomes else (counts, False)
    required = set((inventory or {}).get("required_units", (inventory or {}).get("section_unit_ids", [])))
    excluded = set((inventory or {}).get("excluded_unit_ids", (inventory or {}).get("optional_units", [])))
    context = {"inventory_revision": revision, "parent_unit_id": parent_unit.get("unit_id"),
               "parent_evidence_revision": parent_revision,
               "parent_consumed_input_revision": parent_unit.get("requested_identity", {}).get("consumed_input_revision")}
    selector = raw.get("current_inventory") or {}
    valid_inventory = bool(inventory and inventory.get("complete", True) is True
        and not selector.get("invalidated")
        and inventory.get("parent_evidence_revision", inventory.get("parent_input_revision")) == parent_revision
        and (not inventory.get("parent_consumed_input_revision") or
             inventory["parent_consumed_input_revision"] == context["parent_consumed_input_revision"]))
    outcomes = []
    if inventory and inventory.get("mode") == "whole":
        valid_inventory = valid_inventory and required == {parent_unit.get("unit_id")} and not excluded
        complete = bool(valid_inventory and parent_unit and parent_unit.get("source_eligible") and (parent_unit["indexed"] if qualifying_tiers is None else
            parent_unit["tier_covered"] and parent_unit["achieved_tier"] in qualifying_tiers
            and (parent_unit["achieved_tier"] != "fulltext" or parent_unit["readable"])))
        return (counts, complete, outcomes) if with_outcomes else (counts, complete)
    counts["required"], counts["excluded"] = len(required - excluded), len(excluded)
    for unit_id in sorted(required | excluded):
        member = by_id.get(unit_id)
        state, reason = "unknown", "inventory_unknown_or_incompatible"
        if unit_id in excluded:
            state, reason = "excluded", "inventory_optional_or_excluded"
        elif not valid_inventory:
            pass
        elif member is None:
            state, reason = "missing", "required_unit_absent"
        elif member["requirement"].get("inventory_revision") != revision:
            reason = "required_unit_inventory_mismatch"
        elif not _mapping_compatible(
                inventory, member, raw, members, unit_id, parent_revision):
            reason = "required_unit_mapping_or_parent_mismatch"
        elif not parent_unit.get("policy_eligible", True):
            reason = "parent_source_ineligible"
        elif not parent_unit.get("readable"):
            state, reason = _uncovered_state(parent_unit)
            reason = "parent_" + reason
        elif (member["indexed"] if qualifying_tiers is None else
              member["tier_covered"] and member["achieved_tier"] in qualifying_tiers
              and (member["achieved_tier"] != "fulltext" or member["readable"])):
            state, reason = "complete", "required_unit_covered"
        else:
            state, reason = _uncovered_state(member)
        if state != "excluded":
            counts[state] += 1
        outcomes.append({**context, "unit_id": unit_id, "required": unit_id not in excluded,
            "state": state, "reason": reason,
            "expected_section_mapping_revision": (inventory.get("section_mapping_revisions") or {}).get(unit_id),
            "requested_identity": member.get("requested_identity") if member else None,
            "selected_identity": member.get("selected_identity") if member else None,
            "receipt_id": member.get("receipt_id") if member else None,
            "roles": member.get("roles", []) if member else [],
            "input_roles": member.get("input_roles", []) if member else [],
            "aggregate_roles": member.get("aggregate_roles", []) if member else []})
    if not valid_inventory and not required:
        counts["unknown"] = 1
    complete = bool(valid_inventory and counts["required"] > 0 and counts["complete"] == counts["required"])
    return (counts, complete, outcomes) if with_outcomes else (counts, complete)


def _mapping_compatible(inventory, member, parent_row, members, unit_id, parent_revision):
    expected = (inventory.get("section_mapping_revisions") or {}).get(unit_id)
    if expected is None:
        return False
    evidence = member["input_evidence"]
    if (evidence.get("parent_evidence_revision") != parent_revision
            or evidence.get("section_mapping_revision", evidence.get("section_definition_revision")) != expected):
        return False
    mappings = [mapping for raw, _ in members for mapping in raw.get("section_mappings", [])]
    return any(mapping.get("unit_id") == unit_id and mapping.get("mapping_revision") == expected
               and mapping.get("parent_unit_id") == (parent_row.get("unit") or {}).get("unit_id")
               and mapping.get("parent_evidence_revision") == parent_revision for mapping in mappings)


def _group_ids(rows):
    root_records = {row["unit"].get("unit_id"): row["unit"].get("record_key") for row in rows
                    if row["unit"].get("unit_id") is not None}
    parents = {}
    for row in rows:
        unit = row["unit"]
        if unit.get("kind") != "section":
            continue
        manifest = row.get("input_manifest") or {}
        manifest = manifest.get("manifest", manifest)
        revision = manifest.get("section_definition_revision", manifest.get("section_mapping_revision"))
        mappings = [mapping for mapping in row.get("section_mappings", [])
                    if mapping.get("unit_id") == unit.get("unit_id")
                    and (revision is None or mapping.get("mapping_revision") == revision)]
        matches = {mapping["parent_unit_id"] for mapping in mappings}
        if len(matches) == 1:
            parents[unit["unit_id"]] = next(iter(matches))
    result = {}
    for row in rows:
        unit = row["unit"]
        root, seen = unit.get("unit_id"), set()
        while root in parents and root not in seen:
            seen.add(root)
            root = parents[root]
        result[id(row)] = root_records.get(root) or unit.get("record_key") or root
    return result


def query_coverage(repository, scope=None, *, consumed_input_revision="current", recipe_digest="current",
                   generation_id="current", expected_inventory_revision=None, evaluation_time=None,
                   max_observation_age=300):
    """Evaluate a consistent catalog snapshot, never refreshing or repairing evidence.

    ``scope`` is a record key/unit ID or iterable of those IDs. Selectors are exact
    values or the explicit ``current`` requirement selector. Observation age is
    measured from observed_at, never from ingestion/collection insertion time.
    """
    now = _time(evaluation_time) if evaluation_time is not None else datetime.now(timezone.utc)
    if (now is None or not isinstance(max_observation_age, (int, float))
            or not math.isfinite(max_observation_age) or max_observation_age < 0):
        raise ValueError("evaluation_time must be timezone aware and observation age finite and nonnegative")
    selectors = dict(consumed_input_revision=consumed_input_revision, recipe_digest=recipe_digest,
                     generation_id=generation_id)
    if any(value is None for value in selectors.values()):
        raise ValueError("selectors require an exact value or 'current'")
    rows = repository.coverage_rows(scope)
    if generation_id == "current":
        rows = [row for row in rows if not row.get("generation")
                or ((row.get("serving_manifest") or {}).get("generation_id") == (row.get("requirement") or {}).get("generation_id")
                    if row.get("serving_manifest") else row["generation"].get("state") == "active")]
    else:
        rows = [row for row in rows if (row.get("requirement") or {}).get("generation_id") == generation_id]
    group_ids = _group_ids(rows)
    if scope is not None:
        keys = {scope} if isinstance(scope, str) else set(scope)
        record_keys = {group_ids[id(row)] for row in rows
                       if row["unit"].get("unit_id") in keys or row["unit"].get("record_key") in keys}
        rows = [row for row in rows if group_ids[id(row)] in record_keys]
    contexts = {}
    units = [_unit(row, selectors, now, max_observation_age, contexts) for row in rows]
    groups = defaultdict(list)
    for row, unit in zip(rows, units, strict=True):
        groups[group_ids[id(row)]].append((row, unit))
    records = []
    for key, members in groups.items():
        parents = [item for item in members if item[1]["kind"] != "section"]
        parent = parents[0] if len(parents) == 1 else None
        _apply_inventory_eligibility(parent, members, expected_inventory_revision)
        inventory, _, _ = _inventory_context(parent, expected_inventory_revision)
        sections, section_covered, required_outcomes = _sections(parent, members, {u["unit_id"]: u for _, u in members},
                                               expected_inventory_revision, with_outcomes=True)
        record = (parent[0].get("record") if parent else members[0][0].get("record")) or {}
        tier_coverage = {}
        for tier, eligible_tiers in (("fulltext", {"fulltext"}), ("abstract", {"fulltext", "abstract"}),
                                     ("metadata", {"fulltext", "abstract", "metadata"})):
            _, qualified = _sections(parent, members, {u["unit_id"]: u for _, u in members},
                                                expected_inventory_revision, eligible_tiers)
            if qualified is None:
                tier_coverage[tier] = bool(parent and parent[1]["tier_covered"] and parent[1]["achieved_tier"] == tier
                                           and (tier != "fulltext" or parent[1]["readable"]))
            else:
                inventory, _, _ = _inventory_context(parent, expected_inventory_revision)
                required_ids = set((inventory or {}).get("required_units", (inventory or {}).get("section_unit_ids", [])))
                required_ids -= set((inventory or {}).get("optional_units", (inventory or {}).get("excluded_unit_ids", [])))
                tier_coverage[tier] = bool(qualified and any(u["unit_id"] in required_ids and u["achieved_tier"] == tier
                                                           for _, u in members))
        records.append({"key": key, "kind": parent[1]["kind"] if parent else "paper",
                        "title": record.get("title"), "doi": record.get("doi"),
                        "registered": any(u["registered"] for _, u in members),
                        "readable": bool(parent and parent[1]["readable"]),
                        "published": bool(parent and parent[1]["published"]),
                        "indexed": section_covered if section_covered is not None else bool(parent and parent[1]["indexed"]),
                        "inventory_state": "known" if inventory else "unknown",
                        "sections": sections, "required_unit_outcomes": required_outcomes, "tier_coverage": tier_coverage, "unit_ids": [u["unit_id"] for _, u in members],
                        "tiers": {tier: sum(u["tier_covered"] and u["achieved_tier"] == tier for _, u in members)
                                  for tier in ("fulltext", "abstract", "metadata")}})
    return {"evaluation_time": now.isoformat(), "max_observation_age": max_observation_age,
            "selectors": selectors, "units": units, "records": records,
            "aggregate_contexts": {value[0]["context_id"]: value[0] for value in contexts.values()},
            "totals": {field: sum(bool(record[field]) for record in records)
                       for field in ("registered", "readable", "published", "indexed")}}
