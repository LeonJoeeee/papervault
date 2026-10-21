"""R2 entry-weighted coverage using the unchanged F0 deterministic matcher."""
from __future__ import annotations

from copy import deepcopy
import re
import math

from .repository import payload_hash

FROZEN_WORKLOAD_DIGEST = "def9c999fcfdc8e87f06dde5eca449d33986fee37463aed24af37c37318faeb7"


def _doi(value):
    return (value or "").strip().lower()


def _title(value):
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def match_entries(records, entries):
    """F0 semantics: last DOI record wins, first exact normalized title wins."""
    by_doi, by_title = {}, {}
    for record in records:
        if doi := _doi(record.get("doi")):
            by_doi[doi] = record["key"]
        if title := _title(record.get("title")):
            by_title.setdefault(title, record["key"])
    return [by_doi.get(_doi(entry.get("doi"))) or by_title.get(_title(entry.get("title")))
            for entry in entries]


def _difference_explanation(field, before, after, evidence, outcomes, record):
    reasons = [reason for unit in evidence for reason in unit.get("reasons", [])]
    states = [role.get("state") for unit in evidence for role in
              [*unit.get("roles", []), *unit.get("input_roles", []), *unit.get("aggregate_roles", [])]]
    if not after and field == "readable":
        input_states = [role.get("state") for unit in evidence for role in unit.get("input_roles", [])]
        if "missing" in input_states:
            return {"category": "materialization", "status": "explained", "reasons": ["exact_fulltext_input_missing"]}
        if "stale" in input_states:
            return {"category": "freshness", "status": "explained", "reasons": ["exact_fulltext_verification_expired"]}
        if any((unit.get("input_evidence") or {}).get("tier") in {"abstract", "metadata"} for unit in evidence):
            return {"category": "tier", "status": "explained", "reasons": ["lower_tier_input_has_no_fulltext_evidence"]}
        return {"category": "legacy_provenance", "status": "unresolved",
                "reasons": ["legacy_file_existence_does_not_verify_requested_input"]}
    if not after and field == "metadata":
        return {"category": "identity_mapping", "status": "explained", "reasons": ["no_current_registered_match"]}
    if not after:
        for fragment, category in (("recipe", "recipe"), ("input", "input_change"),
                                   ("receipt", "publication"), ("selected", "publication")):
            matches = [reason for reason in reasons if fragment in reason]
            if matches:
                return {"category": category, "status": "explained", "reasons": matches}
        if record.get("inventory_state") == "unknown" or any(outcome.get("state") not in {"complete", "excluded"} for outcome in outcomes):
            return {"category": "inventory", "status": "explained", "reasons": ["required_inventory_unknown_or_incomplete"]}
        if "stale" in states:
            return {"category": "freshness", "status": "explained", "reasons": ["required_observation_expired"]}
        if "missing" in states:
            return {"category": "materialization", "status": "explained", "reasons": ["required_materialization_missing"]}
        if field == "indexed" and any(unit.get("achieved_tier") in {"abstract", "metadata"} for unit in evidence):
            return {"category": "tier", "status": "explained", "reasons": ["lower_tier_has_no_headline_fulltext_credit"]}
    # A Boolean old-ruler observation cannot explain its own cause or establish
    # historical input/recipe provenance. Keep that difference explicitly open.
    return {"category": "legacy_provenance", "status": "unresolved",
            "reasons": ["legacy_boolean_has_no_version_qualified_cause"],
            "legacy": before, "version_qualified": after}


def coverage_r2(pools, coverage, *, frozen_snapshot, current_legacy_records=None, measurement_label=None):
    """Compare historic aggregates and independently supplied current old-ruler facts.

    Historical entry-level attribution is unavailable in F0 and remains unknown.
    All entries, including unmatched entries, remain in the denominator.
    """
    selectors = coverage.get("selectors") or {}
    age = coverage.get("max_observation_age")
    if not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0:
        raise ValueError("R2 observation age must be finite and nonnegative")
    if measurement_label is None:
        if any(selectors.get(field) != "current" for field in ("generation_id", "recipe_digest", "consumed_input_revision")):
            raise ValueError("acceptance R2 requires all current selectors")
        if age > 300:
            raise ValueError("acceptance R2 observation age must not exceed 300 seconds")
        measurement_label = "current_version_qualified"
    elif not re.fullmatch(r"[a-z][a-z0-9_]*_v[1-9][0-9]*", measurement_label):
        raise ValueError("alternate R2 measurement requires an explicit versioned label")
    entries_total = sum(len(pool.get("pool") or []) for pool in pools)
    if (len(pools), entries_total) != (12, 111) or payload_hash(pools) != FROZEN_WORKLOAD_DIGEST:
        raise ValueError("R2 requires the frozen 12 pools and 111 entries")
    records = [record for record in coverage["records"] if record.get("kind", "paper") == "paper"]
    by_key = {record["key"]: record for record in records}
    # Missing knowledge identities retain record-local evidence without inventing unit IDs.
    units = {unit["unit_id"] if unit["unit_id"] is not None else ("record", unit["record_key"]): unit
             for unit in coverage["units"]}
    legacy_by_key = {record["key"]: record for record in (current_legacy_records or [])}
    current = {"metadata": 0, "readable": 0, "indexed": 0,
               "lower_tiers": {"abstract": 0, "metadata": 0}, "per_pool": []}
    legacy = {"state": "known" if current_legacy_records is not None else "unknown",
              "matched": 0, "readable": 0, "indexed": 0, "unknown_readable": 0,
              "unknown_indexed": 0, "per_pool": []}
    results, differences = [], []
    for pool in pools:
        entries = pool.get("pool") or []
        keys = match_entries(records, entries)
        old_keys = match_entries(current_legacy_records or [], entries)
        counts = {"qid": pool.get("qid"), "n": len(entries), "metadata": 0, "readable": 0, "indexed": 0,
                  "lower_tiers": {"abstract": 0, "metadata": 0}}
        old_counts = {"qid": pool.get("qid"), "n": len(entries), "matched": 0, "readable": 0, "indexed": 0}
        for index, (entry, key, old_key) in enumerate(zip(entries, keys, old_keys, strict=True)):
            record = by_key.get(key) or {}
            old_record = legacy_by_key.get(old_key) or {}
            evidence_keys = [unit_id if unit_id is not None else ("record", key)
                             for unit_id in record.get("unit_ids", [])]
            evidence = [deepcopy(units[unit_key]) for unit_key in evidence_keys if unit_key in units]
            new = {"metadata": bool(key and record.get("registered")),
                   "readable": bool(key and record.get("readable")), "indexed": bool(key and record.get("indexed"))}
            old = {"metadata": bool(old_key), "readable": old_record.get("readable") if old_key else False,
                   "indexed": old_record.get("indexed") if old_key else False}
            for field in new:
                current[field] += new[field]
                counts[field] += new[field]
            for tier in current["lower_tiers"]:
                current["lower_tiers"][tier] += bool((record.get("tier_coverage") or {}).get(tier))
                counts["lower_tiers"][tier] += bool((record.get("tier_coverage") or {}).get(tier))
            if current_legacy_records is not None:
                legacy["matched"] += old["metadata"]
                old_counts["matched"] += old["metadata"]
                for field in ("readable", "indexed"):
                    if old[field] is None:
                        legacy["unknown_" + field] += 1
                    else:
                        legacy[field] += bool(old[field])
                        old_counts[field] += bool(old[field])
            delta = {field: {"legacy": old[field], "version_qualified": new[field]} for field in new
                     if current_legacy_records is not None and old[field] is not None and old[field] != new[field]}
            result = {"qid": pool.get("qid"), "entry_index": index, "entry": deepcopy(entry),
                      "current_key": key, "legacy_key": old_key if current_legacy_records is not None else "unknown",
                      "historic_match": "unknown", "version_qualified": new,
                      "legacy_evidence": deepcopy(old_record) if current_legacy_records is not None else None,
                      "current_legacy": old if current_legacy_records is not None else "unknown", "units": evidence,
                      "required_unit_outcomes": deepcopy(record.get("required_unit_outcomes", []))}
            result["explanations"] = (["identity_mapping_unmatched"] if not key else []) + [
                {"unit_id": unit["unit_id"], "requested_identity": unit.get("requested_identity"),
                 "selected_identity": unit.get("selected_identity"), "receipt_id": unit.get("receipt_id"),
                 "reasons": unit.get("reasons", []), "roles": unit.get("roles", []),
                 "input_roles": unit.get("input_roles", [])} for unit in evidence]
            results.append(result)
            if delta or (current_legacy_records is not None and key != old_key):
                explanations = {field: _difference_explanation(field, old[field], new[field], evidence,
                    record.get("required_unit_outcomes", []), record) for field in delta}
                if key != old_key:
                    explanations["identity"] = {"category": "identity_mapping", "status": "explained",
                        "current_key": key, "legacy_key": old_key, "reason": "frozen_matcher_selected_different_record"}
                differences.append({**result, "differences": delta, "difference_explanations": explanations})
        current["per_pool"].append(counts)
        if current_legacy_records is not None:
            legacy["per_pool"].append(old_counts)
    if current_legacy_records is None:
        legacy = {"state": "unknown", "reason": "contemporaneous_legacy_observations_not_supplied"}
    return {"workload": {"pools": len(pools), "entries_total": entries_total},
            "evaluation_time": coverage.get("evaluation_time"),
            "frozen_legacy_ruler": {"aggregates": deepcopy(frozen_snapshot["r2_coverage"]),
                                    "entry_provenance": "unknown", "measured_at": frozen_snapshot.get("generated_at")},
            "current_legacy_ruler": legacy, measurement_label: current,
            "coverage_policy": {"selectors": deepcopy(selectors), "max_observation_age": age},
            "aggregate_contexts": deepcopy(coverage.get("aggregate_contexts", {})),
            "unresolved_difference_count": (sum(any(item["status"] == "unresolved" for item in
                delta["difference_explanations"].values()) for delta in differences)
                if current_legacy_records is not None else None),
            "entries": results, "current_differences": differences}
