"""Extract reference admission and maintenance reducers under the catalog writer lock.

Receipt and source evidence is retained permanently. Only an explicitly disposable,
superseded markdown output can acquire a permanent deletion tombstone.
"""

from datetime import timedelta
import os
from pathlib import Path
from uuid import uuid4

from psycopg import sql

from .extraction import ExtractionStore
from .knowledge_commands import identifier, now, require, revision, state
from .recovery import checked_path


ACTIONS = frozenset({"set_extract_retention", "acquire_maintenance_lease", "renew_maintenance_lease",
    "release_maintenance_lease", "pin_extract", "release_extract_pin", "prepare_extract_deletion",
    "finish_extract_deletion"})
RETENTION_DAYS = 90


def published_extract(k, completion_id, vault, previous):
    root = str(checked_path(Path(vault)))
    k._insert("extract_retention", {"completion_id": completion_id, "vault": root})
    if previous:
        old = k._one("extract_retention", completion_id=previous)
        require(old, "predecessor retention record missing; run the explicit migration")
        k._update("extract_retention", {"superseded_at": now(k), "revision": old["revision"] + 1},
            completion_id=previous)


def admit_completion(k, completion_id):
    if completion_id:
        require(not k._one("extract_deletions", completion_id=completion_id),
                "extract is permanently tombstoned; create a new version")


def _locators(value, root=None):
    """Match physical identities, including lexical and existing symlink aliases.

    Resolution remains meaningful after unlink: a dangling link still resolves
    to the permanently tombstoned name. Invalid/unreadable paths fail closed.
    """
    paths = set()
    for key in ("path", "locator"):
        if isinstance(value.get(key), str) and value[key]:
            path = Path(value[key])
            paths.add(os.path.normpath(value[key]))
            if root or path.is_absolute():
                paths.add(str((Path(root) / path if root else path).resolve(strict=False)))
    return paths


def input_closure(k, manifest):
    """Follow immutable parent inputs and upstream derivations without cycles."""
    pending, seen = [manifest], set()
    while pending:
        current = pending.pop()
        yield current
        revisions = {current.get("parent_consumed_input_revision")}
        for upstream in current.get("upstream_derivation_ids", []):
            derivation = k.get_derivation(upstream)
            if derivation:
                revisions.add(derivation.get("consumed_input_revision"))
        for identity in revisions - seen - {None}:
            seen.add(identity)
            parent = k.get_input(identity)
            if parent:
                pending.append(parent["manifest"])


def admit_input(k, manifest):
    """Every later use of an admitted input rechecks its upstream tombstones."""
    for current in input_closure(k, manifest):
        for upstream in current.get("upstream_derivation_ids", []):
            if upstream.startswith("extraction-derivation:"):
                admit_completion(k, upstream.removeprefix("extraction-derivation:"))
        if current.get("artifacts"):
            for deletion in k._rows("extract_deletions"):
                paths = set().union(*(_locators(item, deletion["vault"]) for item in current["artifacts"]))
                require(not paths.intersection({deletion["output"]["path"],
                    str(Path(deletion["vault"]) / deletion["output"]["path"])}), "input references a tombstoned extract")


def admit_record(k, payload):
    admit_completion(k, payload.get("extraction_completion_id"))
    admit_input(k, {"artifacts": [{"path": payload[field]} for field in
        ("pdf_path", "md_path", "txt_path") if payload.get(field)]})


def manifest_inputs(k, manifest):
    """Include inherited selected receipts, even when desired entries changed."""
    revisions = {entry["consumed_input_revision"] for entry in manifest.get("entries", [])}
    revisions.update(inventory["parent_consumed_input_revision"] for inventory in
        manifest.get("expected_inventories", {}).values() if inventory.get("parent_consumed_input_revision"))
    for entry in manifest.get("entries", []):
        if entry.get("selected_derivation_id"):
            revisions.add(k.get_derivation(entry["selected_derivation_id"])["consumed_input_revision"])
    receipts = set(manifest.get("selected_unit_receipts", {}).values())
    layers = set(manifest.get("aggregate_layers", []))
    for row in k._rows("knowledge_receipts"):
        receipt = row["receipt"]
        if row["receipt_id"] in receipts or receipt["namespace"] in layers:
            revisions.add(receipt["consumed_input_revision"])
    return revisions


def admit_manifest(k, manifest):
    for identity in manifest_inputs(k, manifest):
        admit_input(k, k.get_input(identity)["manifest"])


def _uses(k, manifest, completion_id, paths, *, lineage=True, vault=None):
    return any((lineage and "extraction-derivation:" + completion_id in current.get("upstream_derivation_ids", []))
        or any(paths.intersection(_locators(item, vault)) for item in current.get("artifacts", []))
        for current in (input_closure(k, manifest) if lineage else [manifest]))


def protections(k, completion_id, *, stamp=None):
    """Read current admissions, including older reader manifests and uncertain jobs."""
    stamp = stamp or now(k)
    retained = require(k._one("extract_retention", completion_id=completion_id), "extract retention missing")
    receipt = k.get_extraction_receipt(completion_id)["receipt"]
    path = receipt["output_inventory"][0]["path"]
    paths = {path}
    if retained["vault"]:
        paths.add(str(Path(retained["vault"]) / path))
    reasons = []
    if not retained["disposable"]:
        reasons.append("protected")
    if not retained["superseded_at"] or retained["superseded_at"] + timedelta(days=RETENTION_DAYS) > stamp:
        reasons.append("retention")
    if k._one("extraction_heads", completion_id=completion_id):
        reasons.append("current")
    references = k.connection.execute(sql.SQL("SELECT payload->>'extraction_completion_id' AS completion_id, "
        "payload->>'pdf_path' AS pdf, payload->>'md_path' AS md, payload->>'txt_path' AS txt FROM {} "
        "WHERE payload ?| ARRAY['extraction_completion_id','pdf_path','md_path','txt_path']"
    ).format(k.repository._table("records")))
    referenced = any(r["completion_id"] == completion_id or any(paths.intersection(
        _locators({"path": r[key]}, retained["vault"])) for key in ("pdf", "md", "txt")) for r in references)
    inputs = {r["consumed_input_revision"]: r["manifest"] for r in k._rows("input_manifests")}
    # Direct physical references remain required recovery dependencies even when
    # their job is historical. Copied inputs have a separate physical identity.
    referenced = referenced or any(_uses(k, m, completion_id, paths, lineage=False, vault=retained["vault"]) for m in inputs.values())
    live_inputs = {r["consumed_input_revision"] for r in k.connection.execute(sql.SQL(
        "SELECT q.consumed_input_revision FROM {} q JOIN {} g USING(generation_id) "
        "WHERE g.state IN ('active','candidate') AND NOT q.excluded "
        "UNION SELECT consumed_input_revision FROM {} WHERE kind='distill' AND state IN ('queued','running')"
    ).format(k.repository._table("knowledge_requirements"), k.repository._table("generations"),
        k.repository._table("jobs")))}
    for generation in k._rows("generations"):
        if generation["state"] not in {"active", "candidate"}:
            continue
        for selection in k._rows("selections", generation_id=generation["generation_id"]):
            live_inputs.add(k.get_derivation(selection["derivation_id"])["consumed_input_revision"])
        # An aggregate may inherit output layers that are still part of its
        # physical dependency closure after a desired requirement advances.
        manifest = k.get_manifest(generation["generation_id"])["manifest"]
        inherited = {**manifest, "entries": []}
        live_inputs.update(manifest_inputs(k, inherited))
        for row in k._rows("current_inventories", generation_id=generation["generation_id"]):
            inventory = k.get_inventory(row["parent_unit_id"], row["inventory_revision"])
            if inventory.get("parent_consumed_input_revision"):
                live_inputs.add(inventory["parent_consumed_input_revision"])
    pinned = any(not p["released_at"] and (p["expires_at"] is None or p["expires_at"] > stamp)
                 for p in k._rows("extract_pins", completion_id=completion_id))
    for pin in k._rows("knowledge_pins"):
        if pin["released_at"] or pin["expires_at"] is not None and pin["expires_at"] <= stamp:
            continue
        manifest = k.get_manifest(pin["generation_id"], pin["manifest_revision"]) if pin["generation_id"] else None
        revisions = manifest_inputs(k, manifest["manifest"]) if manifest else set()
        revisions.update(manifest_inputs(k, {"aggregate_layers": pin["namespaces"]}))
        attempt = k.get_attempt(pin["owner"])
        if attempt:
            revisions.add(k.get_job(attempt["job_id"])["consumed_input_revision"])
        pinned = pinned or any(_uses(k, inputs.get(r, {}), completion_id, paths, vault=retained["vault"]) for r in revisions)
    if referenced or any(_uses(k, inputs.get(r, {}), completion_id, paths, vault=retained["vault"]) for r in live_inputs):
        reasons.append("referenced")
    if pinned:
        reasons.append("pinned")
    return reasons


def validate_lease(k, claim, *, resource="extract_gc"):
    current = require(k._one("maintenance_leases", resource=resource), "maintenance lease absent")
    runtime = state(k)
    require(runtime["authority"] == "catalog" and runtime["authority_epoch"] == current["authority_epoch"],
            "maintenance lease authority changed")
    require(all(current[field] == claim.get(field) for field in
        ("resource", "owner", "lease_id", "fencing_token", "authority_epoch", "window_id")), "maintenance lease changed")
    window = k._one("knowledge_maintenance_windows", window_id=current["window_id"])
    stamp = now(k)
    require(window and window["starts_at"] <= stamp < min(window["ends_at"], current["expires_at"]),
            "maintenance lease or window expired")
    return current


def dispatch_lifecycle(k, action, p):
    if action == "set_extract_retention":
        old = require(k._one("extract_retention", completion_id=p["completion_id"]), "extract absent")
        revision(old, p)
        require(type(p.get("disposable")) is bool and p.get("operator_nonce") and p.get("reason"),
                "explicit extract disposition requires reason and operator nonce")
        admit_completion(k, p["completion_id"])
        vault = str(checked_path(Path(p["vault"])))
        require(old["vault"] in {None, vault}, "extract vault differs from publication")
        ExtractionStore(vault).verify(k.get_extraction_receipt(p["completion_id"])["receipt"])
        return k._update("extract_retention", {"disposable": p["disposable"], "vault": vault,
            "revision": old["revision"] + 1}, completion_id=p["completion_id"])

    if action == "acquire_maintenance_lease":
        require(p.get("resource") in {"extract_gc", "rebuild"} and p.get("owner"), "invalid maintenance lease owner/resource")
        window = require(k._one("knowledge_maintenance_windows", window_id=p["window_id"]), "maintenance window absent")
        stamp, runtime = now(k), state(k)
        require(runtime["authority"] == "catalog", "catalog authority not committed")
        require(window["starts_at"] <= stamp < window["ends_at"], "maintenance window not active")
        seconds = p.get("lease_seconds", 300)
        require(type(seconds) is int and 0 < seconds <= 7200, "invalid maintenance lease duration")
        old = k._one("maintenance_leases", resource=p["resource"])
        require(not old or old["expires_at"] <= stamp, "maintenance lease already held")
        values = {"lease_id": uuid4().hex, "owner": p["owner"], "authority_epoch": runtime["authority_epoch"],
            "fencing_token": old["fencing_token"] + 1 if old else 1, "window_id": p["window_id"],
            "expires_at": min(stamp + timedelta(seconds=seconds), window["ends_at"])}
        return k._update("maintenance_leases", values, resource=p["resource"]) if old else k._insert(
            "maintenance_leases", {"resource": p["resource"], **values})

    if action in {"renew_maintenance_lease", "release_maintenance_lease"}:
        current = validate_lease(k, p["lease"], resource=p["lease"]["resource"])
        expiry = now(k)
        if action == "renew_maintenance_lease":
            seconds = p.get("lease_seconds", 300)
            require(type(seconds) is int and 0 < seconds <= 7200, "invalid maintenance lease duration")
            window = k._one("knowledge_maintenance_windows", window_id=current["window_id"])
            expiry = min(expiry + timedelta(seconds=seconds), window["ends_at"])
        return k._update("maintenance_leases", {"expires_at": expiry}, resource=current["resource"])

    if action == "pin_extract":
        require(k.get_extraction_receipt(p["completion_id"]), "extract absent")
        admit_completion(k, p["completion_id"])
        require(p.get("kind") in {"reader", "job", "recovery"} and p.get("owner"), "invalid extract pin")
        expiry = None
        if p["kind"] == "reader":
            seconds = p.get("ttl_seconds", 300)
            require(type(seconds) is int and 0 < seconds <= 7200, "invalid reader pin lifetime")
            expiry = now(k) + timedelta(seconds=seconds)
        return k._insert("extract_pins", {"pin_id": identifier("extract-pin"), "completion_id": p["completion_id"],
            "kind": p["kind"], "owner": p["owner"], "expires_at": expiry})

    if action == "release_extract_pin":
        pin = require(k._one("extract_pins", pin_id=p["pin_id"]), "extract pin absent")
        revision(pin, p)
        require(pin["owner"] == p.get("owner") and not pin["released_at"], "extract pin owner changed or released")
        require(pin["kind"] == "reader" or p.get("disposition") == "reviewed_recovery_disposition"
            and p.get("reason") and p.get("operator_nonce"), "job/recovery pin requires reviewed disposition")
        return k._update("extract_pins", {"released_at": now(k), "revision": pin["revision"] + 1}, pin_id=pin["pin_id"])

    if action in {"prepare_extract_deletion", "finish_extract_deletion"}:
        validate_lease(k, p["lease"])
        completion_id = p["completion_id"]
        retained = require(k._one("extract_retention", completion_id=completion_id), "extract absent")
        vault = str(checked_path(Path(p["vault"])))
        require(retained["vault"] == vault, "extract vault differs from publication")
        deletion = k._one("extract_deletions", completion_id=completion_id)
        reasons = protections(k, completion_id)
        require(not reasons, "extract cannot be deleted: " + ", ".join(reasons))
        if action == "prepare_extract_deletion":
            if deletion:
                return deletion
            receipt = k.get_extraction_receipt(completion_id)["receipt"]
            ExtractionStore(vault).verify(receipt)
            return k._insert("extract_deletions", {"completion_id": completion_id,
                "deletion_id": uuid4().hex, "vault": vault, "output": receipt["output_inventory"][0], "state": "deleting"})
        require(deletion and deletion["deletion_id"] == p["deletion_id"], "deletion identity changed")
        # Finalization may only acknowledge an absent output. Its stable intent
        # remains a reference-admission fence across a lost database response.
        from .extraction_gc import output_directory
        with output_directory(vault, completion_id) as directory:
            try:
                os.stat("output.md", dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                require(False, "extract output still exists")
        if deletion["state"] == "deleted":
            return deletion
        return k._update("extract_deletions", {"state": "deleted", "deleted_at": now(k)}, completion_id=completion_id)
    raise ValueError("unknown extract lifecycle action")
