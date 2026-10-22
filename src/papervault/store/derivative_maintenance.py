"""Explicit operator plans and bounded GC, candidate fork and promotion commands.

Forking only queues isolated work. The ordinary capacity-gated scheduler executes
it inside the recorded monthly window; selection changes only after promotion.
"""

from pathlib import Path
from uuid import uuid4

from .extract_lifecycle import validate_lease
from .extraction_gc import collect_extract, plan_gc
from .knowledge import ClaimRejected
from .recovery import checked_path, maintenance_deadline
from .repository import AuditEvent


_COMMANDS = {"window": "begin_maintenance_window", "disposition": "set_extract_retention",
             "fork": "fork_generation", "promote": "promote_generation"}


def maintain(k, action, payload, *, vault, operation_id, apply=False, owner=None, window_id=None,
             lease_seconds=300):
    """Plan without writes by default; apply exact caller-supplied revisions.

    A retry uses the same operation ID and payload for fork/promotion/disposition.
    GC retries use permanent per-completion tombstones, even under a new lease.
    No target is inferred from runtime settings or a service dotenv.
    """
    if action not in {*_COMMANDS, "inspect", "gc"}:
        raise ValueError("unknown maintenance action")
    if not isinstance(payload, dict) or not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError("maintenance requires a payload object and stable operation_id")
    root = checked_path(Path(vault))
    if action == "disposition":
        if payload.get("vault", str(root)) != str(root):
            raise ValueError("disposition vault differs from explicit target")
        payload = {**payload, "vault": str(root)}
    if not apply or action == "inspect":
        return k._json({"action": action, "apply": False, "payload": payload,
            "authority": k.get_authority_state(), "window": k.get_active_maintenance_window(),
            "gc": plan_gc(k, root), "generation": k.get_generation(payload["generation_id"])
                if payload.get("generation_id") else None})
    if action in {"window", "disposition"}:
        return {"action": action, "apply": True, "result": k.command(_COMMANDS[action], operation_id, payload)}
    if not owner or not window_id:
        raise ValueError("apply requires an explicit maintenance owner and window_id")
    if action in {"fork", "promote"} and k.get_operation(operation_id) is not None:
        # Replaying a committed outcome is delivery, not new maintenance. The
        # reducer still verifies the operation's action and exact payload hash.
        return {"action": action, "apply": True, "result": k.command(_COMMANDS[action], operation_id, payload)}
    if action == "gc":
        identities = payload.get("completion_ids")
        if (not isinstance(identities, list) or not 1 <= len(identities) <= 100
                or not all(isinstance(value, str) for value in identities) or len(set(identities)) != len(identities)):
            raise ValueError("GC requires 1–100 distinct explicit completion_ids")
    lease = k.command("acquire_maintenance_lease", "maintenance-acquire:" + uuid4().hex,
        {"resource": "extract_gc" if action == "gc" else "rebuild", "window_id": window_id,
         "owner": owner, "lease_seconds": lease_seconds})
    try:
        if action == "gc":
            result = [collect_extract(k, root, identity, lease) for identity in identities]
        else:
            # Keep lease validation and the existing revision/manifest reducer in
            # one writer transaction. Expiration during validation rolls it back.
            with k.repository._write(AuditEvent("maintenance", action)):
                current = validate_lease(k, lease, resource="rebuild")
                seconds = (current["expires_at"] - k.connection.execute(
                    "SELECT clock_timestamp() AS now").fetchone()["now"]).total_seconds()
                with maintenance_deadline(seconds):
                    result = k.command(_COMMANDS[action], operation_id, payload)
                    validate_lease(k, lease, resource="rebuild")
        return {"action": action, "apply": True, "result": result}
    finally:
        try:
            k.command("release_maintenance_lease", "maintenance-release:" + uuid4().hex, {"lease": lease})
        except ClaimRejected:
            pass  # Expired/replaced leases cannot be released by the old owner.
