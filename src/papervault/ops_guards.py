"""Shared systemd co-tenancy gate for maintenance-window ops (issue #32 + PR #54).

Two operator entry points must REFUSE to run while the live papervault service (or its
MinerU unit) is systemd-active, each for its own reason but with IDENTICAL probe mechanics:

  - the downstream eval (``eval/run_eval.py``): an eval process loads its own reranker +
    embedder — an un-budgeted third GPU tenant next to a running service (OOM), and running
    it mid-flight breaks the corpus-freeze discipline.
  - operator-doc ingest (``papervault ingest-doc``): a maintenance-window write to the SAME
    ledger + LightRAG graph the live service is concurrently serving — a co-writer risks
    DB co-write corruption.

The freeze discipline (stop services before the op) becomes code here. The systemd probe is
the single source of truth; each caller supplies its own override env + risk ``reason`` so
the two refusals read correctly while sharing one gate (no copy-paste drift on the unit
names / permissive-CI break / 'activating'-counts semantics).
"""
from __future__ import annotations

import os
import subprocess
import sys


def active_service_units() -> list[str]:
    """papervault service units that are systemd-active (or *activating*), else [].

    Probes ``papervault.service`` + the MinerU unit (operator override
    ``PAPER_LIBRARY_MINERU_UNIT``; the hardcoded default would no-op the gate on a renamed
    deploy). ``FileNotFoundError`` (no systemd at all — CI / container) breaks the loop
    permissively (empty list → callers proceed). A transient probe failure on one unit is
    swallowed and the scan continues, but a positive already in hand is NEVER discarded
    (fail-loud philosophy).
    """
    units = ("papervault.service",
             os.environ.get("PAPER_LIBRARY_MINERU_UNIT", "papervault-mineru.service"))
    active: list[str] = []
    for unit in units:
        try:
            r = subprocess.run(["systemctl", "--user", "is-active", unit],
                               capture_output=True, text=True, timeout=10)
            if r.stdout.strip() in ("active", "activating"):
                active.append(unit)
        except FileNotFoundError:
            break  # no systemd at all (CI/container): empty `active` falls through permissively
        except Exception:  # noqa: BLE001 — transient probe failure: keep checking; never
            continue      # discard a positive already in hand (fail-loud philosophy)
    return active


def require_services_stopped(*, override_env: str, reason: str) -> None:
    """Refuse (``SystemExit(2)``) if a papervault service unit is systemd-active.

    ``override_env`` = the caller's deliberate-co-residency escape hatch (set to ``1`` to
    proceed anyway). ``reason`` = one sentence explaining WHY co-tenancy is unsafe for this
    caller; it is printed to stderr on abort, framed by the shared ABORT header + the
    stop-the-service / override remediation. Fail-loud, same philosophy as the eval
    workspace gate.
    """
    if os.environ.get(override_env) == "1":
        sys.stderr.write(f"NOTE: co-tenancy override active ({override_env}=1).\n")
        return
    active = active_service_units()
    if active:
        sys.stderr.write(
            f"ABORT: {', '.join(active)} is RUNNING — {reason} "
            f"Stop the service first, or set {override_env}=1 for a deliberate override.\n"
        )
        raise SystemExit(2)
