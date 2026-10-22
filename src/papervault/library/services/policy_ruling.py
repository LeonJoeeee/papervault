"""Scoped, revisioned policy acts over the current catalog authority."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from filelock import FileLock

from ..state_v2 import Merge, Policy, PolicyOverride, Retrieval
from ..store import Library
from .correction_ledger import CorrectionLedger, utc_now
from .identity_backfill import read_identity_records
from .operator_actions import stage_operator

ACTIONS = {"acquire": "acquire", "override": "acquire", "exclude": "exclude",
           "restrict": "exclude", "clear": "clear"}


def _rulings(state: dict) -> list[dict]:
    rows = state.get("policy_rulings", [])
    if not isinstance(rows, list):
        raise ValueError("invalid policy ruling ledger")
    for row in rows:
        if (not isinstance(row, dict) or row.get("action") not in ACTIONS
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or type(row.get("revision")) is not int or row["revision"] < 1
                or not isinstance(row.get("keys"), list) or not row["keys"]
                or any(not isinstance(k, str) or not k for k in row["keys"])
                or not isinstance(row.get("timestamp"), str)):
            raise ValueError("invalid policy ruling ledger row")
        if datetime.fromisoformat(row["timestamp"]).tzinfo is None:
            raise ValueError("policy ruling timestamp must be timezone-aware")
    return rows


def _history(root, library=None):
    if os.environ.get("PAPERVAULT_CATALOG_AUTHORITY", "json") == "postgres":
        owned = library is None
        library = library or Library(root, read_only=True)
        try:
            with library._catalog_backend.repository() as repository:
                return [r for r in repository.list_rulings() if r["action"] in ACTIONS]
        finally:
            if owned:
                library.close()
    return _rulings(CorrectionLedger(root).read())


def _mismatch(paper) -> str:
    state = paper.acquisition_state
    if (state.merge != Merge.CANONICAL or state.alias_key or state.quarantine
            or paper.merge_intent is not None
            or (paper.reopen_claim and paper.reopen_claim.phase not in {"completed", "not_sent", "released"})):
        return "unsafe_merge_or_claim_state"
    if type(state.required_revision) is not int or state.required_revision < 0:
        return "invalid_required_revision"
    return ""


def _inspect(papers, rows, *, keys, action, reason):
    if action not in ACTIONS or not isinstance(reason, str) or not reason.strip():
        raise ValueError("policy ruling requires a supported action and a nonempty reason")
    if not keys or any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError("policy ruling requires nonempty citation keys")
    action = ACTIONS[action]
    matched, missing, mismatches, records = [], [], [], []
    revision = 1
    for key in dict.fromkeys(keys):
        paper = papers.get(key)
        if paper is None:
            missing.append(key)
            records.append({"key": key, "policy": None, "outcome": "missing"})
            continue
        current = paper.acquisition_state
        mismatch = _mismatch(paper)
        records.append({"key": key, "policy": current.policy.value,
                        "required_revision": current.required_revision,
                        "outcome": "mismatch" if mismatch else "matched"})
        if mismatch:
            mismatches.append({"key": key, "reason": mismatch})
            continue
        matched.append(key)
        prior = next((r for r in reversed(rows) if key in r["keys"]), None)
        same = prior and ACTIONS[prior["action"]] == action and prior["reason"] == reason
        if same and prior["revision"] >= current.required_revision:
            proposed = prior["revision"]
        else:
            proposed = max(1, current.required_revision + 1,
                           current.override.revision + 1 if current.override else 1,
                           prior["revision"] + 1 if prior else 1)
        revision = max(revision, proposed)
    return {"dry_run": True, "action": action, "reason": reason, "revision": revision,
            "requested": len(dict.fromkeys(keys)), "matched_keys": matched, "matched_count": len(matched),
            "missing_keys": missing, "missing_count": len(missing),
            "mismatches": mismatches, "mismatch_count": len(mismatches), "records": records,
            "applied": 0, "rulings_added": 0}


def inspect_ruling(root: Path, *, keys: list[str], action: str, reason: str) -> dict:
    """Inspect authority without creating locks, projections, or directories."""
    papers = {p.key: p for p in read_identity_records(Path(root))}
    return _inspect(papers, _history(root), keys=keys, action=action, reason=reason)


def apply_ruling(root: Path, *, keys: list[str], action: str, reason: str, actor="operator") -> dict:
    """Commit a whole explicit selection, ruling, audit and pending work together.

    JSON compatibility retains intent before state, allowing interrupted saves to
    reuse that exact evidence. PG rulings are authoritative in the transaction;
    the file ledger is a compatibility projection written only after commit.
    """
    report = inspect_ruling(root, keys=keys, action=action, reason=reason)
    report["dry_run"] = False
    if report["missing_count"] or report["mismatch_count"]:
        return report
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("policy ruling requires a nonempty actor")
    with FileLock(str(Path(root) / ".reopen-campaign.lock"), timeout=0):
        library = Library(root)
        try:
            rows = _history(root, library)
            report = _inspect(library._papers, rows, keys=keys, action=action, reason=reason)
            report["dry_run"] = False
            if report["missing_count"] or report["mismatch_count"]:
                return report
            revision, action = report["revision"], report["action"]
            new_keys = []
            for key in report["matched_keys"]:
                prior = next((r for r in reversed(rows) if key in r["keys"]), None)
                if not (prior and ACTIONS[prior["action"]] == action
                        and prior["reason"] == reason and prior["revision"] == revision):
                    new_keys.append(key)
            if new_keys:
                row = {"timestamp": utc_now(), "action": action, "scope": "acquire" if action == "acquire" else "policy",
                       "reason": reason, "keys": new_keys, "revision": revision,
                       "covered_policy_reasons": {k: list(library.get(k).acquisition_state.policy_reasons)
                                                  for k in new_keys},
                       "actor": actor, "event": "policy_ruling"}
                stage_operator(library, action=action, keys=new_keys, reason=reason, actor=actor,
                               event="policy_ruling", row=row)
                report["rulings_added"] = 1
            for key in report["matched_keys"]:
                paper = library.get(key)
                current = paper.acquisition_state
                if key not in new_keys:
                    # PG intents are atomic. JSON intent precedes state: a
                    # strictly newer installed revision proves application.
                    # Never reapply it to restrictions observed afterward.
                    if library._catalog_backend is not None or (
                            paper.terminal_state and paper.terminal_state.required_revision >= revision):
                        continue
                intent = row if key in new_keys else next(r for r in reversed(rows) if key in r["keys"])
                covered = tuple(intent.get("covered_policy_reasons", {}).get(key, ()))
                if key not in new_keys and current.policy_reasons != covered:
                    raise ValueError("policy restrictions changed since intent; record a new ruling reason")
                if action == "acquire":
                    grant = PolicyOverride(scope=("acquire",), reasons=covered,
                                           revision=revision, key=key)
                    fresh_grant = current.policy != Policy.OVERRIDE or current.override != grant
                    desired = replace(current, policy=Policy.OVERRIDE,
                        retrieval=(Retrieval.ELIGIBLE if fresh_grant and paper.download_status in {"metadata_only", "failed", "pending"}
                                   else current.retrieval), required_revision=revision,
                        override=grant)
                else:
                    if action == "exclude" and paper.download_status != "metadata_only":
                        paper.download_status = "metadata_only"
                    desired = replace(paper.acquisition_state,
                        policy=Policy.EXCLUDED if action == "exclude" else Policy.CLEAR,
                        policy_reasons=current.policy_reasons if action == "exclude" else (),
                        override=None, required_revision=revision)
                if paper.terminal_state != desired:
                    paper.terminal_state = desired
                    report["applied"] += 1
            if report["applied"] or report["rulings_added"]:
                # On JSON replay the intent already exists. PG never has a
                # committed ruling without its state; stale snapshots fail CAS.
                library.save(force=True, durable=True)
            return report
        finally:
            library.close()
