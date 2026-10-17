"""Explicit ADS UNKNOWN-policy rulings, recorded before scoped grants persist."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from filelock import FileLock

from ..download_sources.ads import _ads_bibcode
from ..state_v2 import Merge, Policy, PolicyOverride, Retrieval, check_index_readable
from ..store import Library
from .correction_ledger import CorrectionLedger, utc_now
from .identity_backfill import read_identity_records


def _rulings(state: dict) -> list[dict]:
    rows = state.get("policy_rulings", [])
    if not isinstance(rows, list):
        raise ValueError("invalid policy ruling ledger")
    for row in rows:
        if (not isinstance(row, dict) or row.get("action") != "acquire"
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or type(row.get("revision")) is not int or row["revision"] < 1
                or not isinstance(row.get("keys"), list) or not row["keys"]
                or any(not isinstance(k, str) or not k for k in row["keys"])
                or not isinstance(row.get("timestamp"), str)):
            raise ValueError("invalid policy ruling ledger row")
        if datetime.fromisoformat(row["timestamp"]).tzinfo is None:
            raise ValueError("policy ruling timestamp must be timezone-aware")
    return rows


def _mismatch(paper) -> str:
    state = paper.acquisition_state
    if paper.source != "ads" or paper.doi or paper.arxiv_id:
        return "outside_ads_identifier_cohort"
    if (not paper.url or not paper.paper_id or _ads_bibcode(paper) != paper.paper_id
            or not re.fullmatch(r"/abs/[^/]+/abstract/?", urlsplit(paper.url).path)):
        return "ads_identity_mismatch"
    if paper.download_status != "metadata_only" or state.retrieval != Retrieval.EXHAUSTED:
        return "not_metadata_only_exhausted"
    if (paper.domain_status is not None or state.policy_reasons
            or state.policy not in {Policy.UNKNOWN, Policy.OVERRIDE}):
        return "policy_restriction"
    if (state.merge != Merge.CANONICAL or state.alias_key or state.quarantine
            or paper.merge_intent is not None
            or (paper.reopen_claim and paper.reopen_claim.phase not in {"completed", "not_sent", "released"})):
        return "unsafe_merge_or_claim_state"
    if type(state.required_revision) is not int or state.required_revision < 0:
        return "invalid_required_revision"
    if state.policy == Policy.OVERRIDE:
        grant = state.override
        if (not isinstance(grant, PolicyOverride) or grant.key != paper.key
                or grant.scope != ("acquire",) or grant.reasons
                or type(grant.revision) is not int or grant.revision < 1):
            return "different_override_scope"
    return ""


def inspect_ruling(root: Path, *, keys: list[str], action: str, reason: str) -> dict:
    """Inspect complete records without constructing a writable Library or lock."""
    if action != "acquire" or not isinstance(reason, str) or not reason.strip():
        raise ValueError("policy ruling requires action acquire and a nonempty reason")
    if not keys or any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError("policy ruling requires nonempty citation keys")
    check_index_readable(root)
    papers = {p.key: p for p in read_identity_records(Path(root))}
    rows = _rulings(CorrectionLedger(root).read())
    matched, missing, mismatches, records = [], [], [], []
    revision = 1
    for key in dict.fromkeys(keys):
        paper = papers.get(key)
        if paper is None:
            missing.append(key)
            records.append({"key": key, "policy": None, "outcome": "missing"})
            continue
        state = paper.acquisition_state
        mismatch = _mismatch(paper)
        records.append({"key": key, "policy": state.policy.value,
                        "required_revision": state.required_revision,
                        "outcome": "mismatch" if mismatch else "matched"})
        if mismatch:
            mismatches.append({"key": key, "reason": mismatch})
            continue
        matched.append(key)
        revision = max(revision, state.required_revision,
                       state.override.revision if state.override else 0)
    revision = max([revision, *(r["revision"] for r in rows if any(k in matched for k in r["keys"]))])
    return {"dry_run": True, "action": action, "reason": reason, "revision": revision,
            "requested": len(dict.fromkeys(keys)), "matched_keys": matched, "matched_count": len(matched),
            "missing_keys": missing, "missing_count": len(missing),
            "mismatches": mismatches, "mismatch_count": len(mismatches), "records": records,
            "applied": 0, "rulings_added": 0}


def apply_ruling(root: Path, *, keys: list[str], action: str, reason: str) -> dict:
    """Append evidence first; interrupted index writes can replay without new rows.

    The CLI owns the stopped-service preflight. Standalone campaigns and rulings
    share one writer lock; no invalid request can partially grant permission.
    """
    report = inspect_ruling(root, keys=keys, action=action, reason=reason)
    report["dry_run"] = False
    if report["missing_count"] or report["mismatch_count"]:
        return report
    with FileLock(str(Path(root) / ".reopen-campaign.lock"), timeout=0):
        report = inspect_ruling(root, keys=keys, action=action, reason=reason)
        report["dry_run"] = False
        if report["missing_count"] or report["mismatch_count"]:
            return report
        library = Library(root)
        revision = report["revision"]
        with library.corrections.transaction(durable=True) as state:
            rows = _rulings(state)
            recorded = {(k, row["action"], row["revision"]) for row in rows for k in row["keys"]}
            new_keys = [k for k in report["matched_keys"] if (k, action, revision) not in recorded]
            if new_keys:
                state.setdefault("policy_rulings", []).append({
                    "timestamp": utc_now(), "action": action, "reason": reason,
                    "keys": new_keys, "revision": revision})
                report["rulings_added"] = 1
        for key in report["matched_keys"]:
            paper = library.get(key)
            current = paper.acquisition_state
            desired = replace(current, policy=Policy.OVERRIDE,
                              override=PolicyOverride(scope=(action,), revision=revision, key=key))
            if current != desired:
                paper.terminal_state = desired
                report["applied"] += 1
        if report["applied"]:
            library.save(force=True, durable=True)
    return report
