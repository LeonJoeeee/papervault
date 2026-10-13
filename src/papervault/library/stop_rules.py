"""Executable A5 safety rules, with pure detectors and existing locked actuators.

Each round replays the durable ledger once and snapshots bounded send windows.
It never scans manifest history or treats a rejected request as an emitted send.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from papervault.config import polite_v2_enabled, stoprules_v2_enabled

from .claims_v2 import BudgetLedger, ClaimStore, ReopenClaim, _outstanding, record_generation
from .politeness_v2 import QuotaSnapshot, Scope

DUPLICATE_INFLIGHT_THRESHOLD = 0
QUOTA_WINDOW_VIOLATION_THRESHOLD = 0
IDENTITY_MISMATCH_PUBLICATION_THRESHOLD = 0
# Measurement definition only: never enforce an unmeasured fourth stop rule.
SIGNIFICANT_REGRESSION_FRACTION = 0.20


@dataclass(frozen=True)
class InFlight:
    key: str
    identity: str
    campaign_id: str
    lease_id: str


@dataclass(frozen=True)
class Publication:
    claim: ReopenClaim | None
    identity_matches: bool
    key: str = ""
    generation: int = 0


@dataclass(frozen=True)
class Finding:
    rule: str
    campaign_ids: tuple[str, ...]
    keys: tuple[str, ...] = ()
    scope: Scope | None = None


def adopted_identity(paper) -> str:
    """Compare already-adopted identities; this never resolves new evidence."""
    from .identity_v2 import select_canonical
    identity = select_canonical({name: getattr(paper, name) for name in
                                 ("doi", "arxiv_id", "inspire_record_id", "paper_id", "source", "url")})
    return f"{identity.rung.name.lower()}:{identity.key}" if identity else "key:" + paper.key


def detect_duplicate_inflight(attempts) -> tuple[Finding, ...]:
    by_identity = defaultdict(dict)
    for attempt in attempts:
        # Same lease observed by two instruments is still one executor.
        by_identity[attempt.identity][(attempt.key, attempt.lease_id)] = attempt
    findings = []
    for values in by_identity.values():
        if len(values) - 1 > DUPLICATE_INFLIGHT_THRESHOLD:
            findings.append(Finding("duplicate_inflight_identity",
                tuple(sorted({a.campaign_id for a in values.values() if a.campaign_id})),
                tuple(sorted({a.key for a in values.values()}))))
    return tuple(findings)


def quota_violation_observations(windows, sends):
    """Find violating emitted-send endpoints, including windows ending before now."""
    stamps = sorted(sends)
    observations = []
    for window in windows:
        start = 0
        for end, at in enumerate(stamps):
            while stamps[start] <= at - window.seconds:
                start += 1
            count = end - start + 1
            if count > window.requests:
                observations.append((window.seconds, at, count))
    return tuple(observations)


def detect_quota_violations(snapshots: dict[Scope, QuotaSnapshot], campaigns) -> tuple[Finding, ...]:
    findings = []
    for scope, snapshot in snapshots.items():
        violations = (snapshot.violations - snapshot.handled if snapshot.violations else
                      len(quota_violation_observations(snapshot.windows, snapshot.sends)))
        if violations > QUOTA_WINDOW_VIOLATION_THRESHOLD:
            findings.append(Finding("quota_window_violation", tuple(sorted(campaigns)), scope=scope))
    return tuple(findings)


def detect_publication_mismatches(publications) -> tuple[Publication, ...]:
    mismatches = tuple(p for p in publications if p.identity_matches is False)
    return mismatches if len(mismatches) > IDENTITY_MISMATCH_PUBLICATION_THRESHOLD else ()


def _inflight(library, rows):
    keys = {row["claim"]["key"] for row in rows if "claim" in row}
    redirects = {row["absorbed_key"]: row["canonical_key"] for row in rows if row["event"] == "merge"}
    def identity(key):
        seen = set()
        while key in redirects and key not in seen:
            seen.add(key)
            key = redirects[key]
        from .merge_v2 import resolve_alias
        paper = resolve_alias(library, key)
        return adopted_identity(paper) if paper else "key:" + key
    attempts = [InFlight(t.key, identity(t.key), t.campaign_id, t.lease_id)
                for t in _outstanding(rows, keys)]
    # This registry counts executors that entered without a campaign claim.
    # Later claim adoption cannot erase their still-active resource count.
    with library._merge_execution_lock:
        for key, count in getattr(library, "_stoprule_active", {}).items():
            paper = library.get(key)
            if paper:
                attempts.extend(InFlight(key, identity(key), "", f"executor:{n}") for n in range(count))
    return attempts


async def evaluate_round(library, *, ledger=None, accounting=None, inflight=None,
                         publications=None) -> tuple[Finding, ...]:
    """Apply named resulting states; OFF returns before reading any input."""
    if not stoprules_v2_enabled():
        return ()
    ledger = ledger or BudgetLedger(library.root)
    store = ClaimStore(library, ledger=ledger)
    # Serialize snapshot and quarantine with the existing writer, never await
    # a second acquisition of that lock from an actuator.
    async with store.lock:
        rows = ledger.events()
        starts = {r["campaign_id"] for r in rows if r["event"] == "campaign_start"}
        finishes = {r["campaign_id"] for r in rows if r["event"] == "campaign_finish"}
        campaigns = starts - finishes
        attempts = _inflight(library, rows) if inflight is None else inflight
        if accounting is None and polite_v2_enabled():
            from .polite_http import get_ledger
            accounting = get_ledger()
        quotas = accounting.quota_snapshots() if accounting is not None else {}
        from .download_telemetry import manifest_lock
        with manifest_lock:
            pending = getattr(library, "_stoprule_publications", {})
            batch = tuple(pending.items()) if publications is None else ()
            observed = tuple(value for _, value in batch) if publications is None else publications
        mismatches = detect_publication_mismatches(observed)
        findings = (*detect_duplicate_inflight(attempts), *detect_quota_violations(quotas, campaigns),
                    *(Finding("identity_mismatched_publication",
                              (p.claim.campaign_id,) if p.claim else tuple(sorted(campaigns)),
                              (p.claim.key if p.claim else p.key,))
                      for p in mismatches))
        for finding in findings:
            for campaign in finding.campaign_ids:
                ledger.halt_campaign(campaign, finding.rule)
            if finding.scope is not None:
                accounting.enter_cooldown(finding.scope, finding.rule)
        for publication in mismatches:
            paper = store.current(publication.claim) if publication.claim else library.get(publication.key)
            if publication.claim is None and record_generation(paper) != publication.generation:
                continue
            if paper is not None and not paper.acquisition_state.quarantine:
                store._quarantine(paper, "identity_mismatched_publication")
        if publications is None:
            with manifest_lock:
                for key, publication in batch:
                    if pending.get(key) is publication:
                        pending.pop(key)
        return tuple(findings)
