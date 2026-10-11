"""Explicit correction notifications and a durable, monthly per-cohort sweep."""

from __future__ import annotations

import asyncio
import math
import time
import copy
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from . import concurrency, identity_backfill
from .correction_ledger import CorrectionLedger
from ..download_sources import ads, core
from ..models import Paper
from ..store import Library

TRIGGERS = ("credential_arrived", "source_repaired", "identifiers_arrived")
# CORE's measured low ceiling puts it last on the initial pass (#198 Q1).
# The durable source cursor rotates subsequent passes so it cannot starve.
COHORTS = ("ads", "openalex", "inspire", "arxiv", "semantic_scholar", "publisher", "other", "core")
DEFAULT_COHORT_CAP = 20
MAX_SWEEP_CAP = 100
SWEEP_INTERVAL = timedelta(days=30)
MIN_PACE_SECONDS = 1.0
MAX_ROUTES_PER_PAPER = 3


def _now(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("correction time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _cap(value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"cap must be between 1 and {maximum}")


def _pace(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < MIN_PACE_SECONDS:
        raise ValueError(f"pace_seconds must be at least {MIN_PACE_SECONDS}")


def cohort(paper: Paper) -> str:
    """Local routing labels only; never infer a shared identity from source IDs."""
    valid = core._http_url(paper.url)
    host = urlsplit(valid).hostname if valid else ""
    hosts = {"ui.adsabs.harvard.edu": "ads", "adsabs.harvard.edu": "ads",
             "core.ac.uk": "core", "www.core.ac.uk": "core", "api.core.ac.uk": "core",
             "openalex.org": "openalex", "api.openalex.org": "openalex",
             "inspirehep.net": "inspire", "arxiv.org": "arxiv", "www.arxiv.org": "arxiv"}
    if host in hosts:
        return hosts[host]
    source = paper.source.strip().lower().split("+")[0]
    if source in COHORTS:
        return source
    return "publisher" if valid else "other"


def _papers(library: Library, source: str) -> dict[str, Paper]:
    return {p.key: p for p in library.all_papers() if not p.domain_status and cohort(p) == source}


def _actionable(paper: Paper, routes: dict) -> bool:
    return identity_backfill.eligible(paper) or any(
        f"{paper.key}:{d.url}" not in routes for d in _stored_routes(paper))


def _stored_routes(paper: Paper):
    evidence = paper.ads_availability
    if (not evidence.checked_at or not evidence.bibcode
            or (paper.source == "ads" and paper.paper_id and paper.paper_id != evidence.bibcode)):
        return []
    return [d for d in evidence.documents
            if d.outcome == "pdf" and d.document_kind == "full_text" and d.identity_match
            and not d.identity_rejected and core._http_url(d.url)][:MAX_ROUTES_PER_PAPER]


def _validate_routes(findings: list[dict], papers: dict[str, Paper], keys: list[str], cap: int) -> None:
    if not isinstance(findings, list) or len(findings) > cap * MAX_ROUTES_PER_PAPER:
        raise ValueError("route findings exceed the bounded event packet")
    counts = {}
    for finding in findings:
        if not isinstance(finding, dict) or finding.get("key") not in keys:
            raise ValueError("route finding outside selected event keys")
        key = finding["key"]
        provenance = finding.get("provenance")
        if (not core._http_url(finding.get("url"))
                or not isinstance(provenance, dict)
                or provenance.get("verification") != "remote_pdf"
                or not provenance.get("checked_at")
                or not finding.get("source_identifier")
                or ads._ads_bibcode(papers[key]) != finding["source_identifier"]):
            raise ValueError("route requires matching ADS identity and verified survey provenance")
        _now(datetime.fromisoformat(provenance["checked_at"]))
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > MAX_ROUTES_PER_PAPER:
            raise ValueError("too many routes for one paper")


def _rotated(keys: list[str], after: str) -> list[str]:
    return [k for k in keys if k > after] + [k for k in keys if k <= after]


async def _inspect(papers: list[Paper], cap: int, keys: list[str], pace_seconds: float) -> dict:
    # Discovery owns snapshots only. Join cancellation so no detached discovery
    # remains after the background task's shutdown; application has not started.
    task = asyncio.create_task(asyncio.to_thread(identity_backfill.inspect_identities,
        papers, cap=cap, keys=keys, pace_seconds=pace_seconds))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def _upsert_route(state: dict, record: dict) -> int:
    identifier = f"{record['key']}:{record['url']}"
    old = state["routes"].get(identifier)
    if old is None:
        state["routes"][identifier] = {**record, "observations": [copy.deepcopy(record["provenance"])]}
        return 1
    observations = copy.deepcopy(old.get("observations", [old["provenance"]]))
    if record["provenance"] not in observations:
        observations.append(copy.deepcopy(record["provenance"]))
    if datetime.fromisoformat(record["provenance"]["checked_at"]) > datetime.fromisoformat(old["provenance"]["checked_at"]):
        state["routes"][identifier] = {**record, "observations": observations}
    else:
        old["observations"] = observations
    return 0


def _record_routes(selected: list[Paper], state: dict) -> int:
    """Copy positive stored observations for #174; no survey, fetch, or reset."""
    added = 0
    for paper in selected:
        evidence = paper.ads_availability
        for document in _stored_routes(paper):
            added += _upsert_route(state, {
                "key": paper.key, "doi": paper.doi, "source": "ads", "namespace": "ads_bibcode",
                "source_identifier": evidence.bibcode, "url": document.url, "target_issue": 174,
                "provenance": {"checked_at": evidence.checked_at, "reason": evidence.reason,
                               "document": document.model_dump(), "existing_assets": evidence.existing_assets},
            })
    return added


async def _pass(library: Library, *, trigger: str, source: str, cap: int,
                keys: list[str], pace_seconds: float, now: datetime,
                route_findings: list[dict] | None = None) -> dict:
    async with concurrency.lib_write_lock:
        selected = [library.get(k).model_copy(deep=True) for k in keys if library.get(k)]
        library.log({"event": "correction_start", "trigger": trigger, "source": source,
                     "cap": cap, "keys": keys, "pace_seconds": pace_seconds})
    report = await _inspect(selected, cap, keys, pace_seconds)
    async with concurrency.lib_write_lock:
        before = library.corrections.read()
        snapshots = {p.key: p for p in selected}
        fresh = {p.key for p in selected if library.get(p.key) is not None
                 and library.get(p.key).model_dump(include=identity_backfill.SNAPSHOT_FIELDS)
                 == p.model_dump(include=identity_backfill.SNAPSHOT_FIELDS)}
        with library.corrections.capture_staged() as staged:
            result = identity_backfill.apply_identity_report(library, report, acquire=False, doi_only=True,
                                                             snapshots=snapshots)
        result.update(trigger=trigger, source=source, keys=keys, selected=len(selected),
                      pace_seconds=pace_seconds,
                      deferred_non_doi=sum(r["outcome"] == "deferred_non_doi" for r in result["records"]))
        # Application may have staged conflicts; read within the transaction,
        # then write the monthly aggregates and available routes atomically.
        with library.corrections.transaction() as state:
            result["route_findings"] = _record_routes([library.get(k) for k in fresh], state)
            result["stale_route_findings"] = 0
            for finding in route_findings or []:
                if finding["key"] not in fresh:
                    result["stale_route_findings"] += 1
                    continue
                result["route_findings"] += _upsert_route(state, {**finding, "source": "ads", "namespace": "ads_bibcode",
                    "doi": library.get(finding["key"]).doi, "target_issue": 174})
            result["conflicts_staged"] = len(staged)
            result["conflict_ids"] = sorted(staged)
            result["manifest_ids"] = [mid for mid, manifest in state["manifests"].items()
                if mid not in before["manifests"] and any(c["id"] in staged for c in manifest["conflicts"])]
            month = state["months"].setdefault(now.strftime("%Y-%m"), {})
            for field in ("selected", "scanned", "recovered", "deferred_non_doi", "route_findings", "conflicts_staged"):
                month[field] = month.get(field, 0) + result[field]
            month["runs"] = month.get("runs", 0) + 1
        library.log({"event": "correction_pass", **result})
        return result


@asynccontextmanager
async def _run_gate(library: Library, pace_seconds: float):
    async with library.correction_run_lock:
        delay = library.correction_next_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            yield
        finally:
            library.correction_next_at = time.monotonic() + pace_seconds


async def run_event(library: Library, *, trigger: str, source: str, cap: int,
                    keys: list[str] | None = None, pace_seconds: float = MIN_PACE_SECONDS,
                    now: datetime | None = None, route_findings: list[dict] | None = None) -> dict:
    """Notify affected rows through one serial library-owned correction lane.

    identifiers_arrived names stored rows whose source identifiers/locator URLs
    gained evidence. It never replaces existing DOI/arXiv claims.
    """
    _cap(cap, identity_backfill.MAX_CAP)
    _pace(pace_seconds)
    now = _now(now)
    if trigger not in TRIGGERS or source not in COHORTS:
        raise ValueError("unknown correction trigger or source cohort")
    if trigger == "identifiers_arrived" and not keys:
        raise ValueError("identifiers_arrived requires explicit affected keys")
    async with _run_gate(library, pace_seconds):
        return await _run_event(library, trigger=trigger, source=source, cap=cap, keys=keys,
                                pace_seconds=pace_seconds, now=now, route_findings=route_findings)


async def _run_event(library: Library, *, trigger: str, source: str, cap: int,
                    keys: list[str] | None = None, pace_seconds: float = MIN_PACE_SECONDS,
                    now: datetime | None = None, route_findings: list[dict] | None = None) -> dict:
    papers = _papers(library, source)
    routes = library.corrections.read()["routes"]
    order = (list(dict.fromkeys(keys)) if keys is not None
             else sorted(k for k, p in papers.items() if _actionable(p, routes)))
    selected = [key for key in order if key in papers][:cap]
    _validate_routes([] if route_findings is None else route_findings, papers, selected, cap)
    return await _pass(library, trigger=trigger, source=source, cap=cap,
                       keys=selected, pace_seconds=pace_seconds, now=now, route_findings=route_findings)


def seconds_until_sweep(library: Library, *, now: datetime | None = None) -> float:
    now = _now(now)
    with library.corrections.transaction() as state:
        due = state["schedule"].setdefault("next_due", (now + SWEEP_INTERVAL).isoformat())
    return max(0, (datetime.fromisoformat(due) - now).total_seconds())


async def monthly_sweep(library: Library, *, per_cohort_cap: int = DEFAULT_COHORT_CAP,
                        total_cap: int = MAX_SWEEP_CAP, pace_seconds: float = MIN_PACE_SECONDS,
                        now: datetime | None = None) -> dict:
    _cap(per_cohort_cap, identity_backfill.MAX_CAP)
    _cap(total_cap, MAX_SWEEP_CAP)
    _pace(pace_seconds)
    now = _now(now)
    async with _run_gate(library, pace_seconds):
        return await _monthly_sweep(library, per_cohort_cap=per_cohort_cap,
                                    total_cap=total_cap, pace_seconds=pace_seconds, now=now)


async def _monthly_sweep(library: Library, *, per_cohort_cap: int = DEFAULT_COHORT_CAP,
                        total_cap: int = MAX_SWEEP_CAP, pace_seconds: float = MIN_PACE_SECONDS,
                        now: datetime | None = None) -> dict:
    async with concurrency.lib_write_lock:
        with library.corrections.transaction() as state:
            schedule = state["schedule"]
            due = schedule.get("next_due")
            # A scheduled first sweep may be brought forward by the explicit
            # operator command; subsequent sweeps always respect the 30-day floor.
            if schedule.get("last_sweep") and due and datetime.fromisoformat(due) > now:
                return {"skipped": "not_due", "next_due": due}
            schedule.update(last_sweep=now.isoformat(), next_due=(now + SWEEP_INTERVAL).isoformat())
            start = schedule.get("source_cursor", 0)
            month = state["months"].setdefault(now.strftime("%Y-%m"), {})
            month["sweeps"] = month.get("sweeps", 0) + 1
    passes, remaining = [], total_cap
    order = list(range(start, len(COHORTS))) + list(range(start))
    try:
        for index in order:
            if not remaining:
                break
            source = COHORTS[index]
            papers = _papers(library, source)
            state = library.corrections.read()
            papers = {k: p for k, p in papers.items() if _actionable(p, state["routes"])}
            after = state["schedule"].get("key_cursors", {}).get(source, "")
            selected = _rotated(sorted(papers), after)[:min(per_cohort_cap, remaining)]
            if not selected:
                continue
            if passes:
                await asyncio.sleep(pace_seconds)
            result = await _pass(library, trigger="monthly_sweep", source=source,
                                 cap=min(per_cohort_cap, remaining), keys=selected,
                                 pace_seconds=pace_seconds, now=now)
            passes.append(result)
            remaining -= len(selected)
            with library.corrections.transaction() as state:
                state["schedule"].setdefault("key_cursors", {})[source] = selected[-1]
                state["schedule"]["source_cursor"] = (index + 1) % len(COHORTS)
    except BaseException as exc:
        with library.corrections.transaction() as state:
            state["months"][now.strftime("%Y-%m")]["sweep_error"] = type(exc).__name__
        raise
    result = {"per_cohort_cap": per_cohort_cap, "total_cap": total_cap,
              "selected": total_cap - remaining, "passes": passes,
              "report": monthly_report(library, now.strftime("%Y-%m"))}
    library.log({"event": "correction_monthly_report", **result["report"],
                 "per_cohort_cap": per_cohort_cap, "total_cap": total_cap})
    return result


def monthly_report(library: Library, month: str) -> dict:
    return ledger_report(library.corrections, month)


def ledger_report(ledger: CorrectionLedger, month: str) -> dict:
    state = ledger.read()
    # Historical aggregate counts plus current unresolved disposition; retained
    # needs_human evidence is available in the ledger, not copied into reports.
    return {"month": month, "aggregate": state["months"].get(month, {}),
            "conflicts": ledger.summary(), "route_findings": len(state["routes"]),
            "ledger": str(ledger.path)}


async def monthly_loop(library: Library, log) -> None:
    """Sleep until a durable monthly due time; no short-interval polling."""
    while True:
        await asyncio.sleep(seconds_until_sweep(library))
        try:
            await monthly_sweep(library)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("monthly correction sweep failed; see correction ledger/report")
