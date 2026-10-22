"""Named operator evidence, committed with the existing catalog delta batch."""

from __future__ import annotations

from dataclasses import replace

from .correction_ledger import utc_now


def mirror_events(library, rows):
    """JSON compatibility evidence; PostgreSQL remains the authority after commit."""
    if not rows:
        return
    with library.corrections.transaction(durable=True) as state:
        for row in rows:
            name = "policy_rulings" if row["event"] == "policy_ruling" else "operator_events"
            events = state.setdefault(name, [])
            if row not in events:
                events.append(row)


def stage_operator(library, *, action, keys, reason, actor="operator", event="operator_reopen", row=None, detail=None):
    library._check_writer()
    if any(not isinstance(v, str) or not v.strip() for v in (action, reason, actor, event)):
        raise ValueError("operator action requires nonempty action, actor and reason")
    keys = list(dict.fromkeys(keys))
    if not keys or any(library.get(key) is None for key in keys):
        raise ValueError("operator action requires existing record keys")
    row = row or {"timestamp": utc_now(), "action": action, "scope": action,
        "keys": keys, "revision": max(1, *(library.get(k).acquisition_state.required_revision for k in keys)),
        "reason": reason, "actor": actor, "event": event, "detail": detail or {}}
    if library._catalog_backend is not None:
        library._catalog_backend.operator_rulings.append(row)
    else:
        # JSON is the legacy compatibility path: retain intent before state IO.
        mirror_events(library, [row])
        library.log({**row, "event": event})
    return row


def reopen_records(library, papers, *, extraction=False, reason, actor="operator", force_refresh=False):
    """Reopen one explicit selection, preserving policy, aliases and quarantine."""
    from ..state_v2 import Retrieval
    from .extract_defer import clear_extract_deferred

    if not papers:
        return
    stage_operator(library, action="retry_extract" if extraction else "reopen_download",
                   keys=[p.key for p in papers], reason=reason, actor=actor)
    for paper in papers:
        current = paper.unquiesced_state
        paper.download_status = "ok" if extraction else "pending"
        if not extraction:
            paper.terminal_state = replace(current, retrieval=Retrieval.ELIGIBLE, cause="")
        if extraction or force_refresh:
            paper.extract_attempts = 0
            clear_extract_deferred(paper)
        if force_refresh and not paper.extraction_completion_id:
            paper.firecrawl_pdf_hunt_exhausted = False
    library.save(force=True, durable=True)
