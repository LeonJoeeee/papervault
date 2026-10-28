"""Pure acquisition state contracts (accepted v2 spec, Appendix A1/A3).

These dimensions do not replace the extraction axis or the public statuses.
Legacy state is a read-time projection, never a migration. Lease and budget
providers consult durable campaign claims when present; ordinary acquisition
keeps its existing in-memory coordination.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from os import fstat
from pathlib import Path
from typing import TYPE_CHECKING

from .compat_v2 import INDEX_FORMAT_VERSION, check_readable, check_versions

if TYPE_CHECKING:
    from .models import Paper


def _index_file_stamp(path: Path):
    """Detect rewrites/truncation without reading the full catalog on each hit."""
    try:
        with path.open("rb") as stream:
            stat = fstat(stream.fileno())
            head = stream.read(4096)
            stream.seek(max(0, stat.st_size - 4096))
            tail = stream.read(4096)
        return stat.st_mtime_ns, stat.st_size, head, tail
    except OSError:
        return None


@lru_cache(maxsize=64)
def _index_header(reader, root: Path, primary_stamp, backup_stamp):
    # Stamps and reader identity are cache keys; retain only the small header.
    data = reader(root / "index.json") or reader(root / "index.json.bak")
    if data is None:
        return None
    return data.get("version", INDEX_FORMAT_VERSION), data.get("acquisition_versions", {})


def check_index_readable(root) -> None:
    """Fence v2 entry points using the raw primary/last-good index headers.

    This does not change the legacy index reader or rewrite any fields. Checking
    before typed records are used also prevents an unsupported v2 field from
    being treated as a corrupt row and silently omitted by the legacy loader.
    """
    from .store import Library
    root = Path(root)
    header = _index_header(Library._read_index_file, root,
                           _index_file_stamp(root / "index.json"),
                           _index_file_stamp(root / "index.json.bak"))
    if header is None:
        return  # absence/corruption remains the existing reader's responsibility
    version, versions = header
    check_readable(version, surface="index")
    check_versions(versions)


class Retrieval(str, Enum):
    INELIGIBLE = "ineligible"
    ELIGIBLE = "eligible"
    IN_PROGRESS = "in-progress"
    EXHAUSTED = "exhausted"
    ACQUIRED = "acquired"
    VERIFIED = "verified"


class Policy(str, Enum):
    CLEAR = "clear"
    EXCLUDED = "excluded"
    OVERRIDE = "override"
    UNKNOWN = "unknown"


class Merge(str, Enum):
    CANONICAL = "canonical"
    ALIAS = "alias"


@dataclass(frozen=True)
class PolicyOverride:
    scope: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    revision: int = 0
    key: str = ""  # grants cannot be transferred to a different canonical key


@dataclass(frozen=True)
class TerminalState:
    retrieval: Retrieval = Retrieval.INELIGIBLE
    policy: Policy = Policy.UNKNOWN
    merge: Merge = Merge.CANONICAL
    quarantine: bool = False
    alias_key: str = ""
    policy_reasons: tuple[str, ...] = ()
    override: PolicyOverride | None = None
    required_revision: int = 0
    cause: str = ""  # empty means the legacy terminal cause was not recorded


def _field(record, name: str, default=None):
    return record.get(name, default) if isinstance(record, dict) else getattr(record, name, default)


def extraction_evidence(paper) -> bool:
    """A recorded, labelled markdown extract is existing verification evidence."""
    return bool(_field(paper, "md_path") and _field(paper, "md_engine"))


def project_domain_policy(state: TerminalState, paper) -> TerminalState:
    """Project clean-view exclusions into acquisition policy, retaining artifacts.

    The three documented domain markers also exclude identity recovery. Unknown
    markers fail closed. This pure projection does not clear stored restrictions
    or quarantine; scoped overrides retain their grant and must cover the added
    reason. Callers choose the identity-v2 flag at their read boundary.
    """
    marker = _field(paper, "domain_status")
    if marker is None:
        return state
    known = isinstance(marker, str) and marker in {"off_domain", "non_paper", "bad_extract"}
    reason = marker if known else "unknown_domain_status"
    policy = (Policy.OVERRIDE if state.policy == Policy.OVERRIDE else Policy.EXCLUDED) if known else Policy.UNKNOWN
    return replace(state, policy=policy,
                   policy_reasons=tuple(dict.fromkeys((*state.policy_reasons, reason))))


def derive_legacy(paper, *, include_domain: bool = False) -> TerminalState:
    """Conservatively project the five live statuses, without mutating a row.

    Every cause-unrecorded terminal, including acquired/extract-failed, has
    unknown policy. Extraction failure never means retrieval failure. With the
    opt-in domain projection, a recorded domain exclusion establishes policy
    independently of retrieval (including acquired/verified artifacts).
    """
    status = _field(paper, "download_status")
    if status == "pending":
        state = TerminalState(retrieval=Retrieval.ELIGIBLE, policy=Policy.CLEAR)
    elif status == "ok":
        state = TerminalState(retrieval=(Retrieval.VERIFIED if extraction_evidence(paper)
                                        else Retrieval.ACQUIRED))
    elif status == "extract_failed":
        state = TerminalState(retrieval=Retrieval.ACQUIRED)
    elif status in ("metadata_only", "failed"):
        state = TerminalState(retrieval=Retrieval.EXHAUSTED)
    else:
        state = TerminalState()
    return project_domain_policy(state, paper) if include_domain else state


def after_status_write(state: TerminalState, paper) -> TerminalState:
    """Record a known status transition, preserving every non-retrieval axis."""
    projected = derive_legacy(paper)
    status = _field(paper, "download_status")
    cause = {"ok": "acquired", "extract_failed": "extraction_failed",
             "metadata_only": "retrieval_exhausted", "failed": "retrieval_exhausted"}
    return replace(state, retrieval=projected.retrieval,
                   cause=cause.get(status, "") if isinstance(status, str) else "")


def resolved_identity(paper: Paper) -> bool:
    """Use already-adopted scalar identifiers; observations are not verdicts.

    Slice 2 supplies correspondence resolution. This adapter neither chooses
    among conflicting source observations nor adopts a new identifier.
    """
    return bool(paper.doi or paper.arxiv_id or paper.paper_id
                or paper.inspire_record_id or paper.url)


def effective_policy(state, *, key: str = "", action: str = "acquire",
                     required_revision: int = 0) -> bool:
    """Allow clear policy without exclusions, or a valid scoped override.

    A grant covers every recorded exclusion and is bound to the original
    canonical key. Copying it during metadata fusion cannot amplify permission.
    Unknown policy always denies; only a separate ruling may resolve it.
    """
    if (not isinstance(state, TerminalState)
            or not _valid_scope(state.policy_reasons)):
        return False
    if state.policy == Policy.CLEAR:
        return not state.policy_reasons
    if state.policy != Policy.OVERRIDE or not isinstance(state.override, PolicyOverride):
        return False
    grant = state.override
    if (type(grant.revision) is not int or type(required_revision) is not int
            or type(state.required_revision) is not int
            or not _valid_scope(grant.scope) or not _valid_scope(grant.reasons)):
        return False
    return bool(
        key and grant.key == key
        and action in grant.scope
        and grant.revision >= max(0, required_revision, state.required_revision)
        and all(reason in grant.reasons for reason in state.policy_reasons)
    )


def _valid_scope(values) -> bool:
    return isinstance(values, tuple) and all(isinstance(value, str) and value for value in values)


def retain_observed_restrictions(current, observed) -> TerminalState:
    """Retain constraints during metadata fusion, without moving retrieval.

    Missing dimensions on disk still carry the conservative legacy policy.
    Metadata observation cannot resolve unknown policy or release quarantine.
    This preserves fields in the existing merge path; it performs no identity
    merge, redirect, drain, or retrieval transition.
    """
    if not isinstance(current, TerminalState) or not isinstance(observed, TerminalState):
        return TerminalState()
    left, right = current.override, observed.override
    required_revision = max([0, *(s.required_revision for s in (current, observed)
                                 if type(s.required_revision) is int)])
    # A discarded grant's revision remains spent, including legacy grants whose
    # writer did not advance required_revision when installing the override.
    dropped_revision = max([required_revision, *(g.revision for g in (left, right)
        if isinstance(g, PolicyOverride) and type(g.revision) is int and g.revision > 0)])
    if (not _valid_scope(current.policy_reasons) or not _valid_scope(observed.policy_reasons)
            or type(current.required_revision) is not int or type(observed.required_revision) is not int
            or current.policy not in tuple(Policy) or observed.policy not in tuple(Policy)):
        return replace(current, policy=Policy.UNKNOWN, override=None,
                       required_revision=dropped_revision)
    reasons = tuple(dict.fromkeys((*current.policy_reasons, *observed.policy_reasons)))
    grant = None
    if (current.policy == observed.policy == Policy.OVERRIDE
            and isinstance(left, PolicyOverride) and isinstance(right, PolicyOverride)
            and left.key and left.key == right.key
            and _valid_scope(left.scope) and _valid_scope(right.scope)
            and _valid_scope(left.reasons) and _valid_scope(right.reasons)
            and type(left.revision) is int and type(right.revision) is int):
        grant = PolicyOverride(scope=tuple(a for a in left.scope if a in right.scope),
                               reasons=tuple(r for r in left.reasons if r in right.reasons),
                               revision=min(left.revision, right.revision), key=left.key)
    policies = (current.policy, observed.policy)
    if Policy.UNKNOWN in policies:
        policy = Policy.UNKNOWN
    elif Policy.EXCLUDED in policies:
        policy = Policy.EXCLUDED
    elif Policy.OVERRIDE in policies:
        policy = Policy.OVERRIDE if grant is not None else Policy.EXCLUDED
    else:
        policy = Policy.EXCLUDED if reasons else Policy.CLEAR
    return replace(current, policy=policy, policy_reasons=reasons, override=grant,
                   required_revision=dropped_revision if grant is None else required_revision,
                   quarantine=current.quarantine or observed.quarantine,
                   merge=(Merge.ALIAS if Merge.ALIAS in (current.merge, observed.merge) else current.merge),
                   alias_key=current.alias_key or observed.alias_key,
                   cause=current.cause or (observed.cause if current.retrieval == observed.retrieval else ""))


def admission(state, *, identity_resolved: bool = False, key: str = "",
              lease_ok: bool = True, budget_ok: bool = True,
              action: str = "acquire", required_revision: int = 0) -> bool:
    """The single, fail-closed pre-dispatch acquisition predicate (v4)."""
    return bool(
        isinstance(state, TerminalState)
        and state.quarantine is False
        and state.retrieval == Retrieval.ELIGIBLE
        and state.merge == Merge.CANONICAL and not state.alias_key
        and identity_resolved is True
        and effective_policy(state, key=key, action=action, required_revision=required_revision)
        and lease_ok is True
        and budget_ok is True
    )


def valid_lease(paper: Paper, *, lease_id: str = "", now: float | None = None) -> bool:
    """Only the executor holding a current campaign capability can use its lease."""
    import time
    from papervault.config import reopen_v2_enabled
    claim = paper.reopen_claim
    if claim is None:
        return True
    return bool(reopen_v2_enabled() and lease_id == claim.lease_id
                and claim.phase in {"queued", "dispatched"}
                and (time.time() if now is None else now) < claim.lease_expiry)


def admitted_budget(paper: Paper, *, library=None) -> bool:
    """Campaign consumption must be readable in the independent durable ledger."""
    from papervault.config import merge_v2_enabled, reopen_v2_enabled
    from .claims_v2 import BudgetLedger, LedgerError, record_generation
    if paper.reopen_claim is None:
        if not reopen_v2_enabled() and not merge_v2_enabled() and paper.merge_budget is None:
            return True
        if library is None or getattr(library, "root", None) is None:
            return False
        try:
            # Index rollback cannot turn a campaign-owned generation into
            # ordinary in-memory work. Initial/no-history normal keys still admit.
            ledger = BudgetLedger(library.root)
            if ledger.merged_into(paper.key) or ledger.latest_generation(paper.key) != record_generation(paper):
                return False
            if paper.merge_budget is not None:
                account = ledger.key_budget(paper.key)
                return account["spent"] < account["budget"]
            return True
        except LedgerError:
            return False
    return bool(reopen_v2_enabled() and library is not None
                and BudgetLedger(library.root).admitted(paper.reopen_claim))
