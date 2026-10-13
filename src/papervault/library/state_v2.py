"""Pure acquisition state contracts (accepted v2 spec, Appendix A1/A3).

These dimensions do not replace the extraction axis or the public statuses.
Legacy state is a read-time projection, never a migration. Lease and budget
providers are explicit slice-3 extension points; their current defaults admit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Paper


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


def derive_legacy(paper) -> TerminalState:
    """Conservatively project the five live statuses, without mutating a row.

    Every cause-unrecorded terminal, including acquired/extract-failed, has
    unknown policy. Extraction failure never means retrieval failure.
    """
    status = _field(paper, "download_status")
    if status == "pending":
        return TerminalState(retrieval=Retrieval.ELIGIBLE, policy=Policy.CLEAR)
    if status == "ok":
        return TerminalState(retrieval=(Retrieval.VERIFIED if extraction_evidence(paper)
                                       else Retrieval.ACQUIRED))
    if status == "extract_failed":
        return TerminalState(retrieval=Retrieval.ACQUIRED)
    if status in ("metadata_only", "failed"):
        return TerminalState(retrieval=Retrieval.EXHAUSTED)
    return TerminalState()


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
    if (not _valid_scope(current.policy_reasons) or not _valid_scope(observed.policy_reasons)
            or type(current.required_revision) is not int or type(observed.required_revision) is not int
            or current.policy not in tuple(Policy) or observed.policy not in tuple(Policy)):
        return replace(current, policy=Policy.UNKNOWN, override=None)
    reasons = tuple(dict.fromkeys((*current.policy_reasons, *observed.policy_reasons)))
    grant = None
    left, right = current.override, observed.override
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
                   required_revision=max(current.required_revision, observed.required_revision),
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


def valid_lease(paper: Paper) -> bool:
    """Slice-3 placeholder provider: no lease machinery exists in slice 1."""
    return True


def admitted_budget(paper: Paper) -> bool:
    """Slice-3 placeholder provider: no budget ledger exists in slice 1."""
    return True
