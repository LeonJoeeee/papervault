"""Durable campaign claims and append-only merge accounting (spec A3).

Record mutations use the existing async library write-lock. Like concurrency.py,
this assumes one library writer process; standalone campaigns refuse a live
service. The ledger's process lock serializes replay/check/append, including reads
from download threads. It is independent of index backups and process memory.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from papervault.config import reopen_v2_enabled

from .state_v2 import Merge, Retrieval, admission, effective_policy, resolved_identity

MAX_CAMPAIGN_CAP = 200
WEEK_SECONDS = 7 * 86400
DEFAULT_LEASE_SECONDS = 3600
_ledger_lock = threading.RLock()


class ReopenClaim(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    key: str = Field(min_length=1)
    generation: int = Field(ge=1)
    lease_id: str = Field(min_length=1)
    lease_expiry: float
    campaign_id: str = Field(min_length=1)
    scope: str = Field(default="reopen", min_length=1)
    phase: Literal["queued", "dispatched", "completed", "not_sent", "released"] = "queued"


class LedgerError(ValueError):
    """Accounting is unavailable or a standing authorization bound refuses."""


class BudgetLedger:
    def __init__(self, root: Path, *, clock=time.time):
        self.path = Path(root) / "reopen-budget.jsonl"
        self.clock = clock

    def _read(self) -> list[dict]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Absence is initial zero only for a vault with no durable history.
            # A removed ledger must never reset budgets or weekly authorization.
            for name in ("index.json", "index.json.bak"):
                try:
                    index = json.loads((self.path.parent / name).read_text(encoding="utf-8"))
                    if any(row.get("reopen_claim") is not None or row.get("merge_budget") is not None
                           for row in index.get("papers", {}).values()):
                        raise LedgerError("budget ledger missing for persisted claims")
                except FileNotFoundError:
                    continue
                except (OSError, UnicodeError, ValueError, AttributeError, TypeError) as exc:
                    raise LedgerError(f"budget ledger missing; cannot establish initial accounting: {exc}") from exc
            return []
        except (OSError, UnicodeError) as exc:
            raise LedgerError(f"unreadable budget ledger: {exc}") from exc
        try:
            if not raw or not raw.endswith("\n"):
                raise ValueError("empty or truncated ledger")
            rows = [json.loads(line) for line in raw.splitlines()]
            campaigns, reservations, phases = {}, {}, {}
            aliases, merged_keys = set(), set()
            history = []
            for row in rows:
                if (not isinstance(row, dict) or type(row.get("version")) is not int or row["version"] != 1
                        or type(row.get("at")) not in (int, float)
                        or not math.isfinite(row["at"])):
                    raise ValueError("unsupported ledger format")
                event = row["event"]
                campaign = row["campaign_id"]
                if not isinstance(campaign, str) or not campaign:
                    raise ValueError("missing campaign")
                if event == "merge":
                    left, right = row["canonical_key"], row["absorbed_key"]
                    if (not isinstance(left, str) or not left or not isinstance(right, str) or not right
                            or left == right or left in aliases or right in aliases
                            or not isinstance(row.get("merge_id"), str) or not row["merge_id"]
                            or campaign != "merge:" + row["merge_id"]
                            or any(r.get("merge_id") == row["merge_id"] for r in history)):
                        raise ValueError("invalid merge identity")
                    before = [_key_budget(history, key) for key in (left, right)]
                    if (row["generations_before"] != [a["generation"] for a in before]
                            or type(row["generation_after"]) is not int
                            or row["generation_after"] != sum(a["generation"] for a in before) + 1
                            or type(row["spent"]) is not int or row["spent"] != sum(a["spent"] for a in before)
                            or type(row["budget"]) is not int or row["budget"] != min(a["budget"] for a in before)
                            or row.get("drain_evidence") != {"claims": "settled", "queue": "drained", "executors": "drained"}
                            or _outstanding(history, (left, right))):
                        raise ValueError("merge without conservative accounting or drain evidence")
                    aliases.add(right)
                    merged_keys.add(left)
                elif event == "campaign_start":
                    if (campaign in campaigns or type(row["cap"]) is not int
                            or not 1 <= row["cap"] <= MAX_CAMPAIGN_CAP
                            or not isinstance(row["keys"], list)
                            or any(not isinstance(key, str) or not key for key in row["keys"])
                            or any(not item["finished"] for item in campaigns.values())):
                        raise ValueError("invalid campaign authorization")
                    if sum(row["at"] - WEEK_SECONDS < item["at"] <= row["at"]
                           for item in campaigns.values()) >= 3:
                        raise ValueError("weekly authorization exceeded")
                    campaigns[campaign] = {**row, "finished": False, "halted": False, "count": 0}
                elif event == "campaign_halt":
                    if (campaign not in campaigns or campaigns[campaign]["halted"]
                            or not isinstance(row.get("reason"), str) or not row["reason"].strip()):
                        raise ValueError("invalid campaign halt")
                    campaigns[campaign]["halted"] = True
                elif event == "campaign_finish":
                    if campaign not in campaigns or campaigns[campaign]["finished"]:
                        raise ValueError("invalid campaign finish")
                    campaigns[campaign]["finished"] = True
                elif event in {"reserve", "dispatch", "refund", "complete", "fence"}:
                    token = ReopenClaim.model_validate(row["claim"])
                    if campaign != token.campaign_id or campaign not in campaigns:
                        raise ValueError("claim without campaign")
                    identifier = token.lease_id
                    if event == "reserve":
                        authorization = campaigns[campaign]
                        account = _key_budget(history, token.key)
                        if (identifier in reservations or authorization["finished"] or authorization["halted"]
                                or token.key not in authorization["keys"]
                                or authorization["count"] >= authorization["cap"]
                                or token.key in aliases or token.generation <= account["generation"]
                                or (token.key in merged_keys and account["spent"] >= account["budget"])
                                or any(r.key == token.key and r.generation >= token.generation
                                       for r in reservations.values())):
                            raise ValueError("invalid reservation")
                        authorization["count"] += 1
                        reservations[identifier] = token
                        phases[identifier] = "reserve"
                    elif event == "fence":
                        if not isinstance(row.get("evidence"), str) or not row["evidence"]:
                            raise ValueError("fence without evidence")
                    else:
                        original = reservations.get(identifier)
                        if original is None or not same_attempt(original, token):
                            raise ValueError("unknown reservation")
                        previous = phases[identifier]
                        allowed = {"dispatch": {"reserve"}, "refund": {"reserve", "dispatch"},
                                   "complete": {"dispatch"}}
                        if previous not in allowed[event]:
                            raise ValueError("invalid reservation transition")
                        if event == "dispatch" and campaigns[campaign]["halted"]:
                            raise ValueError("dispatch after campaign halt")
                        if event == "refund" and row.get("proven_not_sent") is not True:
                            raise ValueError("refund without proof")
                        phases[identifier] = event
                else:
                    raise ValueError("unknown ledger event")
                history.append(row)
            return rows
        except (ValueError, KeyError, TypeError) as exc:
            raise LedgerError(f"unreadable budget ledger: {exc}") from exc

    def _append(self, event: str, campaign_id: str, **fields) -> None:
        row = {"version": 1, "event": event, "at": self.clock(),
               "campaign_id": campaign_id, **fields}
        data = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if os.write(fd, data) != len(data):
                raise OSError("short budget ledger append")
            os.fsync(fd)
        finally:
            os.close(fd)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def check_campaign(self, *, cap: int) -> None:
        if type(cap) is not int or not 1 <= cap <= MAX_CAMPAIGN_CAP:
            raise ValueError(f"cap must be between 1 and {MAX_CAMPAIGN_CAP}")
        with _ledger_lock:
            rows = self._read()
            starts = [row for row in rows if row["event"] == "campaign_start"]
            finishes = {row["campaign_id"] for row in rows if row["event"] == "campaign_finish"}
            if any(row["campaign_id"] not in finishes for row in starts):
                raise LedgerError("another campaign is active; verified drain or human recovery required")
            if sum(row["at"] > self.clock() - WEEK_SECONDS for row in starts) >= 3:
                raise LedgerError("standing cap: at most three campaigns in any seven days")

    def start_campaign(self, *, cap: int, keys: list[str]) -> str:
        with _ledger_lock:
            self.check_campaign(cap=cap)
            campaign = uuid.uuid4().hex
            self._append("campaign_start", campaign, cap=cap, keys=list(dict.fromkeys(keys)))
            return campaign

    def finish_campaign(self, campaign_id: str, *, drained: bool) -> None:
        if drained is not True:
            raise LedgerError("campaign finish requires verified executor drain")
        with _ledger_lock:
            rows = self._read()
            if not any(r["campaign_id"] == campaign_id and r["event"] == "campaign_start" for r in rows):
                raise LedgerError("unknown campaign")
            if not any(r["campaign_id"] == campaign_id and r["event"] == "campaign_finish" for r in rows):
                self._append("campaign_finish", campaign_id, evidence="executor-drained")

    def latest_generation(self, key: str) -> int:
        with _ledger_lock:
            return _key_budget(self._read(), key)["generation"]

    def halt_campaign(self, campaign_id: str, reason: str) -> None:
        """Stop new work durably; this supplies no executor drain or refund proof."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("campaign halt requires a reason")
        with _ledger_lock:
            rows = self._read()
            if not any(r["event"] == "campaign_start" and r["campaign_id"] == campaign_id for r in rows):
                raise LedgerError("unknown campaign")
            if not any(r["event"] == "campaign_halt" and r["campaign_id"] == campaign_id for r in rows):
                self._append("campaign_halt", campaign_id, reason=reason)

    def campaign_halted(self, campaign_id: str) -> bool:
        with _ledger_lock:
            return any(r["event"] == "campaign_halt" and r["campaign_id"] == campaign_id
                       for r in self._read())

    def events(self) -> tuple[dict, ...]:
        """One locked replay for an entire round; callers own the detached rows."""
        with _ledger_lock:
            return tuple(self._read())

    def key_budget(self, key: str) -> dict[str, int]:
        """Campaign caps supply the conservative ceiling; merges retain ancestry."""
        with _ledger_lock:
            return _key_budget(self._read(), key)

    def outstanding(self, keys) -> list[ReopenClaim]:
        with _ledger_lock:
            return _outstanding(self._read(), keys)

    def merged_into(self, key: str) -> str:
        with _ledger_lock:
            return next((r["canonical_key"] for r in self._read()
                         if r["event"] == "merge" and r["absorbed_key"] == key), "")

    def append_merge(self, canonical_key: str, absorbed_key: str, *, merge_id: str) -> dict:
        """Append before index publication; an uncertain index write never replays it."""
        with _ledger_lock:
            rows = self._read()
            if (canonical_key == absorbed_key or self.merged_into(canonical_key)
                    or self.merged_into(absorbed_key) or _outstanding(rows, (canonical_key, absorbed_key))
                    or any(r.get("merge_id") == merge_id for r in rows)):
                raise LedgerError("merge already committed or claims have not drained")
            before = [_key_budget(rows, key) for key in (canonical_key, absorbed_key)]
            fields = dict(merge_id=merge_id, canonical_key=canonical_key, absorbed_key=absorbed_key,
                          generations_before=[a["generation"] for a in before],
                          generation_after=sum(a["generation"] for a in before) + 1,
                          spent=sum(a["spent"] for a in before), budget=min(a["budget"] for a in before),
                          drain_evidence={"claims": "settled", "queue": "drained", "executors": "drained"})
            self._append("merge", "merge:" + merge_id, **fields)
            return fields

    def phase(self, token: ReopenClaim) -> str:
        with _ledger_lock:
            return next((row["event"] for row in reversed(self._read())
                         if "claim" in row and row["claim"]["lease_id"] == token.lease_id), "")

    def reserve(self, token: ReopenClaim) -> None:
        with _ledger_lock:
            rows = self._read()
            if self.phase(token):
                if self.admitted(token):
                    return
                raise LedgerError("reservation already settled")
            start = next((row for row in rows if row["campaign_id"] == token.campaign_id
                          and row["event"] == "campaign_start"), None)
            count = sum(row["campaign_id"] == token.campaign_id and row["event"] == "reserve" for row in rows)
            if (start is None or token.key not in start["keys"] or count >= start["cap"]
                    or token.generation <= self.latest_generation(token.key)
                    or self.merged_into(token.key)
                    or (any(r["event"] == "merge" and r["canonical_key"] == token.key for r in rows)
                        and _key_budget(rows, token.key)["spent"] >= _key_budget(rows, token.key)["budget"])
                    or any(row["campaign_id"] == token.campaign_id and row["event"] in {"campaign_finish", "campaign_halt"}
                           for row in rows)):
                raise LedgerError("paper refused by campaign budget")
            self._append("reserve", token.campaign_id, claim=token.model_dump(),
                         transition={"from": "exhausted", "to": "eligible"})

    def admitted(self, token: ReopenClaim) -> bool:
        try:
            with _ledger_lock:
                phase = self.phase(token)
                return (phase in {"reserve", "dispatch"}
                        and (phase == "dispatch" or not self.campaign_halted(token.campaign_id))
                        and token.generation == self.latest_generation(token.key)
                        and any(same_attempt(ReopenClaim.model_validate(row["claim"]), token)
                                for row in self._read() if row["event"] == "reserve"))
        except LedgerError:
            return False

    def dispatch(self, token: ReopenClaim) -> None:
        with _ledger_lock:
            if not self.admitted(token) or self.phase(token) != "reserve":
                raise LedgerError("reservation cannot dispatch")
            self._append("dispatch", token.campaign_id, claim=token.model_dump())

    def refund_if_not_sent(self, token: ReopenClaim, *, proven_not_sent: bool) -> bool:
        if proven_not_sent is not True:
            return False
        with _ledger_lock:
            # Halt denies admission, not proof-based settlement of queued work.
            try:
                valid = (self.phase(token) in {"reserve", "dispatch"}
                         and token.generation == self.latest_generation(token.key)
                         and any(same_attempt(ReopenClaim.model_validate(r["claim"]), token)
                                 for r in self._read() if r["event"] == "reserve"))
            except LedgerError:
                return False
            if not valid:
                return False
            self._append("refund", token.campaign_id, claim=token.model_dump(), proven_not_sent=True)
            return True

    def complete(self, token: ReopenClaim) -> None:
        with _ledger_lock:
            if self.phase(token) == "complete":
                return
            if self.phase(token) != "dispatch":
                raise LedgerError("completion without dispatch")
            self._append("complete", token.campaign_id, claim=token.model_dump())

    def fence(self, token: ReopenClaim, *, evidence: str) -> None:
        with _ledger_lock:
            self._read()
            self._append("fence", token.campaign_id, claim=token.model_dump(), evidence=evidence)

    def spent(self, campaign_id: str) -> int:
        with _ledger_lock:
            return sum(1 if row["event"] == "reserve" else -1 if row["event"] == "refund" else 0
                       for row in self._read() if row["campaign_id"] == campaign_id)


def same_attempt(left: ReopenClaim | None, right: ReopenClaim) -> bool:
    return bool(left and all(getattr(left, field) == getattr(right, field)
                            for field in ("key", "generation", "lease_id", "campaign_id", "scope")))


def record_generation(paper) -> int:
    if paper is None:
        return 0
    return max(paper.reopen_claim.generation if paper.reopen_claim else 0,
               paper.merge_budget.generation if paper.merge_budget else 0)


def _key_budget(rows, key: str) -> dict[str, int]:
    account = {"spent": 0, "budget": MAX_CAMPAIGN_CAP, "generation": 0}
    for row in rows:
        event = row["event"]
        if event == "campaign_start" and key in row["keys"]:
            account["budget"] = min(account["budget"], row["cap"])
        if "claim" in row and row["claim"]["key"] == key:
            account["generation"] = max(account["generation"], row["claim"]["generation"])
            if event == "reserve":
                account["spent"] += 1
            elif event == "refund":
                account["spent"] -= 1
        if event == "merge" and key in (row["canonical_key"], row["absorbed_key"]):
            account = {"spent": row["spent"], "budget": row["budget"], "generation": row["generation_after"]}
    return account


def _outstanding(rows, keys) -> list[ReopenClaim]:
    pending = {}
    for row in rows:
        claim = row.get("claim")
        if not claim or claim["key"] not in keys:
            continue
        if row["event"] == "reserve":
            pending[claim["lease_id"]] = ReopenClaim.model_validate(claim)
        elif row["event"] in {"refund", "complete"}:
            pending.pop(claim["lease_id"], None)
        elif row["event"] == "fence":
            pending = {lease: token for lease, token in pending.items()
                       if token.key != claim["key"] or token.generation >= claim["generation"]}
    return list(pending.values())


def reopenable(paper) -> bool:
    state = paper.acquisition_state
    return (paper.merge_intent is None and state.retrieval == Retrieval.EXHAUSTED and not state.quarantine
            and state.merge == Merge.CANONICAL and not state.alias_key
            and effective_policy(state, key=paper.key) and resolved_identity(paper)
            and (paper.reopen_claim is None or paper.reopen_claim.phase in {"completed", "not_sent", "released"}))


class ClaimStore:
    def __init__(self, library, *, ledger: BudgetLedger | None = None, clock=time.time):
        from .services import concurrency
        self.lock = concurrency.lib_write_lock
        self.library, self.clock = library, clock
        self.ledger = ledger or BudgetLedger(library.root, clock=clock)

    def save(self) -> None:
        self.library.save(force=True, durable=True)

    async def claim(self, key: str, *, expected_state, expected_generation: int,
                    campaign_id: str, ttl: float = DEFAULT_LEASE_SECONDS) -> ReopenClaim | None:
        if not reopen_v2_enabled():
            raise ValueError("PAPERVAULT_REOPEN_V2 is disabled")
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("lease TTL must be positive and finite")
        async with self.lock:
            paper = self.library.get(key)
            generation = record_generation(paper)
            if (paper is None or paper.acquisition_state != expected_state
                    or generation != expected_generation or not reopenable(paper)
                    or generation != self.ledger.latest_generation(key)):
                return None
            token = ReopenClaim(key=key, generation=generation + 1, lease_id=uuid.uuid4().hex,
                                lease_expiry=float(self.clock() + ttl), campaign_id=campaign_id)
            self.ledger.reserve(token)  # Never undone if the following durable save is uncertain.
            paper.reopen_claim = token
            paper.terminal_state = replace(paper.acquisition_state, retrieval=Retrieval.ELIGIBLE, cause="")
            paper.download_status = "pending"
            self.save()
            return token

    def current(self, token: ReopenClaim):
        paper = self.library.get(token.key)
        return paper if paper and same_attempt(paper.reopen_claim, token) else None

    def allowed(self, token: ReopenClaim, *, executing: bool = False) -> bool:
        paper = self.current(token)
        if not reopen_v2_enabled() or paper is None or self.clock() >= paper.reopen_claim.lease_expiry:
            return False
        expected = Retrieval.IN_PROGRESS if executing else Retrieval.ELIGIBLE
        state = paper.acquisition_state
        if state.retrieval != expected:
            return False
        return admission(replace(state, retrieval=Retrieval.ELIGIBLE), key=paper.key,
                         identity_resolved=resolved_identity(paper), lease_ok=True,
                         budget_ok=self.ledger.admitted(token))

    def _quarantine(self, paper, reason: str) -> None:
        paper.terminal_state = replace(paper.acquisition_state, quarantine=True)
        self.save()
        self.library.log({"event": "reopen_quarantine", "key": paper.key, "reason": reason,
                          "generation": record_generation(paper)})

    async def quarantine(self, token: ReopenClaim, *, reason: str) -> None:
        async with self.lock:
            paper = self.current(token)
            if paper is not None:
                self._quarantine(paper, reason)

    async def expire(self, key: str, *, token: ReopenClaim | None = None) -> bool:
        async with self.lock:
            paper = self.library.get(key)
            if (paper is None or paper.reopen_claim is None
                    or (token is not None and not same_attempt(paper.reopen_claim, token))
                    or paper.reopen_claim.phase not in {"queued", "dispatched"}
                    or self.clock() < paper.reopen_claim.lease_expiry):
                return False
            self._quarantine(paper, "lease expired without drain evidence")
            return True

    async def pre_dispatch(self, token: ReopenClaim) -> bool:
        from .telemetry_v2 import admission_denied
        async with self.lock:
            paper = self.current(token)
            if paper is None:
                admission_denied(self.library, token, "stale_attempt")
                return False
            phase = self.ledger.phase(token)  # Corruption raises; no request/refund.
            if paper.reopen_claim.phase not in {"queued", "dispatched"}:
                admission_denied(self.library, token, "settled_attempt")
                return False
            if paper.reopen_claim.phase == "dispatched" or phase == "dispatch":
                self._quarantine(paper, "dispatch outcome uncertain; refusing replay")
                admission_denied(self.library, token, "uncertain_dispatch")
                return False
            if self.clock() >= paper.reopen_claim.lease_expiry:
                self._quarantine(paper, "lease expired before dispatch")
                admission_denied(self.library, token, "expired_lease")
                return False
            if not self.allowed(token):
                admission_denied(self.library, token, "acquisition_guard")
                self._not_sent(paper, token)
                return False
            self.ledger.dispatch(token)
            paper.reopen_claim = token.model_copy(update={"phase": "dispatched"})
            paper.terminal_state = replace(paper.acquisition_state, retrieval=Retrieval.IN_PROGRESS)
            self.save()
            return True

    def _not_sent(self, paper, token: ReopenClaim) -> bool:
        if not self.ledger.refund_if_not_sent(token, proven_not_sent=True):
            return False
        paper.reopen_claim = paper.reopen_claim.model_copy(update={"phase": "not_sent"})
        if (paper.download_status == "pending"
                and paper.acquisition_state.retrieval in {Retrieval.ELIGIBLE, Retrieval.IN_PROGRESS}):
            paper.download_status = "failed"
        self.save()
        return True

    async def not_sent(self, token: ReopenClaim, *, executor_proof: bool = False) -> bool:
        """Queued refusals are proven; dispatched attempts need executor evidence."""
        async with self.lock:
            paper = self.current(token)
            if paper is None:
                return False
            if not executor_proof and (paper.reopen_claim.phase != "queued" or self.ledger.phase(token) != "reserve"):
                return False
            return self._not_sent(paper, token) if paper else False

    async def release_quarantine(self, key: str, *, drained_lease_id: str = "",
                                 human_reason: str = "") -> bool:
        async with self.lock:
            paper = self.library.get(key)
            if (paper is None or paper.reopen_claim is None or not paper.acquisition_state.quarantine
                    or not (human_reason.strip() or drained_lease_id == paper.reopen_claim.lease_id)):
                return False
            token = paper.reopen_claim.model_copy(update={"generation": paper.reopen_claim.generation + 1,
                                                         "phase": "released"})
            self.ledger.fence(token, evidence=human_reason.strip() or f"drained:{drained_lease_id}")
            paper.reopen_claim = token
            if (paper.download_status == "pending"
                    and paper.acquisition_state.retrieval in {Retrieval.ELIGIBLE, Retrieval.IN_PROGRESS}):
                paper.download_status = "failed"
            paper.terminal_state = replace(paper.acquisition_state, quarantine=False)
            self.save()
            return True
