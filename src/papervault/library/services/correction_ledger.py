"""Library-owned conflict evidence and bounded, orchestrator-owned dispatch packets."""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

BATCH_PAPERS = 10
MAX_BATCH_CONFLICTS = 50
CONFLICT_FIELDS = (
    "title", "authors", "year", "venue", "volume", "pages", "issue", "abstract",
    "doi", "arxiv_id", "paper_id", "source", "url", "inspire_record_id",
)
_STAGED = ContextVar("correction_staged", default=None)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CorrectionLedger:
    """Atomic state beside index.json; construction and read never create files."""

    def __init__(self, root: Path):
        self.path = Path(root) / "correction-ledger.json"

    def read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "conflicts": {}, "manifests": {}, "adjudications": {},
                    "routes": {}, "months": {}, "schedule": {}}
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if (state.get("version") != 1 or any(not isinstance(state.get(k), dict) for k in
                    ("conflicts", "manifests", "adjudications", "routes", "months", "schedule"))):
                raise ValueError("invalid correction ledger schema")
            return state
        except (AttributeError, json.JSONDecodeError) as exc:
            raise ValueError(f"unreadable correction ledger at {self.path}") from exc

    @contextmanager
    def transaction(self):
        # One lock across instances/processes; manifest assignment and emission
        # share the evidence write, so a restart cannot dispatch a batch twice.
        from ..store import _atomic_write
        with FileLock(str(self.path) + ".lock"):
            state = self.read()
            before = copy.deepcopy(state)
            yield state
            if state != before:
                _atomic_write(self.path, json.dumps(state, ensure_ascii=False, indent=2))

    def stage(self, *, key: str, fields: dict, existing: dict, incoming: dict,
              provenance: dict) -> str:
        """Keep original sides; fingerprint evidence, not mutable routing/timestamps."""
        identity = {"key": key, "fields": fields, "operation": provenance.get("operation"),
                    "sides": [{k: side.get(k) for k in
                               ("key", "doi", "arxiv_id", "source", "paper_id", "url")}
                              for side in (existing, incoming)]}
        recovery = provenance.get("recovery")
        if recovery:
            identity["recovery"] = {k: v for k, v in recovery.items() if k != "observed_at"}
        identity["source_provenance"] = {k: provenance.get(k) for k in
                                         ("incoming_source", "verification", "source_url", "lookup_doi")}
        if "identity_owner" in fields:
            identity["bibliography"] = [{k: side.get(k) for k in CONFLICT_FIELDS}
                                       for side in (existing, incoming)]
        identifier = _digest(identity)
        created = False
        with self.transaction() as state:
            if identifier not in state["conflicts"]:
                created = True
                state["conflicts"][identifier] = {
                    "id": identifier, "key": key, "fields": copy.deepcopy(fields),
                    "existing": copy.deepcopy(existing), "incoming": copy.deepcopy(incoming),
                    "provenance": copy.deepcopy(provenance), "observed_at": utc_now(),
                    "status": "pending", "manifest_id": None,
                }
                self._emit(state)
        captured = _STAGED.get()
        if created and captured is not None and captured[0] == self.path:
            captured[1].add(identifier)
        return identifier

    @contextmanager
    def capture_staged(self):
        """Attribute only this application stack, excluding download threads."""
        identifiers = set()
        token = _STAGED.set((self.path, identifiers))
        try:
            yield identifiers
        finally:
            _STAGED.reset(token)

    @staticmethod
    def _emit(state: dict) -> None:
        pending = [c for c in state["conflicts"].values() if c["status"] == "pending"]
        keys = list(dict.fromkeys(c["key"] for c in pending))
        if len(keys) < BATCH_PAPERS:
            return
        keys = keys[:BATCH_PAPERS]
        # Include one conflict for each paper before filling remaining slots.
        selected = [next(c for c in pending if c["key"] == key) for key in keys]
        ids = {c["id"] for c in selected}
        selected += [c for c in pending if c["key"] in keys and c["id"] not in ids][
            :MAX_BATCH_CONFLICTS - len(selected)]
        identifier = _digest({"conflicts": [c["id"] for c in selected]})
        state["manifests"][identifier] = {
            "id": identifier, "created_at": utc_now(), "dispatch_owner": "orchestrator",
            "executor": {"carrier": "codex-cli", "model": "gpt-6-luna", "effort": "max"},
            "paper_keys": keys, "conflicts": copy.deepcopy(selected),
            "default_disposition": "needs_human",
            "instructions": (
                "Investigate both sides and provenance. DOI is the primary anchor; the non-DOI "
                "identity ladder is TBD. Return manifest_id and results with conflict_id, status "
                "(resolved or needs_human), decision and evidence. Unadjudicable items remain "
                "needs_human. Do not overwrite fields, merge records, acquire, or reset terminal "
                "states. Field changes require a separately authorized correction."
            ),
        }
        for conflict in selected:
            conflict.update(status="batched", manifest_id=identifier)

    def adjudicate(self, packet: dict) -> None:
        """Record investigation only; missing answers become retained needs_human."""
        if not isinstance(packet, dict) or not isinstance(packet.get("results"), list):
            raise ValueError("adjudication requires manifest_id and results")
        with self.transaction() as state:
            identifier = packet.get("manifest_id")
            manifest = state["manifests"].get(identifier)
            if not manifest:
                raise ValueError("unknown manifest_id")
            allowed = {c["id"] for c in manifest["conflicts"]}
            answers = {}
            for result in packet["results"]:
                if not isinstance(result, dict):
                    raise ValueError("invalid result")
                cid = result.get("conflict_id")
                status = result.get("status", "needs_human")
                if cid not in allowed or cid in answers or status not in {"resolved", "needs_human"}:
                    raise ValueError("foreign, duplicate, or invalid disposition")
                if status == "resolved" and (not isinstance(result.get("evidence"), str)
                        or not result["evidence"].strip() or not isinstance(result.get("decision"), str)
                        or not result["decision"].strip()):
                    raise ValueError("resolved requires decision and evidence")
                answers[cid] = {"status": status, "decision": result.get("decision", "unadjudicated"),
                                "evidence": result.get("evidence", "")}
            normalized = {cid: answers.get(cid, {"status": "needs_human", "decision": "unadjudicated",
                                                "evidence": ""}) for cid in sorted(allowed)}
            previous = state["adjudications"].get(identifier)
            if previous is not None:
                if previous != normalized:
                    raise ValueError("manifest already adjudicated with different evidence")
                return
            state["adjudications"][identifier] = normalized
            for cid, answer in normalized.items():
                state["conflicts"][cid].update(answer)

    def summary(self) -> dict:
        state = self.read()
        return {status: sum(c["status"] == status for c in state["conflicts"].values())
                for status in ("pending", "batched", "resolved", "needs_human")}

    def stage_fields(self, existing, incoming: dict, *, operation="merge", provenance: dict | None = None,
                     proposals: dict | None = None) -> None:
        fields = {}
        for field in CONFLICT_FIELDS:
            old = getattr(existing, field)
            new = (proposals or {}).get(field, incoming.get(field))
            if old in (None, "", [], 0) or new in (None, "", [], 0) or old == new:
                continue
            if field == "doi" and str(old).lower() == str(new).lower():
                continue
            if field == "arxiv_id":
                from ..store import _normalize_arxiv
                if _normalize_arxiv(old) == _normalize_arxiv(new):
                    continue
            fields[field] = {"existing": old, "incoming": new}
        if fields:
            self.stage(key=existing.key, fields=fields, existing=existing.model_dump(),
                       incoming=incoming, provenance={"operation": operation,
                       "existing_source": existing.source,
                       "incoming_source": incoming.get("source") or "unknown", **(provenance or {})})

    def stage_identity(self, paper, incoming: dict, holder, *, provenance: dict) -> None:
        side = holder or paper
        fields = {"identity_owner": {"existing": side.key, "incoming": paper.key}}
        for name in ("doi", "arxiv_id"):
            if incoming.get(name):
                fields[name] = {"existing": getattr(side, name), "incoming": incoming[name]}
        self.stage(key=paper.key, fields=fields, existing=side.model_dump(),
                   incoming=paper.model_dump(), provenance=provenance)
