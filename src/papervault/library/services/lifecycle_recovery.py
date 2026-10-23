"""Audited operator recovery through existing claim and campaign guards."""

from __future__ import annotations

from ..claims_v2 import BudgetLedger, ClaimStore, LedgerError, same_attempt
from ..state_v2 import Merge
from .operator_actions import stage_operator


async def operate(library, *, action, key=None, campaign_id=None, reason, actor,
                  executor_proof=False, drained=False, lease_id="") -> dict:
    """Record intent without turning a reason or a refused action into proof.

    A queued attempt can prove itself unsent. Dispatched attempts still require
    explicit executor evidence; human quarantine disposition fences without a
    refund. A drain assertion never overrides outstanding ledger or record work.
    """
    if action not in {"claim-recover", "release-quarantine", "campaign-halt"}:
        raise ValueError("unknown lifecycle recovery action")
    if any(not isinstance(value, str) or not value.strip() for value in (reason, actor)):
        raise ValueError("operator recovery requires nonempty actor and reason")
    if type(executor_proof) is not bool or type(drained) is not bool:
        raise ValueError("executor proof and drain evidence must be explicit booleans")
    if not isinstance(lease_id, str):
        raise ValueError("lease identity must be a string")

    ledger = BudgetLedger(library.root)
    store = ClaimStore(library, ledger=ledger)
    token = None
    if action == "campaign-halt":
        start = next((row for row in ledger.events()
                      if row["event"] == "campaign_start" and row["campaign_id"] == campaign_id), None)
        if start is None:
            raise ValueError("existing campaign required")
        keys = start["keys"]
        if key is not None and key not in keys:
            raise ValueError("record key is outside the campaign")
    else:
        paper = library.get(key) if key else None
        if paper is None:
            raise ValueError("existing record key required")
        token = store.recovery_claim(key)
        if token is None:
            state = paper.acquisition_state
            if (action != "release-quarantine" or lease_id or campaign_id is not None
                    or not state.quarantine or state.merge != Merge.CANONICAL
                    or state.alias_key or paper.merge_intent is not None):
                raise ValueError("record has no recovery claim")
        else:
            if lease_id and lease_id != token.lease_id:
                raise ValueError("claim lease changed")
            if campaign_id is not None and campaign_id != token.campaign_id:
                raise ValueError("claim campaign changed")
        keys = [key]

    stage_operator(library, action=action, keys=keys, reason=reason, actor=actor,
                   event=action.replace("-", "_"), detail={"campaign_id": token.campaign_id if token else campaign_id,
                       "lease_id": lease_id or (token.lease_id if token else ""),
                       "executor_proof": executor_proof, "drained": drained})
    try:
        if action == "campaign-halt":
            already_halted = ledger.campaign_halted(campaign_id)
            ledger.halt_campaign(campaign_id, reason)
            if drained:
                # Reconcile only crash gaps here. Existing queued/dispatched
                # record work still requires its explicit recovery disposition.
                # Include refunded orphan reservations too: a prior interrupted
                # save may have settled accounting without recording generation.
                for selected_key in keys:
                    claim = store.recovery_claim(selected_key)
                    if claim is None or claim.campaign_id != campaign_id:
                        continue
                    paper = library.get(claim.key)
                    if (paper is not None and (not same_attempt(paper.reopen_claim, claim)
                            or paper.reopen_claim.phase == "completed")):
                        await store.recover(claim)
                if any(token.campaign_id == campaign_id for token in ledger.outstanding(keys)):
                    raise LedgerError("campaign has outstanding reservations")
                if any(paper.reopen_claim is not None
                       and paper.reopen_claim.campaign_id == campaign_id
                       and paper.reopen_claim.phase in {"queued", "dispatched"}
                       for paper in (library.get(k) for k in library.keys())):
                    raise LedgerError("campaign has unsettled record claims")
                ledger.finish_campaign(campaign_id, drained=True)
            return {"action": action, "campaign_id": campaign_id,
                    "changed": not already_halted or drained, "finished": drained}
        if action == "claim-recover":
            changed = await store.recover(token, executor_proof=executor_proof)
        else:
            changed = await store.release_quarantine(key, human_reason=reason,
                                                    expected_lease_id=token.lease_id if token else "")
        return {"action": action, "key": key, "changed": changed}
    finally:
        # Ledger-only mutations and safe refusal intents must commit their
        # staged PostgreSQL ruling too; an unchanged record is not a no-op.
        library.save(force=True, durable=True)
