"""S3/S4 real-machine integration probe (SDD §6.5 / §6.6, blocker ②).

Runs the FULL run_round() orchestration chain — reconcile_terminal → diff → REMOVE →
REDISTILL_DELETE → DISTILL_BATCH → tail process — against REAL Neo4j + Postgres + MiMo on
the ISOLATED l0_probe workspace, over a FIXED SMALL key subset (run_round's only_keys
entry, blocker ③). It is the first time run_round is exercised outside offline FakeRag.

Why it exists: run_round/distill/reconcile_terminal were only ever covered by tests/
test_round.py with an in-memory FakeRag. The iter8 bug (aget_docs_by_ids returns a plain
dict at RUNTIME, not a DocProcessingStatus object — so getattr(st,'status') was always
None, mis-flipping every PROCESSED doc to error via stuck_guard) was caught by source
reading + a mock that *encodes the shape I assumed*. A mock can only reproduce the shape I
think it is; only a real LightRAG can falsify it. So this probe also PRINTS the raw
aget_docs_by_ids return (PROBE marker) to pin the actual runtime shape in the log as real-
machine backing for the §6.6 ★ contract (finding: real-machine dict contract had zero
regression protection).

  NEO4J_WORKSPACE=l0_probe POSTGRES_WORKSPACE=l0_probe \
    uv run python scripts/probe_round_s4.py

MANUAL REAL-MACHINE GATE (when you MUST re-run — pin the process, not memory):
  This probe lives in scripts/ and is NOT collected by pytest (it needs Neo4j + PG +
  MiMo; forcing it into default CI would violate the lightweight-CI / single-process-on-
  l0_probe rules). The dict-shape regression class (iter8) IS netted in CI by
  tests/test_doc_status_shape.py, but run_round's full real-machine CLOSURE (real adelete
  clearing, REDISTILL delete-then-insert, synth/cited closure) has NO automatic trigger.
  Therefore: after ANY change to run_round / distill / reconcile_terminal REAL-MACHINE
  SEMANTICS (not just dict shape), you MUST re-run this probe on l0_probe and confirm the
  final line reads `PROBE OK | OVERALL` with ZERO `PROBE FAIL`. A green 55-passed pytest does
  NOT cover this; the real-machine guardrail rides on running this probe, not on memory.

Asserts (each prints a PROBE marker):
  (a) the subset keys reach ledger.status == 'done' (processing → done via the second-
      round reconcile_terminal — enqueue only returns a track_id, §6.6).
  (b) doc_status terminal = PROCESSED for those doc_ids (real aget_docs_by_ids dict,
      _doc_status parses it correctly — the iter8 contract, verified on-machine).
  (c) query(NL intent) returns a REAL synthesized answer (non-empty AND not the synth
      failure-fallback sentinel); cited_papers ⊆ this batch's ingested keys (discriminative,
      not 永真); kb_coverage does not crash (∈ {empty, thin, strong}).
  (e) one REAL REDISTILL round on-machine: mutate the victim's ledger fingerprint → diff
      classifies it to_redistill → REDISTILL_DELETE old doc → re-insert → back to 'done' +
      doc_status PROCESSED + ZERO dup-<hash> FAILED rows (the §9 F2 delete-then-insert
      invariant — previously FakeRag-only, which cannot falsify real adelete clearing).
  (d) one REMOVE closes the loop: remove_one → ledger row gone + graph doc deleted.

Real-machine coverage note: (a)/(b) cover to_distill→done; (e) covers REDISTILL_DELETE +
re-insert. The error (F9) and pending_remove (F3) branches remain FakeRag-only (real-machine
zero-coverage) — they need an injected FAILED/403, out of this probe's small-subset scope.

PROD-SAFETY (铁律): guarded by assert_safe_workspace() (refuses prod 'l0' without opt-in);
confined to a fixed 3-paper subset via only_keys (never the full ~4k vault, rule (4));
all l0_probe rows it touches are cleaned up at the end (and pre-cleaned for idempotent
re-runs). NEVER touches prod 'l0'.
"""
import asyncio
import traceback

SUBSET_SIZE = 3
ROUND_LIMIT = 12          # safety cap on poll rounds (enqueue→process→reconcile is async)
INTENT = "What methods and phenomena are discussed across the ingested papers?"


def _mark(ok: bool, label: str, detail: str = "") -> None:
    print(f"PROBE {'OK ' if ok else 'FAIL'} | {label}" + (f" | {detail}" if detail else ""), flush=True)


async def _failed_dup_ids(workspace: str) -> list[str]:
    """doc_status ids that are FAILED and dup-<hash> (the §6.1 F2 pollution signature).

    REDISTILL must delete-then-insert; if it re-inserts over a still-present doc_id,
    LightRAG writes a `dup-<hash>` FAILED row (lightrag.py:1456) and starves the LLM pool.
    A clean REDISTILL leaves ZERO such rows. Reads PG directly (no LightRAG stack), filtered
    by workspace — same pattern as cli._count_doc_status.
    """
    import psycopg

    from papervault.knowledge.config import CONFIG

    async with await psycopg.AsyncConnection.connect(CONFIG.postgres.dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id FROM lightrag_doc_status "
                "WHERE workspace=%s AND status='failed' AND id LIKE 'dup-%%'",
                (workspace,),
            )
            rows = await cur.fetchall()
    return [r[0] for r in rows]


async def _drive_until_done(rag, keys, ledger, run_round, *, label):
    """Run run_round(only_keys=keys) repeatedly until every key is terminal or cap hit.

    enqueue only returns a track_id; reconcile_terminal closes 'processing' → 'done' on a
    LATER round once LightRAG has actually processed the doc. So one run_round is not
    enough — we poll up to ROUND_LIMIT rounds, printing each round's summary + ledger
    states so the async settle is visible in the log.
    """
    last = {}
    for n in range(1, ROUND_LIMIT + 1):
        summary = await run_round(rag, only_keys=keys)
        led = await ledger.load("paper")
        states = {k: (led[k].status if k in led else "absent") for k in keys}
        last = states
        print(f"PROBE .. | {label} round {n} | {states} | {summary}", flush=True)
        # terminal = done / done_meta / error (not processing, not absent-after-distill)
        if all(states[k] in ("done", "done_meta", "error") for k in keys):
            return states, summary
        await asyncio.sleep(3)
    return last, None


async def main() -> None:
    from papervault.knowledge.store.graph import UnsafeWorkspaceError, assert_safe_workspace

    try:
        ws = assert_safe_workspace()
        _mark(True, "workspace-guard", f"workspace={ws}")
    except UnsafeWorkspaceError as e:
        _mark(False, "workspace-guard", str(e))
        return

    from lightrag.base import DocStatus

    from papervault.knowledge.ingest.distill import doc_id, remove_one
    from papervault.knowledge.ingest.vault import load_clean_index, read_extract_raw
    from papervault.knowledge.ledger import store as ledger
    from papervault.knowledge.query.aquery import query
    from papervault.knowledge.query.synth import SYNTH_FAILED_PREFIX
    from papervault.knowledge.scheduler.round import _doc_status, run_round
    from papervault.knowledge.store.graph import get_graph

    # pick a fixed small subset of papers that actually have DISTILLABLE full text.
    # ★ The predicate MUST match distill's own text gate, not merely "a path field is
    # declared". distill reads via read_extract_raw (vault.py), which returns text only when
    # `full.exists()`; a declared-but-missing/empty file → None → distill_batch routes the
    # key to done_meta (no_text), never real 'done'. If we picked on `r.md_path or r.txt_path`
    # (path DECLARED) and the 3 chosen files happened to be missing/empty, (a) would be all
    # done_meta, done_keys empty, and the probe would hit the subset-unusable early return,
    # silently skipping (b)/(c)/(e)/(d). Filtering on read_extract_raw being non-empty
    # guarantees the SUBSET_SIZE picks have distillable text, eliminating that spurious skip.
    idx = load_clean_index()
    keys = [k for k, r in idx.items() if (read_extract_raw(r) or "").strip()][:SUBSET_SIZE]
    if len(keys) < SUBSET_SIZE:
        _mark(False, "pick-papers", f"only {len(keys)} papers with distillable text available")
        return
    _mark(True, "pick-papers", f"keys={keys}")

    try:
        rag = await get_graph()
        _mark(True, "get_graph")
    except Exception as e:  # noqa: BLE001
        _mark(False, "get_graph", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return

    await ledger.ensure_schema()

    # ---- pre-clean: idempotent re-run. Remove any leftover graph docs + ledger rows for
    # these keys from a prior probe run, so we start from a known-empty state for the subset.
    for k in keys:
        try:
            await remove_one(rag, k, delete_ledger=True)
        except Exception as e:  # noqa: BLE001 — pre-clean is best-effort
            print(f"PROBE .. | pre-clean {k} skipped: {type(e).__name__}: {e}", flush=True)
    await rag.apipeline_process_enqueue_documents()

    ok_all = True
    try:
        # ---- (a) distill the subset, poll until terminal ----
        states, _ = await _drive_until_done(rag, keys, ledger, run_round, label="distill")
        done_keys = [k for k in keys if states.get(k) == "done"]
        # done_meta is legitimate (no-text/content-dup); for the probe we need at least one
        # real 'done' so doc_status + query have graph content to verify.
        a_ok = len(done_keys) >= 1 and all(states.get(k) in ("done", "done_meta") for k in keys)
        _mark(a_ok, "(a) ledger processing->done", f"states={states} done={done_keys}")
        ok_all &= a_ok

        # If NO key reached real 'done' (all done_meta/error — e.g. files went missing/empty,
        # or the subset is internally content-dup'd), (b)/(c)/(d) have no graph content to
        # verify and would cascade into misleading FAILs. That's a "pick other keys" signal,
        # not a broken chain — separate it explicitly and stop here (subset is unusable).
        if not done_keys:
            _mark(False, "subset-unusable",
                  f"no key reached real 'done' (states={states}); "
                  "pick a subset with real full text — chain is NOT proven broken")
            ok_all = False
            return  # finally: still cleans up the subset

        # ---- (b) doc_status terminal = PROCESSED (REAL aget_docs_by_ids dict contract) ----
        doc_ids = [doc_id(k) for k in done_keys]
        raw = await rag.aget_docs_by_ids(doc_ids)
        # PIN THE REAL RUNTIME SHAPE (SDD §6.6 ★): proves the dict-vs-object contract the
        # iter8 fix relies on, against a real LightRAG — not a mock that assumes it.
        sample = raw.get(doc_ids[0]) if doc_ids else None
        print(f"PROBE .. | (b) RAW aget_docs_by_ids[{doc_ids[0] if doc_ids else None}] "
              f"type={type(sample).__name__} value={sample!r}", flush=True)
        parsed = {d: _doc_status(raw.get(d)) for d in doc_ids}
        b_ok = bool(doc_ids) and all(parsed[d] == DocStatus.PROCESSED for d in doc_ids)
        _mark(b_ok, "(b) doc_status=PROCESSED", f"parsed={parsed}")
        # explicit getattr counter-example: the OLD buggy access returns None on the dict
        if isinstance(sample, dict):
            _mark(getattr(sample, "status", None) is None,
                  "(b') getattr-on-dict is None (iter8 contract)",
                  "confirms dict-subscript is required")
        ok_all &= b_ok

        # ---- (c) query(intent): non-empty answer, cited_papers carry keys, no crash ----
        try:
            res = await query(INTENT, rag=rag)
            answer = res.get("answer", "")
            cited = res.get("cited_papers", [])
            cov = res.get("kb_coverage")
            # A non-empty answer is NOT enough: synth_answer returns SYNTH_FAILED_PREFIX (a
            # non-empty honesty fallback) instead of raising when the MiMo call dies. Exclude
            # that sentinel so "synth LLM integrally failed while retrieval succeeded" FAILs
            # here rather than green-passing — (c) must prove real synthesized prose.
            synth_failed = isinstance(answer, str) and answer.startswith(SYNTH_FAILED_PREFIX)
            c_ok = (
                isinstance(answer, str) and len(answer.strip()) > 0 and not synth_failed
                and cov in ("empty", "thin", "strong")
            )
            _mark(c_ok, "(c) query non-empty (real synth) + kb_coverage ok",
                  f"answer_len={len(answer)} synth_failed={synth_failed} "
                  f"kb_coverage={cov} cited={cited}")
            ok_all &= c_ok

            # (c') cited_papers must carry ONLY keys from this batch's ingested papers — a
            # discriminative, GATED check (the old `... or bool(cited)` was 永真: any non-empty
            # cited passed even if it held unknown_source / a wrong key / a mis-stripped prefix,
            # the F12 regression class).
            #   NEGATIVE direction (no wrong/foreign key): cited ⊆ done_keys.
            #   POSITIVE direction (the link actually extracted THIS batch's keys): without it,
            #   cited==[] passes vacuously (∅ ⊆ anything) and a prefix-parse regression in
            #   _cited_papers (paper/ vs paper:) or a references[] shape change would green-pass
            #   on EMPTY citations — exactly the "shape assumption ≠ real-machine falsified"
            #   hazard blocker ② exists to close. So in the broad-INTENT / kb_coverage=='strong'
            #   regime (38 chunks across the freshly-ingested papers), zero citations is itself a
            #   defect: when c_ok holds AND cov=='strong', additionally require len(cited)>=1 to
            #   pin the positive extraction link. (Outside that regime — thin/empty coverage — we
            #   still tolerate "none hit" but emit a non-fatal WARN so a vacuous pass isn't
            #   misread as "citation link verified".)
            no_foreign_key = set(cited).issubset(set(done_keys))
            if c_ok and cov == "strong":
                cprime_ok = no_foreign_key and len(cited) >= 1
                _mark(cprime_ok,
                      "(c') cited_papers carry >=1 of this batch's keys, none foreign (strong cov)",
                      f"cited={cited} done={done_keys}")
            else:
                cprime_ok = no_foreign_key
                _mark(cprime_ok, "(c') cited_papers carry only this batch's paper keys",
                      f"cited={cited} done={done_keys} cov={cov}")
                if not cited:
                    _mark(True, "(c') WARN: cited empty — positive citation link NOT verified "
                          "this run (cov!=strong, tolerated)", f"cov={cov}")
            ok_all &= cprime_ok
        except Exception as e:  # noqa: BLE001
            _mark(False, "(c) query crashed", f"{type(e).__name__}: {e}")
            traceback.print_exc()
            ok_all = False

        # ---- (e) one REAL REDISTILL round on-machine (F2 delete-then-insert invariant) ----
        # The to_distill path is now real-machine-backed by (a)/(b); REDISTILL_DELETE was only
        # ever exercised by FakeRag (tests/test_round.py), which CANNOT falsify the §9 invariant
        # that adelete truly clears the old doc so re-insert is clean (no dup-<hash> FAILED row).
        # Force a redistill by mutating the victim's ledger fingerprint (diff sees fp≠led.fp →
        # to_redistill → REDISTILL_DELETE old doc → DISTILL_BATCH re-insert), then poll back to
        # 'done' and assert: doc_status PROCESSED + ZERO dup-<hash> FAILED rows.
        redistill_victim = done_keys[0]
        try:
            led_before = await ledger.load("paper")
            rec0 = led_before[redistill_victim]
            # overwrite fingerprint to a sentinel ≠ the real content hash → next diff classifies
            # it to_redistill (content-changed branch). doc_id/status preserved.
            await ledger.upsert("paper", redistill_victim, doc_id=rec0.doc_id,
                                status="done", fingerprint="REDISTILL-PROBE-SENTINEL")
            dup_before = await _failed_dup_ids(ws)
            states_e, _ = await _drive_until_done(
                rag, [redistill_victim], ledger, run_round, label="redistill")
            dup_after = await _failed_dup_ids(ws)
            # the re-inserted doc must be back to 'done' (real content, fp restored by distill)
            re_done = states_e.get(redistill_victim) == "done"
            raw_e = await rag.aget_docs_by_ids([doc_id(redistill_victim)])
            re_status = _doc_status(raw_e.get(doc_id(redistill_victim)))
            re_processed = re_status == DocStatus.PROCESSED
            led_e = await ledger.load("paper")
            fp_restored = (redistill_victim in led_e
                           and led_e[redistill_victim].fingerprint != "REDISTILL-PROBE-SENTINEL")
            # F2 invariant: REDISTILL introduced NO new dup-<hash> FAILED rows
            new_dups = sorted(set(dup_after) - set(dup_before))
            no_dup_pollution = not new_dups
            e_ok = re_done and re_processed and fp_restored and no_dup_pollution
            _mark(e_ok, "(e) REDISTILL delete-then-insert clean (F2)",
                  f"state={states_e.get(redistill_victim)} doc_status={re_status} "
                  f"fp_restored={fp_restored} new_dup_failed_rows={new_dups}")
            ok_all &= e_ok
        except Exception as exc:  # noqa: BLE001
            _mark(False, "(e) REDISTILL crashed", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            ok_all = False

        # ---- (d) one REMOVE closes the loop: ledger row gone + graph doc deleted ----
        # ★ Decouple (d) from (e): (e) churned done_keys[0] via REDISTILL_DELETE+re-insert, so
        # removing it would only prove "REMOVE a just-reinserted doc", leaving the pristine
        # to_distill→done REMOVE path unverified. Prefer a done key NOT touched by (e) so REMOVE
        # closes the loop on a clean done doc, decoupled from the redistill re-insert path.
        # (Both routes hit the same adelete_by_doc_id, but the assertion should stand on a
        # pristine doc.) Fallback to the redistill victim only if the subset has just one key.
        victim = next((k for k in done_keys if k != redistill_victim),
                      done_keys[0] if done_keys else keys[0])
        r = await remove_one(rag, victim, delete_ledger=True)
        await rag.apipeline_process_enqueue_documents()
        led_after = await ledger.load("paper")
        gone_from_ledger = victim not in led_after
        # confirm the graph doc is gone too: aget_docs_by_ids should no longer return it
        raw_after = await rag.aget_docs_by_ids([doc_id(victim)])
        gone_from_graph = doc_id(victim) not in raw_after
        d_ok = r == "removed" and gone_from_ledger and gone_from_graph
        _mark(d_ok, "(d) REMOVE closes loop",
              f"remove={r} ledger_gone={gone_from_ledger} graph_gone={gone_from_graph}")
        ok_all &= d_ok

    finally:
        # ---- cleanup: remove every subset key from graph + ledger (leave l0_probe clean) ----
        for k in keys:
            try:
                await remove_one(rag, k, delete_ledger=True)
            except Exception as e:  # noqa: BLE001
                print(f"PROBE .. | cleanup {k} skipped: {type(e).__name__}: {e}", flush=True)
        try:
            await rag.apipeline_process_enqueue_documents()
        except Exception:  # noqa: BLE001
            pass
        await ledger.close_pool()

    _mark(ok_all, "OVERALL", "all critical assertions passed" if ok_all else "see FAIL markers above")
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
