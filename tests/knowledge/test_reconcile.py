"""Pure unit tests for the reconcile diff (SDD §6.1 阶段1). No DB / no files."""
from papervault.knowledge.ledger.store import LedgerRecord
from papervault.knowledge.scheduler.reconcile import diff


def _led(key: str, fp: str, status: str = "done") -> LedgerRecord:
    return LedgerRecord("l0_probe", "paper", key, fp, f"paper:{key}", status)


def test_diff_three_way_classification():
    # idx rec == its key (fp_of stubbed by key)
    idx = {"A": "A", "B": "B", "C": "C"}
    fps = {"A": "h1", "B": "h2_new", "C": "META"}
    led = {
        "B": _led("B", "h2_old"),   # changed → redistill
        "C": _led("C", "META"),     # unchanged (META==META) → nothing
        "D": _led("D", "h4"),       # gone from idx → remove
    }
    d = diff(idx, led, fp_of=lambda rec: fps[rec])

    assert sorted(k for k, _ in d.to_distill) == ["A"]        # new (not in ledger)
    assert sorted(k for k, _ in d.to_redistill) == ["B"]      # fingerprint changed
    assert d.to_remove == ["D"]                                # gone from index
    # C is unchanged → in no list
    touched = {k for k, _ in d.to_distill} | {k for k, _ in d.to_redistill} | set(d.to_remove)
    assert "C" not in touched


def test_diff_carries_fingerprint():
    idx = {"A": "A"}
    d = diff(idx, {}, fp_of=lambda rec: "deadbeef")
    assert d.to_distill == [("A", "deadbeef")]


def test_diff_empty_when_in_sync():
    idx = {"A": "A", "B": "B"}
    led = {"A": _led("A", "h1"), "B": _led("B", "h2")}
    d = diff(idx, led, fp_of=lambda rec: {"A": "h1", "B": "h2"}[rec])
    assert d.is_empty


def test_diff_error_row_retried_even_when_fingerprint_unchanged():
    # F9: an 'error' row whose fingerprint already equals the current hash must
    # still be re-driven (delete-then-insert), else it sticks in error forever.
    idx = {"A": "A", "B": "B"}
    fps = {"A": "h1", "B": "h2"}
    led = {
        "A": _led("A", "h1", status="error"),  # fp unchanged BUT error → retry
        "B": _led("B", "h2", status="done"),   # fp unchanged + done → nothing
    }
    d = diff(idx, led, fp_of=lambda rec: fps[rec])
    assert sorted(k for k, _ in d.to_redistill) == ["A"]
    assert d.to_redistill[0] == ("A", "h1")     # carries current fp
    assert d.to_distill == []
    assert d.to_remove == []


def test_diff_in_index_pending_remove_redistilled():
    # F3 (drill-r7): an in-index pending_remove row (REDISTILL_DELETE hit 403 busy,
    # old doc still in graph, key still in vault) has NO consumer in to_remove
    # (to_remove only collects key∉idx). If its fingerprint already equals the
    # current hash (e.g. the redistill originated from an 'error'/F9 retry), diff's
    # fp-change and 'error' branches both miss → it would stick forever (old doc
    # never deleted, new content never reinserted). So diff MUST re-drive it through
    # to_redistill (delete-then-insert, REDISTILL idempotent) — its续删 exit.
    idx = {"A": "A"}
    led = {"A": _led("A", "h1", status="pending_remove")}  # fp unchanged, key in idx
    d = diff(idx, led, fp_of=lambda rec: "h1")
    assert sorted(k for k, _ in d.to_redistill) == ["A"]
    assert d.to_redistill[0] == ("A", "h1")  # carries current fp
    assert d.to_distill == []
    assert d.to_remove == []


def test_diff_not_in_index_pending_remove_goes_to_remove():
    # The OTHER pending_remove path: a real REMOVE (key gone from vault) that hit 403.
    # key∉idx → to_remove re-drives the delete next round; NOT to_redistill (no point
    # reinserting a paper pl has purged).
    idx: dict[str, str] = {}
    led = {"A": _led("A", "h1", status="pending_remove")}  # key NOT in idx
    d = diff(idx, led, fp_of=lambda rec: "h1")
    assert d.to_remove == ["A"]
    assert d.to_redistill == []
    assert d.to_distill == []


def test_diff_only_keys_subset_scopes_all_lists():
    # blocker ③ subset entry: only_keys narrows BOTH idx and led before classifying.
    # A is new (in subset), B is gone-from-idx but OUTSIDE subset → must NOT be removed,
    # GONE_IN is gone-from-idx AND in subset → IS removed. This proves to_remove can only
    # name subset keys (else a subset round would delete every out-of-subset ledger row).
    idx = {"A": "A", "OTHER": "OTHER"}                 # OTHER in idx but not in subset
    led = {
        "B": _led("B", "h_b"),          # in ledger, gone from idx, OUTSIDE subset
        "GONE_IN": _led("GONE_IN", "g"),# in ledger, gone from idx, INSIDE subset
        "OTHER": _led("OTHER", "h_o"),  # in both but not in subset
    }
    d = diff(idx, led, fp_of=lambda rec: {"A": "h_a", "OTHER": "h_o"}[rec],
             only_keys={"A", "GONE_IN"})
    assert sorted(k for k, _ in d.to_distill) == ["A"]   # A new, in subset
    assert d.to_remove == ["GONE_IN"]                    # only the in-subset removal
    assert "B" not in d.to_remove                        # out-of-subset row untouched
    assert "OTHER" not in d.to_remove and not d.to_redistill
