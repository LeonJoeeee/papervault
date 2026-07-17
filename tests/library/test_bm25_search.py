"""Direct unit tests for the pure functions in ``services/bm25_search.py``
(no LLM, no network): ``bm25_per_term_ranklists``, ``round_robin``, ``_node_key``,
and ``paper_to_dict`` — the V6 library-arm BM25 + fair-share round-robin shortlister
(search_papers Stage 1 + Stage 2.5, §2/§4).
"""

from __future__ import annotations

from papervault.library.services.bm25_search import (
    RRF_K,
    _node_key,
    bm25_per_term_ranklists,
    paper_to_dict,
    reorder_ranklists_by_rrf,
    round_robin,
    rrf_fuse,
)


# ───────────────────────── bm25_per_term_ranklists ─────────────────────────


def test_bm25_empty_pool_returns_empty_list_per_term():
    """Empty pool → one EMPTY ranklist per term (NEVER construct BM25Okapi([]))."""
    out = bm25_per_term_ranklists([], ["alpha", "beta", "gamma"])
    assert out == [[], [], []]


# A pool of "distractor" docs that share NO tokens with the matched term, so the
# matched docs stay a MINORITY → BM25 IDF stays POSITIVE (Okapi IDF goes
# negative once a term appears in > half the corpus, which would zero out the
# > 0 filter for tiny degenerate corpora).
def _distractors(n: int) -> list[dict]:
    return [{"key": f"D{i}", "title": "unrelated quantum chromodynamics lattice gauge",
             "abstract": "lattice"} for i in range(n)]


def test_bm25_empty_token_term_returns_empty():
    """A degenerate term with NO [a-z0-9] tokens (punctuation only) tokenizes to
    [] → an EMPTY ranklist (NOT pool[:k]). _tokenize is ASCII-alphanumeric only,
    so a punctuation-only string is the realistic empty-token trigger."""
    pool = [{"key": "P1", "title": "cosmic ray transport modulation", "abstract": "physics"}]
    pool += _distractors(5)
    out = bm25_per_term_ranklists(pool, ["--- !!! ...", "transport modulation"])
    assert out[0] == []                       # empty-token term → no votes
    assert [n["key"] for n in out[1]] == ["P1"]   # real term still scores the doc


def test_bm25_all_empty_vocab_corpus_returns_empty_no_crash():
    """A NON-empty pool whose EVERY doc tokenizes to [] (all text fields empty and
    a None year, so ``_paper_text`` is "" → no [a-z0-9] tokens) yields an empty
    global vocabulary. ``BM25Okapi`` would raise ZeroDivisionError in _calc_idf
    (avg_idf = idf_sum / 0); the empty-VOCAB guard must degrade to no-votes
    (one [] per term), NEVER crash the tool."""
    pool = [
        {"key": "C1", "title": "", "venue": "", "abstract": "",
         "authors": [], "year": None},
        {"key": "C2", "title": "", "venue": "", "abstract": "",
         "authors": [""], "year": None},
    ]
    out = bm25_per_term_ranklists(pool, ["cosmic ray transport", "modulation"])
    assert out == [[], []]                     # no votes, no ZeroDivisionError crash


def test_bm25_score_gt_zero_filter():
    """Only docs with BM25 score > 0 for a term appear in that term's ranklist;
    a doc sharing NO tokens with the term is excluded."""
    pool = [
        {"key": "HIT", "title": "cosmic ray transport modulation", "abstract": "x"},
    ] + _distractors(5)
    out = bm25_per_term_ranklists(pool, ["cosmic ray transport modulation"])
    keys = [n["key"] for n in out[0]]
    assert keys == ["HIT"]                     # only the matching doc; distractors filtered


def test_bm25_equal_score_tiebreak_orders_by_ascending_key():
    """Two docs with IDENTICAL text get IDENTICAL BM25 scores; the (-score, key)
    tiebreak orders them by ASCENDING stable Paper.key. The matched pair stays a
    minority (distractors padding) so the shared IDF is positive."""
    pool = [
        {"key": "Bbb", "title": "cosmic ray transport modulation", "abstract": "same body text"},
        {"key": "Aaa", "title": "cosmic ray transport modulation", "abstract": "same body text"},
    ] + _distractors(6)
    out = bm25_per_term_ranklists(pool, ["cosmic ray transport modulation"])
    assert [n["key"] for n in out[0]] == ["Aaa", "Bbb"]   # ascending key tiebreak


# ───────────────────────── round_robin ─────────────────────────


def test_round_robin_shared_head_emitted_once():
    """A node that is the HEAD of two ranklists is emitted EXACTLY ONCE — the
    second term's cursor skips it (its key is already in ``emitted``)."""
    shared = {"key": "SHARED", "title": "shared"}
    n0b = {"key": "T0B", "title": "t0b"}
    n1b = {"key": "T1B", "title": "t1b"}
    ranklists = [[shared, n0b], [shared, n1b]]
    out = round_robin(ranklists, cap=10, key_fn=lambda n: n["key"])
    keys = [n["key"] for n in out]
    assert keys.count("SHARED") == 1          # shared head emitted once
    assert set(keys) == {"SHARED", "T0B", "T1B"}
    # Interleave order: term0 head (SHARED), term1 head is SHARED→skip→T1B,
    # then cycle 2: term0 next = T0B, term1 exhausted.
    assert keys == ["SHARED", "T1B", "T0B"]


def test_round_robin_cap_truncates_mid_cycle_in_interleave_order():
    """``cap`` truncates mid-cycle: the emit stops the instant the pool reaches cap,
    in round-robin interleave order (term0[0], term1[0], term0[1], ...)."""
    rl0 = [{"key": "A0"}, {"key": "A1"}, {"key": "A2"}]
    rl1 = [{"key": "B0"}, {"key": "B1"}, {"key": "B2"}]
    out = round_robin([rl0, rl1], cap=3, key_fn=lambda n: n["key"])
    assert [n["key"] for n in out] == ["A0", "B0", "A1"]   # cap hit mid-cycle


def test_round_robin_skips_empty_ranklists():
    """A term with an empty ranklist is inactive and never blocks the cycle."""
    rl0 = [{"key": "A0"}, {"key": "A1"}]
    out = round_robin([[], rl0, []], cap=10, key_fn=lambda n: n["key"])
    assert [n["key"] for n in out] == ["A0", "A1"]


# ───────────────────────── rrf_fuse ─────────────────────────

_KEY = lambda n: n["key"]   # noqa: E731 (test-local per-arm key_fn)


def test_rrf_fuse_single_list_is_reciprocal_of_k_plus_rank():
    """A node alone in ONE list at 0-based ``rank0`` scores exactly 1/(k+rank0)."""
    rl = [{"key": "A"}, {"key": "B"}, {"key": "C"}]
    rrf = rrf_fuse([rl], _KEY)
    assert rrf["A"] == 1.0 / (RRF_K + 0)
    assert rrf["B"] == 1.0 / (RRF_K + 1)
    assert rrf["C"] == 1.0 / (RRF_K + 2)


def test_rrf_fuse_accumulates_consensus_across_lists():
    """A node appearing in MULTIPLE lists accrues the SUM of its per-list
    1/(k+rank0) — cross-term/cross-backend CONSENSUS. A node high in two lists
    outscores a node that is the lone head of a single list."""
    consensus = {"key": "CONS"}      # head of BOTH lists
    lone_head = {"key": "LONE"}      # head of one list only
    rl0 = [consensus, lone_head]     # CONS rank0=0, LONE rank0=1
    rl1 = [consensus, {"key": "X"}]  # CONS rank0=0 again
    rrf = rrf_fuse([rl0, rl1], _KEY)
    assert rrf["CONS"] == 1.0 / RRF_K + 1.0 / RRF_K          # summed over both lists
    assert rrf["LONE"] == 1.0 / (RRF_K + 1)                  # one list only
    # consensus (two head appearances) > a single list's lone head
    assert rrf["CONS"] > rrf["X"]
    assert rrf["CONS"] > rrf["LONE"]


def test_rrf_fuse_empty_lists_contribute_nothing():
    """Empty ranklists add no scores; a niche node alone in one list still scores."""
    rrf = rrf_fuse([[], [{"key": "N"}], []], _KEY)
    assert rrf == {"N": 1.0 / RRF_K}


def test_rrf_fuse_stamps_nothing_on_nodes():
    """rrf_fuse is PURE — it returns a dict and mutates no node."""
    n = {"key": "A"}
    rrf_fuse([[n]], _KEY)
    assert set(n) == {"key"}     # no _rrf / score stamped


# ─────────────────── reorder_ranklists_by_rrf ───────────────────


def test_reorder_floats_consensus_to_each_terms_head():
    """A paper with cross-term consensus rises to the HEAD of every term's list,
    even when it sat lower in that term's native order."""
    cons = {"key": "CONS"}
    rl0 = [{"key": "A0"}, cons]      # CONS is rank0=1 here (low native)
    rl1 = [cons, {"key": "B1"}]      # CONS is rank0=0 here (consensus)
    rrf = rrf_fuse([rl0, rl1], _KEY)
    out = reorder_ranklists_by_rrf([rl0, rl1], rrf, _KEY)
    # CONS has the highest RRF (two appearances) → head of BOTH reordered lists.
    assert out[0][0]["key"] == "CONS"
    assert out[1][0]["key"] == "CONS"


def test_reorder_single_entry_term_is_unchanged_niche_preserved():
    """A term holding ONE niche entry is unchanged — its lone head stays its head,
    so the round-robin floor that surfaces the niche is never disturbed by RRF."""
    niche = [{"key": "NICHE"}]
    deep = [{"key": f"D{i}"} for i in range(10)]
    rrf = rrf_fuse([deep, niche], _KEY)
    out = reorder_ranklists_by_rrf([deep, niche], rrf, _KEY)
    assert out[1] == [{"key": "NICHE"}]      # niche term untouched


def test_reorder_rrf_tie_falls_back_to_native_order():
    """When two nodes have EQUAL RRF (both lone, same rank pattern), the original
    list position is the stable tiebreak — native order is preserved on ties."""
    rl = [{"key": "first"}, {"key": "second"}, {"key": "third"}]
    rrf = rrf_fuse([rl], _KEY)   # strictly descending by position → no real tie,
    # but build an artificial all-equal rrf to exercise the tiebreak branch:
    flat = {k: 1.0 for k in ("first", "second", "third")}
    out = reorder_ranklists_by_rrf([rl], flat, _KEY)
    assert [n["key"] for n in out[0]] == ["first", "second", "third"]
    # sanity: with the real rrf the order is identical here (already DESC)
    out2 = reorder_ranklists_by_rrf([rl], rrf, _KEY)
    assert [n["key"] for n in out2[0]] == ["first", "second", "third"]


def test_reorder_is_pure_returns_new_lists_same_refs():
    """reorder returns NEW lists holding the SAME node objects (no copy, no mutate)."""
    n0, n1 = {"key": "A"}, {"key": "B"}
    src = [[n0, n1]]
    rrf = rrf_fuse(src, _KEY)
    out = reorder_ranklists_by_rrf(src, rrf, _KEY)
    assert out is not src and out[0] is not src[0]    # new container
    assert out[0][0] is n0 and out[0][1] is n1        # same node refs, no mutation
    assert set(n0) == {"key"}                         # nothing stamped


def test_rrf_composition_consensus_up_but_niche_still_surfaces():
    """THE LOCKED COMPOSITION end-to-end (bm25-level): RRF floats a cross-term
    consensus paper to its term's head, AND the round-robin diversity floor still
    surfaces a niche-but-relevant paper that only one term ranks. Both hold."""
    cons = {"key": "CONS"}                       # consensus across term-0 and term-2
    niche = {"key": "NICHE"}                      # alone in term-1
    rl0 = [{"key": "A0"}, {"key": "A1"}, cons]    # CONS low in term-0 native order
    rl1 = [niche]                                 # the niche per-term hit
    rl2 = [cons, {"key": "C1"}]                   # CONS head of term-2 → consensus
    ranklists = [rl0, rl1, rl2]
    rrf = rrf_fuse(ranklists, _KEY)
    reordered = reorder_ranklists_by_rrf(ranklists, rrf, _KEY)
    # CONSENSUS rose to term-0's head (was rank 2 natively).
    assert reordered[0][0]["key"] == "CONS"
    pool = round_robin(reordered, cap=10, key_fn=_KEY)
    keys = [n["key"] for n in pool]
    # NICHE is NOT dropped — the floor (term-1 leads its slot in cycle 1) keeps it.
    assert "NICHE" in keys
    # NICHE appears EARLY (cycle 1: term0 head, term1 head=NICHE, ...).
    assert keys.index("NICHE") == 1
    # CONS emitted exactly once despite appearing under two terms.
    assert keys.count("CONS") == 1


# ───────────────────────── _node_key ─────────────────────────


def test_node_key_doi_wins_over_everything():
    node = {"doi": "10.1/ABC", "arxiv_id": "2401.0001", "title": "Some Title"}
    assert _node_key(node) == "doi:10.1/abc"          # normalized (lowercased)


def test_node_key_arxiv_version_stripped_when_no_doi():
    assert _node_key({"arxiv_id": "2401.0001v3", "title": "T"}) == "arx:2401.0001"
    assert _node_key({"arxiv_id": "2401.0001V2"}) == "arx:2401.0001"   # case-insensitive


def test_node_key_title_when_no_doi_no_arxiv():
    node = {"title": "Hello World!"}
    assert _node_key(node) == "ttl:hello world"       # normalize_title


def test_node_key_precedence_full_ladder():
    """doi > arxiv(version-stripped) > title > _uid, checked in order."""
    assert _node_key({"doi": "10.1/x", "arxiv_id": "2401.0001v1",
                      "title": "t"}).startswith("doi:")
    assert _node_key({"arxiv_id": "2401.0001v1", "title": "t"}).startswith("arx:")
    assert _node_key({"title": "t"}).startswith("ttl:")


def test_node_key_two_all_empty_identity_nodes_get_distinct_keys():
    """Two nodes with NO doi / arxiv / title fall back to a per-OBJECT ``_uid``
    sentinel → DISTINCT keys (they ARE distinct works, must not collide)."""
    a = {"doi": "", "arxiv_id": "", "title": ""}
    b = {"doi": "", "arxiv_id": "", "title": ""}
    ka, kb = _node_key(a), _node_key(b)
    assert ka.startswith("_uid:") and kb.startswith("_uid:")
    assert ka != kb                                   # distinct per-object sentinels
    assert _node_key(a) == ka                         # stable for the same object


# ───────────────────────── paper_to_dict ─────────────────────────


def test_paper_to_dict_dict_passthrough_adds_external_origin():
    """A dict (external candidate) passes through, gaining _source_origin=external."""
    src = {"title": "Ext Paper", "doi": "10.1/x"}
    out = paper_to_dict(src)
    assert out["_source_origin"] == "external"
    assert out["title"] == "Ext Paper" and out["doi"] == "10.1/x"
    assert out is not src                              # a copy, not the same object


def test_paper_to_dict_dict_keeps_existing_origin():
    """An explicit _source_origin is NOT clobbered (setdefault)."""
    out = paper_to_dict({"title": "T", "_source_origin": "library"})
    assert out["_source_origin"] == "library"


def test_paper_to_dict_paper_like_object_maps_to_library_dict():
    """A Paper-like model object maps to the homogeneous library dict shape with
    _source_origin=library and the projected research fields."""
    class _PaperLike:
        key = "Smith2024"
        title = "A Paper Title"
        authors = ["Smith", "Jones"]
        year = 2024
        venue = "ApJ"
        abstract = "an abstract"
        doi = "10.1/smith"
        arxiv_id = "2401.0001"
        citation_count = 12
        is_review = True
        publication_types = ["JournalArticle"]

    out = paper_to_dict(_PaperLike())
    assert out["_source_origin"] == "library"
    assert out["key"] == "Smith2024"
    assert out["authors"] == ["Smith", "Jones"]
    assert out["year"] == 2024
    assert out["citation_count"] == 12
    assert out["is_review"] is True
    assert out["publication_types"] == ["JournalArticle"]


def test_paper_to_dict_real_paper_object():
    """Sanity: the real Paper model also maps to a library dict."""
    from papervault.library.models import Paper
    p = Paper(key="K1", title="Real Paper", authors=["A"], year=2020,
              doi="10.9/real")
    out = paper_to_dict(p)
    assert out["_source_origin"] == "library"
    assert out["key"] == "K1"
    assert out["doi"] == "10.9/real"
