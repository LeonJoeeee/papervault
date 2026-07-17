"""Root test config: environment isolation + the `integration` marker.

Unit tests (the CI set) must not touch live infrastructure. This machine may export
prod env (PAPER_LIBRARY_PATH etc.) pointing at a real vault; strip the vault vars so
unit tests get clean defaults / their own tmp fixtures. Tests needing GPU / Postgres /
Neo4j / network / a live LLM are marked `integration` and excluded from CI via
`pytest -m "not integration"`; run the full suite on reference hardware.
"""
import os
import sys
from pathlib import Path

import pytest

# --- environment isolation (before any papervault.config import resolves paths) ---
for _v in ("PAPER_LIBRARY_PATH", "PAPERVAULT_VAULT", "PAPERVAULT_DATA"):
    os.environ.pop(_v, None)

# The eval harness modules are flat scripts that cross-import by bare name
# (`import backbone`, `import stats`); put their dir on sys.path so those resolve.
_EVAL_DIR = Path(__file__).parent.parent / "src" / "papervault" / "eval"
if _EVAL_DIR.is_dir() and str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

# Whole modules that require live infrastructure.
_INTEGRATION_MODULES = {
    "test_ledger",           # Postgres ledger
    "test_rerank",           # GPU cross-encoder reranker
    "test_server_lifespan",  # scheduler + graph/DB lifecycle
    "test_background",       # background queue timing / daemons
}
# Exact test names (matched on the base function name, so parametrization is covered) that
# genuinely fail WITHOUT LLM credentials. Derived EMPIRICALLY (2026-07-17): every
# currently-marked candidate was run individually in a no-credentials env (PAPERVAULT_LLM_API_KEY
# unset, PAPERVAULT_LLM_KEYS/LLM_KEYS_FILE pointing at an absent file, a fresh empty
# PAPERVAULT_DATA so the default llm_keys.json is absent). Kept ONLY the ones that failed.
#
# The search_papers tool calls get_llm() as an upfront credential preflight
# (library/mcp/server.py) AFTER its empty-query Stage-0 guard but BEFORE the mocked
# collaborators run — so any test that drives it with a NON-empty query raises "No LLM
# credentials" and needs a key. The empty-query / docstring cases short-circuit before the
# preflight and are hermetic, so they were UN-marked (moved back into the CI set). The
# firecrawl re-entry quality-gate tests LOOKED hermetic in a local "no-creds" run, but that
# was ambient-credential leakage (real keys reachable on the dev box -> the gate silently ran a
# REAL judge LLM). On CI (guaranteed no creds) all 7 fail — CI is the ground truth for this
# classification, so they are marked integration by exact name.
_INTEGRATION_NAMES = {
    # test_mcp_server.py — search_papers get_llm() preflight (non-empty query).
    "test_search_malformed_intent_returns_error",
    "test_search_ingest_gate_and_minimal_records",
    "test_search_ingest_plugs_reject_egu_abstract_and_contentless_stub",
    "test_search_fair_share_no_term_starved",
    "test_search_rrf_consensus_floats_up_niche_still_surfaces",
    "test_search_judge_drop_is_not_fail_open",
    "test_search_sort_secondary_by_recency",
    "test_search_sort_secondary_by_importance",
    "test_search_judge_batches_dropped_zero_when_healthy",
    "test_search_ingest_upsert_passes_no_llm",
    # test_firecrawl_quality_gate.py — the re-entry gate drives a live judge LLM.
    "test_firecrawl_reentry_historical_md_passes_gate_stays_ok",
    "test_firecrawl_reentry_historical_stub_fails_gate_is_deleted_terminal",
    "test_firecrawl_reentry_historical_stub_no_abstract_to_failed",
    "test_firecrawl_reentry_gates_body_not_frontmatter",
    "test_firecrawl_reentry_gate_failopen_keeps_md",
    "test_firecrawl_reentry_pass_stamps_pdf_hunt_exhausted",
    "test_firecrawl_reentry_failopen_stamps_exhausted_no_reloop",
}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires GPU / Postgres / Neo4j / network / a live LLM"
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        mod = item.module.__name__.rsplit(".", 1)[-1]
        if mod in _INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
            continue
        # Match on the base function name so a future @parametrize can't silently unmark.
        base_name = getattr(item, "originalname", None) or item.name
        if base_name in _INTEGRATION_NAMES:
            item.add_marker(pytest.mark.integration)
