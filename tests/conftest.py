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
# Search tests now mock the get_llm preflight in their server fixture, so they
# exercise P3 as hermetic unit tests. Firecrawl re-entry tests still drive a real
# judge LLM and remain integration tests.
_INTEGRATION_NAMES = {
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


# The lab .env (loaded by papervault.config at import) may opt into save debouncing;
# tests must see the shipped defaults (off) unless a test sets its own value (issue #34).
os.environ["PAPERVAULT_SAVE_MIN_INTERVAL_S"] = "0"
os.environ["PAPERVAULT_BIB_MIN_INTERVAL_S"] = "0"


@pytest.fixture(autouse=True)
def _reset_search_breaker():
    """The #115 per-backend circuit breaker keeps PROCESS-LIFETIME state keyed by
    backend name. Without a reset it leaks across tests: three tests in a row
    degrading ``semantic_scholar`` would open its circuit, and the NEXT test's
    monkeypatched backend would silently never be called. Reset before AND after
    each test in this shared tests/conftest.py so no module inherits the state."""
    from papervault.library import search

    search._breaker_reset()
    yield
    search._breaker_reset()
