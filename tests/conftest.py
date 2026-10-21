"""Root test config: environment isolation + the `integration` marker.

Unit tests (the CI set) must not touch live infrastructure. Disable dotenv loading
and scrub production configuration before application imports and each test's fixtures.
Catalog tests use JSON unless they explicitly opt into the guarded test DSN.
Tests needing GPU / Postgres / Neo4j / network / a live LLM are marked `integration` and excluded from CI via
`pytest -m "not integration"`; run the full suite on reference hardware.
"""
import os
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.version import Version
import pytest

# --- environment isolation (before ANY application import, including child conftests) ---
# Older dotenv ignores the subprocess-safe disable flag. Refuse that environment
# before collecting tests; the prescribed uv sync installs the locked 1.2.2 release.
if Version(version("python-dotenv")) < Version("1.2"):
    raise pytest.UsageError("Test isolation requires python-dotenv>=1.2; run uv sync --extra dev")

# Capture explicit backend-test inputs before scrubbing them from the process.
# Only the marked integration fixture can access this snapshot; runtime settings
# and ordinary tests retain the same isolated environment.
_BACKEND_TEST_VARIABLES = ("PAPERVAULT_BACKEND_TEST_DSN",) + tuple(
    "PAPERVAULT_BACKEND_TEST_NEO4J_" + suffix
    for suffix in ("URI", "USER", "PASSWORD", "DATABASE"))
# The default Neo4j database exception also needs an explicit disposable opt-in
# and the service's configured URI. Capture both without loading any .env.
_BACKEND_TEST_OPTIONAL_VARIABLES = ("PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE", "NEO4J_URI")
_BACKEND_TEST_TARGETS = {name: os.environ.get(name)
                         for name in _BACKEND_TEST_VARIABLES + _BACKEND_TEST_OPTIONAL_VARIABLES}

# Denylist: application flags/paths/tuning, knowledge stores and libpq discovery,
# OCR/model settings, provider credentials/endpoints, and legacy library settings.
# Prefixes also cover future settings in these namespaces. Keep shell/runtime vars
# (PATH, HOME, CUDA_VISIBLE_DEVICES, pytest options, etc.) for the test runner.
# The ONLY inherited application opt-in is the dedicated catalog test DSN; the
# catalog_test_dsn fixture validates it before a PostgreSQL connection is possible.
_PRODUCTION_ENV_PREFIXES = (
    "PAPERVAULT_", "KS_", "NEO4J_", "POSTGRES_", "PG", "MINERU_", "OPENALEX_",
    "PAPER_", "LIGHTRAG_", "BGE_", "MIMO_", "OPENAI_", "ANTHROPIC_", "LLM_",
    "FIRECRAWL_", "ZOTERO_",
)
_PRODUCTION_ENV_NAMES = {
    "ADS_API_TOKEN", "CORE_API_KEY", "WILEY_TDM_TOKEN", "ELSEVIER_TDM_API_KEY",
    "ANNAS_ARCHIVE_API_KEY", "UNPAYWALL_EMAIL", "ENTITY_EXTRACTION_USE_JSON", "RERANK_TIMEOUT",
}
# Even the default key-pool/data paths must not discover operator state under HOME.
_TEST_DATA = TemporaryDirectory(prefix="papervault-tests-")
_TEST_ENV_DEFAULTS = {
    "PYTHON_DOTENV_DISABLED": "1",  # Also prevents .env reads in subprocesses.
    "PAPERVAULT_DATA": _TEST_DATA.name,
    "PAPERVAULT_CATALOG_AUTHORITY": "json",
    "PAPERVAULT_TERMINAL_V2": "0",
    "PAPERVAULT_IDENTITY_V2": "0",
    "PAPERVAULT_REOPEN_V2": "0",
    "PAPERVAULT_POLITE_V2": "0",
    "PAPERVAULT_MERGE_V2": "0",
    "PAPERVAULT_STOPRULES_V2": "0",
    "PAPERVAULT_LLM_GATEWAY": "0",
    # Tests assert immediate saves unless they explicitly exercise debouncing (#34).
    "PAPERVAULT_SAVE_MIN_INTERVAL_S": "0",
    "PAPERVAULT_BIB_MIN_INTERVAL_S": "0",
}


def _is_production_setting(name):
    return name != "PAPERVAULT_CATALOG_TEST_DSN" and (
        name.startswith(_PRODUCTION_ENV_PREFIXES) or name in _PRODUCTION_ENV_NAMES
    )


for _name in tuple(os.environ):
    if _is_production_setting(_name):
        del os.environ[_name]
os.environ.update(_TEST_ENV_DEFAULTS)

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


@pytest.fixture
def backend_test_targets(request):
    """Explicit opt-in only; the backend harness validates targets before connecting."""
    if request.node.get_closest_marker("integration") is None:
        pytest.fail("backend_test_targets is restricted to integration tests", pytrace=False)
    return dict(_BACKEND_TEST_TARGETS)


@pytest.fixture(autouse=True)
def _isolate_test_environment(monkeypatch):
    """Reset leaks before fixtures; tests opt into their settings with monkeypatch."""
    for name in tuple(os.environ):
        if _is_production_setting(name):
            monkeypatch.delenv(name)
    for name, value in _TEST_ENV_DEFAULTS.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(autouse=True)
def _reset_search_breaker(_isolate_test_environment):
    """The #115 per-backend circuit breaker keeps PROCESS-LIFETIME state keyed by
    backend name. Without a reset it leaks across tests: three tests in a row
    degrading ``semantic_scholar`` would open its circuit, and the NEXT test's
    monkeypatched backend would silently never be called. Reset before AND after
    each test in this shared tests/conftest.py so no module inherits the state."""
    from papervault.library import search

    search._breaker_reset()
    yield
    search._breaker_reset()
