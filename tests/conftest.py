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
# Per-module test-name prefixes that need live infra inside an otherwise-unit module.
_INTEGRATION_NAME_PREFIXES = {
    "test_mcp_server": ("test_search_",),   # search_papers → live intent-parser LLM
}
# Exact test names that invoke a live LLM (the firecrawl re-entry quality gate) inside an
# otherwise-unit module. These pass locally only when ambient LLM creds happen to be present.
_INTEGRATION_NAMES = {
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
        if item.name in _INTEGRATION_NAMES:
            item.add_marker(pytest.mark.integration)
            continue
        for prefix in _INTEGRATION_NAME_PREFIXES.get(mod, ()):
            if item.name.startswith(prefix):
                item.add_marker(pytest.mark.integration)
                break
