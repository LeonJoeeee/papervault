"""Shared configuration: env loading, data paths, and LLM provider settings.

Everything an operator sets lives in one ``.env`` (see ``.env.example``). papervault
talks to ONE OpenAI-compatible LLM endpoint. Keys are supplied either as a single
``PAPERVAULT_LLM_API_KEY`` (the simple case) or, for scale, a JSON key-pool file
(``PAPERVAULT_LLM_KEYS`` — an array of ``{model, api_key, base_url}`` groups with
per-request failover). The key-pool file is the generic mechanism; nothing here is
tied to any specific provider.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> Path | None:
    """Load a single .env. Priority: PAPERVAULT_ENV, then CWD/.env, then repo-root/.env."""
    explicit = os.environ.get("PAPERVAULT_ENV")
    repo_root = Path(__file__).resolve().parent.parent.parent  # src/papervault/config.py
    candidates = [
        Path(explicit) if explicit else None,
        Path.cwd() / ".env",
        repo_root / ".env",
    ]
    for cand in candidates:
        if cand and cand.is_file():
            load_dotenv(cand)
            return cand
    return None


ENV_FILE = _load_env()

# LightRAG's Postgres storage reads POSTGRES_DATABASE (defaulting to 'postgres'), while the
# ingest ledger reads POSTGRES_DB — leave them unbridged and the knowledge graph's KV/vector/
# doc_status store silently targets a DIFFERENT database than the ledger. Bridge them: the graph
# store follows POSTGRES_DB unless the operator set POSTGRES_DATABASE explicitly.
if os.environ.get("POSTGRES_DB") and not os.environ.get("POSTGRES_DATABASE"):
    os.environ["POSTGRES_DATABASE"] = os.environ["POSTGRES_DB"]


def _path(env: str, default: str) -> Path:
    return Path(os.path.expanduser(os.environ.get(env, default)))


# --- data layout (all under one data dir by default) ---
DATA_DIR = _path("PAPERVAULT_DATA", "~/.local/share/papervault")
# The paper vault = the library's on-disk store (index.json + extracts). The legacy
# PAPER_LIBRARY_PATH is honored as a fallback for the originating deployment.
VAULT_PATH = Path(os.path.expanduser(
    os.environ.get("PAPERVAULT_VAULT")
    or os.environ.get("PAPER_LIBRARY_PATH")
    or str(DATA_DIR / "vault")
))
# LightRAG working dir (knowledge graph KV/vector/graphml blobs).
STORAGE_DIR = _path("PAPERVAULT_STORAGE", str(DATA_DIR / "knowledge_store"))
# LLM key-pool file (shared JSON, hot-reloaded). Legacy LLM_KEYS_FILE honored.
LLM_KEYS_FILE = Path(os.path.expanduser(
    os.environ.get("PAPERVAULT_LLM_KEYS")
    or os.environ.get("LLM_KEYS_FILE")
    or str(DATA_DIR / "llm_keys.json")
))
# Citation map (paper_key -> canonical cite), regenerable from the corpus.
CITATION_MAP = _path("PAPERVAULT_CITATION_MAP", str(DATA_DIR / "citation_map.json"))

# --- contact e-mail for scholarly-API polite pools (Crossref/OpenAlex User-Agent) ---
CONTACT_EMAIL = os.environ.get("PAPERVAULT_CONTACT_EMAIL", "papervault@example.invalid")

# --- MCP HTTP transport ---
MCP_ALLOWED_HOSTS: list[str] = [
    host.strip() for host in os.environ.get("PAPERVAULT_MCP_ALLOWED_HOSTS", "").split(",")
    if host.strip()
]

# --- LLM provider (OpenAI-compatible) ---
# Single-endpoint fallback when no key-pool file is present:
LLM_API_KEY = os.environ.get("PAPERVAULT_LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("PAPERVAULT_LLM_BASE_URL", "")
# Two model slots. SYNTH = the strong model (answer synthesis, research plane);
# BUILD = the cheaper model for graph extraction. Set them equal to use one model.
SYNTH_MODEL = os.environ.get("PAPERVAULT_MODEL", "")
BUILD_MODEL = os.environ.get("PAPERVAULT_BUILD_MODEL", "") or SYNTH_MODEL

# --- optional LiteLLM gateway (advanced: many keys behind a local proxy) ---
USE_GATEWAY = os.environ.get("PAPERVAULT_LLM_GATEWAY", "0") == "1"
GATEWAY_URL = os.environ.get("PAPERVAULT_LLM_GATEWAY_URL", "http://127.0.0.1:4000/v1")
GATEWAY_KEY = os.environ.get("PAPERVAULT_LLM_GATEWAY_KEY", "")
