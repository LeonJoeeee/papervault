"""Knowledge System config loader.

Env vars come from the single repo-root ``.env`` (gitignored), loaded by
``papervault.config``. See ``.env.example`` for the template.

The ``MIMO_API_KEY_*`` env path is retained only as legacy back-compat; the
documented LLM surface is a generic OpenAI-compatible endpoint (see the
``PAPERVAULT_LLM_*`` vars in ``.env.example``). Anthropic stays an optional
fallback for the Phase 3+ idea curator.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from papervault import config as _pv  # importing loads the single .env (side effect)


@dataclass(frozen=True)
class PostgresConfig:
    user: str
    password: str
    db: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        # Defaults track deploy/docker-compose.yml (which creates user/db 'papervault')
        # so a bare install with only POSTGRES_PASSWORD set connects to the container it
        # ships. A divergent default here would let `papervault doctor` pass against one
        # database while the runtime (ledger + LightRAG KV/vector store) targets another,
        # nonexistent one. cli.doctor imports THIS resolver so the two never drift.
        return cls(
            user=os.getenv("POSTGRES_USER", "papervault"),
            password=os.getenv("POSTGRES_PASSWORD", "changeme"),
            db=os.getenv("POSTGRES_DB", "papervault"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
        )

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def asyncpg_dsn(self) -> str:
        return f"postgres://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class MimoEndpoint:
    """A single MiMo API endpoint (key + base URL)."""

    api_key: str
    base_url: str

    @property
    def is_valid(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("tp-"))


@dataclass(frozen=True)
class MimoConfig:
    """MiMo pool config — now a thin view over the CENTRAL hot-reloaded key pool
    (the pool file at PAPERVAULT_LLM_KEYS / legacy LLM_KEYS_FILE, see store/llm.py). ``model`` and
    ``valid_endpoint_count`` defer to the pool so they reflect the ACTIVE groups in the
    shared file (hot-reloaded); the ``MIMO_API_KEY_*`` .env path is kept as a fallback
    inside store.llm when the file is missing/broken. ``endpoints`` is retained only for
    backward compat (old per-endpoint .env view); the live pool is the source of truth."""

    endpoints: tuple[MimoEndpoint, ...]

    @classmethod
    def from_env(cls) -> "MimoConfig":
        default_base = _pv.LLM_BASE_URL
        endpoints = []
        # Key 1 uses default base URL (SGP), no suffix in env
        endpoints.append(
            MimoEndpoint(
                api_key=os.getenv("MIMO_API_KEY_1", ""),
                base_url=os.getenv("MIMO_BASE_URL_1", default_base),
            )
        )
        # Keys 2-6 each have explicit base URL
        for i in range(2, 7):
            endpoints.append(
                MimoEndpoint(
                    api_key=os.getenv(f"MIMO_API_KEY_{i}", ""),
                    base_url=os.getenv(f"MIMO_BASE_URL_{i}", default_base),
                )
            )
        return cls(
            endpoints=tuple(ep for ep in endpoints if ep.is_valid),
        )

    @property
    def model(self) -> str:
        """Bare SDK model name from the first active group in the central pool."""
        from papervault.knowledge.store.llm import pool_model

        return pool_model()

    @property
    def valid_endpoint_count(self) -> int:
        """Number of ACTIVE groups in the central key file (hot-reloaded)."""
        from papervault.knowledge.store.llm import active_endpoint_count

        return active_endpoint_count()


@dataclass(frozen=True)
class AnthropicConfig:
    """Optional, only for Phase 3+ idea curator if MiMo insufficient."""

    api_key: str

    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        return cls(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    @property
    def is_set(self) -> bool:
        return bool(
            self.api_key
            and self.api_key.startswith("sk-ant-")
            and "..." not in self.api_key
            and len(self.api_key) > 30
        )


@dataclass(frozen=True)
class LightRAGConfig:
    working_dir: str
    # Chunk SIZE target (unchanged from LightRAG's prior default) — kept as config so
    # it stays easy to tune. The boundary-aware chunker (ingest.chunking) only changes
    # WHERE the cut lands (sentence boundary), never this size.
    chunk_token_size: int
    chunk_overlap_token_size: int

    @classmethod
    def from_env(cls) -> "LightRAGConfig":
        return cls(
            working_dir=os.getenv("LIGHTRAG_WORKING_DIR", str(_pv.STORAGE_DIR)),
            chunk_token_size=int(os.getenv("KS_CHUNK_TOKEN_SIZE", "2400")),
            chunk_overlap_token_size=int(os.getenv("KS_CHUNK_OVERLAP_TOKEN_SIZE", "200")),
        )


@dataclass(frozen=True)
class BGEM3Config:
    model_path: str
    device: str

    @classmethod
    def from_env(cls) -> "BGEM3Config":
        return cls(
            model_path=os.getenv("BGE_M3_MODEL_PATH", "BAAI/bge-m3"),
            device=os.getenv("BGE_M3_DEVICE", "cuda:0"),
        )


@dataclass(frozen=True)
class BGEM3RerankerConfig:
    """BGE reranker (SDD §6.9.1 V1). Same BAAI family as BGE-M3, in-process on KS's pinned
    RTX 3090 (device cuda:0 resolves to the physical 3090 via CUDA_VISIBLE_DEVICES=1, .env 2026-06-07).
    model_path/device come from .env (BGE_RERANKER_MODEL_PATH / BGE_RERANKER_DEVICE),
    defaulting to the v2-m3 reranker + cuda:0 — parallel to BGEM3Config."""

    model_path: str
    device: str

    @classmethod
    def from_env(cls) -> "BGEM3RerankerConfig":
        return cls(
            model_path=os.getenv("BGE_RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3"),
            device=os.getenv("BGE_RERANKER_DEVICE", "cuda:0"),
        )


@dataclass(frozen=True)
class Config:
    postgres: PostgresConfig
    mimo: MimoConfig
    anthropic: AnthropicConfig
    lightrag: LightRAGConfig
    bge_m3: BGEM3Config
    bge_reranker: BGEM3RerankerConfig

    @classmethod
    def load(cls) -> "Config":
        return cls(
            postgres=PostgresConfig.from_env(),
            mimo=MimoConfig.from_env(),
            anthropic=AnthropicConfig.from_env(),
            lightrag=LightRAGConfig.from_env(),
            bge_m3=BGEM3Config.from_env(),
            bge_reranker=BGEM3RerankerConfig.from_env(),
        )


CONFIG = Config.load()
