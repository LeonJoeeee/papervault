-- Knowledge System v3 — PostgreSQL schema initialization
-- Auto-run by Docker on first container init (mounted at /docker-entrypoint-initdb.d/)
-- Per docs/plans/knowledge-system-v3.md §5.2

-- Enable pgvector (LightRAG embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- Sidecar chunk metadata table (§5.2)
-- LightRAG stores chunk content + vector + graph internally (zero-mod)
-- This table stores per-chunk metadata: layer / type / source / tags
CREATE TABLE IF NOT EXISTS chunk_metadata (
    chunk_id           TEXT PRIMARY KEY,                       -- LightRAG chunk id
    layer              INTEGER NOT NULL DEFAULT 0,             -- 0 raw / 1 一阶 / 2 二阶 / 3 idea
    type               TEXT NOT NULL,                          -- paper_chunk / summary / fact / comparison / ...
    source_kind        TEXT NOT NULL,                          -- paper / review / textbook / web / notebook
    source             JSONB NOT NULL DEFAULT '{}'::jsonb,     -- {paper_doi, url, file_path, year, venue, ...}
    derived_from       JSONB NOT NULL DEFAULT '[]'::jsonb,     -- [chunk_id, ...] (empty = raw)
    derivation_depth   INTEGER NOT NULL DEFAULT 0,
    covers_questions   JSONB DEFAULT '[]'::jsonb,              -- [Q_id, ...] (Layer >= 1)
    group_by           JSONB DEFAULT NULL,                     -- 二阶 only: {entity, method, phenomenon, time}
    tags               JSONB NOT NULL DEFAULT '{}'::jsonb,     -- {method, phenomenon, domain, key_entities, time_period}
    confidence         TEXT DEFAULT 'medium',                  -- high / medium / low
    trust_signal       JSONB DEFAULT '{}'::jsonb,              -- {peer_reviewed, author_authority, author_synthesis_done, ...}
    last_updated       TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded_by      TEXT DEFAULT NULL,                      -- 软删 chunk_id (raw 永不 supersede)
    extra              JSONB DEFAULT '{}'::jsonb               -- 未来加新字段不动 schema
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_chunk_metadata_layer ON chunk_metadata(layer);
CREATE INDEX IF NOT EXISTS idx_chunk_metadata_type ON chunk_metadata(type);
CREATE INDEX IF NOT EXISTS idx_chunk_metadata_source_kind ON chunk_metadata(source_kind);
CREATE INDEX IF NOT EXISTS idx_chunk_metadata_tags_gin ON chunk_metadata USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_chunk_metadata_source_gin ON chunk_metadata USING GIN(source);
CREATE INDEX IF NOT EXISTS idx_chunk_metadata_derived_from_gin ON chunk_metadata USING GIN(derived_from);

-- last_updated auto-update trigger
CREATE OR REPLACE FUNCTION update_chunk_metadata_last_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunk_metadata_last_updated ON chunk_metadata;
CREATE TRIGGER trg_chunk_metadata_last_updated
    BEFORE UPDATE ON chunk_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_chunk_metadata_last_updated();
