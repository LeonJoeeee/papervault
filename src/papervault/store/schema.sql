-- P1 shadow only. No public/LightRAG objects are referenced or modified.
-- Identity uniqueness is installed by the importer AFTER conflict classification.
CREATE SCHEMA IF NOT EXISTS catalog;

CREATE TABLE IF NOT EXISTS catalog.schema_version (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS catalog.records (
    key TEXT PRIMARY KEY,
    doi TEXT,
    arxiv_id TEXT,
    download_status TEXT NOT NULL,
    retrieval TEXT CHECK (retrieval IN
        ('ineligible', 'eligible', 'in-progress', 'exhausted', 'acquired', 'verified')),
    policy TEXT CHECK (policy IN ('clear', 'excluded', 'override', 'unknown')),
    merge TEXT CHECK (merge IN ('canonical', 'alias')),
    quarantine BOOLEAN,
    source TEXT,
    year INTEGER,
    title TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload_version INTEGER NOT NULL CHECK (payload_version = 1),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'
        AND payload ? 'key' AND payload ->> 'key' = key)
);
CREATE INDEX IF NOT EXISTS records_status_idx ON catalog.records (download_status);

CREATE TABLE IF NOT EXISTS catalog.aliases (
    alias_key TEXT PRIMARY KEY,
    canonical_key TEXT NOT NULL REFERENCES catalog.records(key),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (alias_key <> canonical_key)
);

CREATE TABLE IF NOT EXISTS catalog.audit (
    id BIGSERIAL PRIMARY KEY,
    at TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor TEXT NOT NULL CHECK (length(btrim(actor)) > 0),
    event TEXT NOT NULL CHECK (length(btrim(event)) > 0),
    key TEXT,
    detail JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog.rulings (
    id BIGSERIAL PRIMARY KEY,
    import_id TEXT NOT NULL UNIQUE,
    at TIMESTAMPTZ NOT NULL,
    action TEXT NOT NULL,
    scope TEXT NOT NULL,
    reasons JSONB NOT NULL,
    keys TEXT[] NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 1)
);

CREATE TABLE IF NOT EXISTS catalog.jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    key TEXT NOT NULL REFERENCES catalog.records(key),
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    detail JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog.derivations (
    id BIGSERIAL PRIMARY KEY,
    key TEXT NOT NULL REFERENCES catalog.records(key),
    artifact_inputs JSONB NOT NULL,
    catalog_revision BIGINT NOT NULL REFERENCES catalog.audit(id),
    recipe JSONB NOT NULL,
    published_at TIMESTAMPTZ NOT NULL
);

CREATE OR REPLACE FUNCTION catalog.reject_history_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'append-only history cannot be changed';
END;
$$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_trigger
                   WHERE tgrelid = 'catalog.audit'::regclass AND tgname = 'audit_append_only') THEN
        CREATE TRIGGER audit_append_only BEFORE UPDATE OR DELETE OR TRUNCATE
        ON catalog.audit FOR EACH STATEMENT EXECUTE FUNCTION catalog.reject_history_change();
    END IF;
    IF NOT EXISTS (SELECT FROM pg_trigger
                   WHERE tgrelid = 'catalog.rulings'::regclass AND tgname = 'rulings_append_only') THEN
        CREATE TRIGGER rulings_append_only BEFORE UPDATE OR DELETE OR TRUNCATE
        ON catalog.rulings FOR EACH STATEMENT EXECUTE FUNCTION catalog.reject_history_change();
    END IF;
END $$;
