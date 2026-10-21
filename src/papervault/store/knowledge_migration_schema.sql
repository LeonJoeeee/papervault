-- Explicit operator migration. No schema constructor installs this automatically.
CREATE TABLE IF NOT EXISTS catalog.knowledge_import_events (
 source TEXT NOT NULL, capture_id TEXT NOT NULL, source_row_id TEXT NOT NULL,
 source_version BIGINT NOT NULL, event_digest TEXT NOT NULL, evidence JSONB NOT NULL,
 PRIMARY KEY(source,capture_id,source_row_id,source_version)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_import_reports (
 plan_digest TEXT PRIMARY KEY, source TEXT NOT NULL, capture_id TEXT NOT NULL, report JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_import_heads (
 source TEXT NOT NULL, capture_id TEXT NOT NULL, source_row_id TEXT NOT NULL,
 source_version BIGINT NOT NULL, job_id TEXT UNIQUE REFERENCES catalog.jobs(id),
 deleted BOOLEAN NOT NULL, event_digest TEXT NOT NULL, projection_digest TEXT NOT NULL,
 PRIMARY KEY(source,capture_id,source_row_id)
);
ALTER TABLE catalog.knowledge_import_heads ADD COLUMN IF NOT EXISTS projection_digest TEXT;
ALTER TABLE catalog.knowledge_import_heads ADD COLUMN IF NOT EXISTS plan_digest TEXT;
ALTER TABLE catalog.knowledge_import_heads ADD COLUMN IF NOT EXISTS projection_revision BIGINT NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS catalog.knowledge_import_projections (
 source TEXT NOT NULL, capture_id TEXT NOT NULL, source_row_id TEXT NOT NULL,
 revision BIGINT NOT NULL, source_version BIGINT NOT NULL,
 plan_digest TEXT NOT NULL REFERENCES catalog.knowledge_import_reports(plan_digest),
 previous_plan_digest TEXT REFERENCES catalog.knowledge_import_reports(plan_digest),
 projection_digest TEXT NOT NULL, evidence JSONB NOT NULL,
 PRIMARY KEY(source,capture_id,source_row_id,revision)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_import_checkpoints (
 source TEXT NOT NULL, capture_id TEXT NOT NULL, checkpoint JSONB NOT NULL,
 plan_digest TEXT NOT NULL, revision BIGINT NOT NULL DEFAULT 1,
 PRIMARY KEY(source,capture_id)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_migration_operations (
 operation_id TEXT PRIMARY KEY, payload_digest TEXT NOT NULL, result JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_cutover_evidence (
 evidence_id TEXT PRIMARY KEY, source TEXT NOT NULL, capture_id TEXT NOT NULL,
 final_sequence BIGINT NOT NULL, evidence JSONB NOT NULL, digest TEXT NOT NULL
);
DO $$
DECLARE name TEXT;
BEGIN
 FOREACH name IN ARRAY ARRAY['knowledge_import_events','knowledge_import_reports','knowledge_import_projections','knowledge_migration_operations','knowledge_cutover_evidence']
 LOOP
  IF NOT EXISTS (SELECT FROM pg_trigger WHERE tgrelid=('catalog.'||name)::regclass
                 AND tgname='migration_append_only') THEN
   EXECUTE format('CREATE TRIGGER migration_append_only BEFORE UPDATE OR DELETE OR TRUNCATE ON catalog.%I '
                  'FOR EACH STATEMENT EXECUTE FUNCTION catalog.reject_history_change()',name);
  END IF;
 END LOOP;
END $$;
