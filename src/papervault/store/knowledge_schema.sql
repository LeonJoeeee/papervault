-- Explicit P4 migration only; runtime constructors never execute this file.
CREATE TABLE IF NOT EXISTS catalog.knowledge_runtime (
 singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK(singleton),
 authority TEXT NOT NULL DEFAULT 'legacy' CHECK(authority IN ('legacy','catalog')),
 authority_epoch BIGINT NOT NULL DEFAULT 1,
 checkpoint JSONB NOT NULL DEFAULT '{}',
 fencing_counter BIGINT NOT NULL DEFAULT 0,
 requirements_revision BIGINT NOT NULL DEFAULT 0,
 budget_limit NUMERIC NOT NULL DEFAULT 1000000 CHECK(budget_limit >= 0),
 budget_reserved NUMERIC NOT NULL DEFAULT 0 CHECK(budget_reserved >= 0),
 serving_generation_id TEXT,
 serving_manifest_revision BIGINT,
 serving_revision BIGINT NOT NULL DEFAULT 0
 ,maintenance_revision BIGINT NOT NULL DEFAULT 0
);
INSERT INTO catalog.knowledge_runtime(singleton) VALUES(TRUE) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS catalog.units (
 unit_id TEXT PRIMARY KEY,
 kind TEXT NOT NULL CHECK(kind IN ('paper','operator_document','section')),
 record_key TEXT REFERENCES catalog.records(key),
 identity JSONB NOT NULL,
 quarantine BOOLEAN NOT NULL DEFAULT FALSE,
 quarantine_reason JSONB NOT NULL DEFAULT '{}',
 revision BIGINT NOT NULL DEFAULT 1
 ,UNIQUE(kind,identity)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_maintenance_windows (
 window_id TEXT PRIMARY KEY,
 calendar_month TEXT NOT NULL UNIQUE,
 timezone TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind='monthly_reboot'),
 starts_at TIMESTAMPTZ NOT NULL,
 ends_at TIMESTAMPTZ NOT NULL,
 operator_nonce TEXT NOT NULL UNIQUE,
 CHECK(ends_at>starts_at AND ends_at-starts_at<=interval '2 hours')
);
CREATE TABLE IF NOT EXISTS catalog.unit_mappings (
 source TEXT NOT NULL, legacy_key TEXT NOT NULL, unit_id TEXT NOT NULL REFERENCES catalog.units,
 PRIMARY KEY(source,legacy_key)
);
CREATE INDEX IF NOT EXISTS units_record_key ON catalog.units(record_key);
CREATE TABLE IF NOT EXISTS catalog.section_mappings (
 unit_id TEXT NOT NULL REFERENCES catalog.units,
 mapping_revision BIGINT NOT NULL CHECK(mapping_revision>0),
 parent_unit_id TEXT NOT NULL REFERENCES catalog.units,
 parent_evidence_revision TEXT NOT NULL,
 definition JSONB NOT NULL,
 PRIMARY KEY(unit_id,mapping_revision), CHECK(unit_id<>parent_unit_id)
);
CREATE INDEX IF NOT EXISTS section_mappings_parent ON catalog.section_mappings(parent_unit_id);
CREATE TABLE IF NOT EXISTS catalog.unit_inventories (
 parent_unit_id TEXT NOT NULL REFERENCES catalog.units,
 inventory_revision TEXT NOT NULL,
 parent_evidence_revision TEXT NOT NULL,
 parent_consumed_input_revision TEXT,
 mode TEXT NOT NULL CHECK(mode IN ('whole','sections')),
 required_units JSONB NOT NULL, optional_units JSONB NOT NULL,
 section_mapping_revisions JSONB NOT NULL,
 PRIMARY KEY(parent_unit_id,inventory_revision)
);
CREATE TABLE IF NOT EXISTS catalog.input_manifests (
 consumed_input_revision TEXT PRIMARY KEY,
 unit_id TEXT NOT NULL REFERENCES catalog.units,
 manifest JSONB NOT NULL,
 manifest_digest TEXT NOT NULL,
 captured_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS catalog.recipes (
 recipe_digest TEXT PRIMARY KEY, spec JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS catalog.generations (
 generation_id TEXT PRIMARY KEY,
 state TEXT NOT NULL CHECK(state IN ('active','candidate','retired')),
 manifest_revision BIGINT NOT NULL DEFAULT 0,
 requirements_revision BIGINT NOT NULL,
 source_generation_id TEXT REFERENCES catalog.generations,
 source_manifest_revision BIGINT,
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE catalog.generations ADD COLUMN IF NOT EXISTS retrieval_contract JSONB;
CREATE UNIQUE INDEX IF NOT EXISTS generations_one_active ON catalog.generations(state) WHERE state='active';
CREATE TABLE IF NOT EXISTS catalog.current_inventories (
 parent_unit_id TEXT NOT NULL REFERENCES catalog.units,
 generation_id TEXT NOT NULL REFERENCES catalog.generations,
 inventory_revision TEXT NOT NULL,
 revision BIGINT NOT NULL DEFAULT 1,
 record_revision BIGINT,
 invalidated BOOLEAN NOT NULL DEFAULT FALSE,
 enqueue_mode TEXT NOT NULL DEFAULT 'active' CHECK(enqueue_mode IN ('active','candidate')),
 request_id TEXT,
 desired_recipe_digest TEXT REFERENCES catalog.recipes,
 PRIMARY KEY(parent_unit_id,generation_id),
 FOREIGN KEY(parent_unit_id,inventory_revision) REFERENCES catalog.unit_inventories(parent_unit_id,inventory_revision)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_requirements (
 unit_id TEXT NOT NULL REFERENCES catalog.units,
 generation_id TEXT NOT NULL REFERENCES catalog.generations,
 consumed_input_revision TEXT NOT NULL REFERENCES catalog.input_manifests,
 recipe_digest TEXT NOT NULL REFERENCES catalog.recipes,
 record_revision BIGINT,
 inventory_revision TEXT,
 revision BIGINT NOT NULL DEFAULT 1,
 invalidated BOOLEAN NOT NULL DEFAULT FALSE,
 excluded BOOLEAN NOT NULL DEFAULT FALSE,
 enqueue_mode TEXT NOT NULL DEFAULT 'active' CHECK(enqueue_mode IN ('active','candidate')),
 PRIMARY KEY(unit_id,generation_id)
);
CREATE TABLE IF NOT EXISTS catalog.generation_manifests (
 generation_id TEXT NOT NULL REFERENCES catalog.generations,
 manifest_revision BIGINT NOT NULL,
 requirements_revision BIGINT NOT NULL,
 manifest JSONB NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(generation_id,manifest_revision)
);
ALTER TABLE catalog.jobs ALTER COLUMN key DROP NOT NULL;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS unit_id TEXT REFERENCES catalog.units;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS consumed_input_revision TEXT REFERENCES catalog.input_manifests;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS recipe_digest TEXT REFERENCES catalog.recipes;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS target_generation TEXT REFERENCES catalog.generations;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS revision BIGINT NOT NULL DEFAULT 1;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS achieved_tier TEXT;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS quarantine BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS quarantine_reason JSONB NOT NULL DEFAULT '{}';
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS current_attempt_id TEXT;
ALTER TABLE catalog.jobs ADD COLUMN IF NOT EXISTS predecessor_job_id TEXT REFERENCES catalog.jobs;
CREATE UNIQUE INDEX IF NOT EXISTS distill_identity ON catalog.jobs
 (unit_id,consumed_input_revision,recipe_digest,target_generation) WHERE kind='distill';
DO $$ BEGIN
 IF NOT EXISTS(SELECT FROM pg_constraint WHERE conrelid='catalog.jobs'::regclass AND conname='distill_identity_required') THEN
  ALTER TABLE catalog.jobs ADD CONSTRAINT distill_identity_required CHECK(kind<>'distill' OR
    (unit_id IS NOT NULL AND consumed_input_revision IS NOT NULL AND recipe_digest IS NOT NULL
     AND target_generation IS NOT NULL AND state IN ('queued','running','succeeded','failed','cancelled')
     AND (achieved_tier IS NULL OR achieved_tier IN ('fulltext','abstract','metadata','unknown'))));
 END IF;
END $$;
CREATE TABLE IF NOT EXISTS catalog.knowledge_attempts (
 attempt_id TEXT PRIMARY KEY,
 job_id TEXT NOT NULL REFERENCES catalog.jobs,
 sequence INTEGER NOT NULL CHECK(sequence>0),
 submission_id TEXT NOT NULL UNIQUE,
 authority_epoch BIGINT NOT NULL,
 fencing_token BIGINT NOT NULL,
 lease_id TEXT NOT NULL,
 lease_expires_at TIMESTAMPTZ NOT NULL,
 namespace TEXT NOT NULL UNIQUE,
 started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 finished_at TIMESTAMPTZ,
 outcome TEXT,
 error JSONB,
 reserved_cost NUMERIC NOT NULL DEFAULT 0,
 actual_cost NUMERIC,
 cost_state TEXT NOT NULL DEFAULT 'reserved',
 accepted BOOLEAN NOT NULL DEFAULT FALSE,
 sealed BOOLEAN NOT NULL DEFAULT FALSE,
 receipt_id TEXT,
 publication_claim_id TEXT,
 publication_lease_expires_at TIMESTAMPTZ,
 aggregate_base_manifest_revision BIGINT NOT NULL,
 base_aggregate_layers JSONB NOT NULL,
 base_selected_unit_receipts JSONB NOT NULL,
 retired_receipt_ids JSONB NOT NULL,
 UNIQUE(job_id,sequence), UNIQUE(authority_epoch,fencing_token)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_attempt_events (
 id BIGSERIAL PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES catalog.knowledge_attempts,
 event TEXT NOT NULL, detail JSONB NOT NULL, at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_receipts (
 receipt_id TEXT PRIMARY KEY, submission_id TEXT NOT NULL UNIQUE,
 attempt_id TEXT NOT NULL REFERENCES catalog.knowledge_attempts,
 receipt JSONB NOT NULL, digest TEXT NOT NULL,
 registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS knowledge_receipts_namespace ON catalog.knowledge_receipts ((receipt->>'namespace'));
CREATE TABLE IF NOT EXISTS catalog.knowledge_output_tombstones (
 namespace TEXT NOT NULL REFERENCES catalog.knowledge_attempts(namespace),
 role TEXT NOT NULL, output_id TEXT NOT NULL,
 PRIMARY KEY(namespace,role,output_id)
);
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS derivation_id TEXT UNIQUE;
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS unit_id TEXT REFERENCES catalog.units;
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS consumed_input_revision TEXT REFERENCES catalog.input_manifests;
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS recipe_digest TEXT REFERENCES catalog.recipes;
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS generation_id TEXT REFERENCES catalog.generations;
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS achieved_tier TEXT;
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS receipt_id TEXT REFERENCES catalog.knowledge_receipts;
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS origin_lineage JSONB NOT NULL DEFAULT '{}';
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS restoration_recipe_digest TEXT REFERENCES catalog.recipes;
ALTER TABLE catalog.derivations ALTER COLUMN key DROP NOT NULL;
-- P5a artifact receipts have no fictitious distillation job or attempt.
CREATE TABLE IF NOT EXISTS catalog.extraction_receipts (
 completion_id TEXT PRIMARY KEY,
 receipt_id TEXT NOT NULL UNIQUE,
 unit_id TEXT NOT NULL REFERENCES catalog.units,
 generation_id TEXT NOT NULL UNIQUE,
 receipt JSONB NOT NULL,
 digest TEXT NOT NULL,
 registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS extraction_receipt_id TEXT REFERENCES catalog.extraction_receipts(receipt_id);
ALTER TABLE catalog.derivations ADD COLUMN IF NOT EXISTS extraction_generation_id TEXT REFERENCES catalog.extraction_receipts(generation_id);
CREATE UNIQUE INDEX IF NOT EXISTS derivations_extraction_receipt ON catalog.derivations(extraction_receipt_id)
 WHERE extraction_receipt_id IS NOT NULL;
-- A completion fence only: this table confers no knowledge coverage or serving selection.
CREATE TABLE IF NOT EXISTS catalog.extraction_heads (
 unit_id TEXT PRIMARY KEY REFERENCES catalog.units,
 completion_id TEXT NOT NULL REFERENCES catalog.extraction_receipts,
 revision BIGINT NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS catalog.publications (
 publication_id TEXT PRIMARY KEY, derivation_id TEXT NOT NULL REFERENCES catalog.derivations(derivation_id),
 unit_id TEXT NOT NULL REFERENCES catalog.units, generation_id TEXT NOT NULL REFERENCES catalog.generations,
 selected BOOLEAN NOT NULL, published_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS catalog.selections (
 unit_id TEXT NOT NULL REFERENCES catalog.units, generation_id TEXT NOT NULL REFERENCES catalog.generations,
 derivation_id TEXT NOT NULL REFERENCES catalog.derivations(derivation_id),
 revision BIGINT NOT NULL DEFAULT 1,
 PRIMARY KEY(unit_id,generation_id)
);
CREATE TABLE IF NOT EXISTS catalog.supersession_events (
 id BIGSERIAL PRIMARY KEY, derivation_id TEXT NOT NULL REFERENCES catalog.derivations(derivation_id),
 event TEXT NOT NULL CHECK(event IN ('selected','superseded')),
 at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_operations (
 operation_id TEXT PRIMARY KEY, action TEXT NOT NULL, payload_digest TEXT NOT NULL,
 result JSONB NOT NULL, at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_outbox (
 outbox_id TEXT PRIMARY KEY, kind TEXT NOT NULL, submission_id TEXT,
 payload JSONB NOT NULL, acknowledged_at TIMESTAMPTZ,
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS catalog.observations (
 observation_id TEXT PRIMARY KEY,
 unit_id TEXT, consumed_input_revision TEXT, recipe_digest TEXT, generation_id TEXT,
 attempt_id TEXT, submission_id TEXT, authority_epoch BIGINT, fencing_token BIGINT,
 receipt_id TEXT, namespace TEXT,
 role TEXT NOT NULL, store TEXT NOT NULL, output_id TEXT,
 state TEXT NOT NULL CHECK(state IN ('present','missing','unknown','not_applicable')),
 observed_at TIMESTAMPTZ NOT NULL, ingested_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 detail JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS observations_unit ON catalog.observations(unit_id,observed_at);
CREATE INDEX IF NOT EXISTS observations_namespace ON catalog.observations(namespace,observed_at);
CREATE TABLE IF NOT EXISTS catalog.observation_heads (
 identity JSONB PRIMARY KEY,
 observation_id TEXT NOT NULL REFERENCES catalog.observations,
 unit_id TEXT, namespace TEXT, receipt_id TEXT, observed_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS observation_heads_unit ON catalog.observation_heads(unit_id);
CREATE INDEX IF NOT EXISTS observation_heads_namespace ON catalog.observation_heads(namespace);
-- Explicit migration backfill, not a runtime repair/refresh.
INSERT INTO catalog.observation_heads AS h(identity,observation_id,unit_id,namespace,receipt_id,observed_at)
SELECT DISTINCT ON (jsonb_build_array(unit_id,consumed_input_revision,recipe_digest,generation_id,attempt_id,submission_id,
 authority_epoch,fencing_token,receipt_id,namespace,role,store,output_id))
 jsonb_build_array(unit_id,consumed_input_revision,recipe_digest,generation_id,attempt_id,submission_id,
 authority_epoch,fencing_token,receipt_id,namespace,role,store,output_id),observation_id,unit_id,namespace,receipt_id,observed_at
FROM catalog.observations ORDER BY jsonb_build_array(unit_id,consumed_input_revision,recipe_digest,generation_id,attempt_id,submission_id,
 authority_epoch,fencing_token,receipt_id,namespace,role,store,output_id),observed_at DESC,ingested_at DESC,observation_id DESC
ON CONFLICT(identity) DO UPDATE SET observation_id=EXCLUDED.observation_id,unit_id=EXCLUDED.unit_id,
 namespace=EXCLUDED.namespace,receipt_id=EXCLUDED.receipt_id,observed_at=EXCLUDED.observed_at
WHERE EXCLUDED.observed_at>h.observed_at;
CREATE TABLE IF NOT EXISTS catalog.knowledge_repairs (
 repair_id TEXT PRIMARY KEY, publication_id TEXT NOT NULL REFERENCES catalog.publications,
 role TEXT NOT NULL, reason TEXT NOT NULL,
 target_generation TEXT REFERENCES catalog.generations,
 linked_at TIMESTAMPTZ,
 revision BIGINT NOT NULL DEFAULT 1,
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(), UNIQUE(publication_id,role,reason)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_discrepancies (
 discrepancy_id TEXT PRIMARY KEY, unit_id TEXT,
 reason TEXT NOT NULL, evidence JSONB NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_pending_inputs (
 record_key TEXT NOT NULL REFERENCES catalog.records,
 record_revision BIGINT NOT NULL, payload_digest TEXT NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(), PRIMARY KEY(record_key,record_revision)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_unit_reducers (
 unit_id TEXT NOT NULL REFERENCES catalog.units,
 consumed_input_revision TEXT NOT NULL REFERENCES catalog.input_manifests,
 failure_streak INTEGER NOT NULL DEFAULT 0 CHECK(failure_streak>=0),
 parked BOOLEAN NOT NULL DEFAULT FALSE,
 terminal_outcome TEXT,
 legacy_checkpoint_id TEXT UNIQUE,
 revision BIGINT NOT NULL DEFAULT 1,
 PRIMARY KEY(unit_id,consumed_input_revision)
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_legacy_checkpoint_applications (
 unit_id TEXT PRIMARY KEY REFERENCES catalog.units,
 checkpoint_id TEXT NOT NULL UNIQUE,
 consumed_input_revision TEXT NOT NULL REFERENCES catalog.input_manifests,
 evidence JSONB NOT NULL,
 applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS catalog.knowledge_pins (
 pin_id TEXT PRIMARY KEY,
 kind TEXT NOT NULL CHECK(kind IN ('job','staging','recovery','reader')),
 owner TEXT NOT NULL,
 authority_epoch BIGINT NOT NULL,
 namespaces JSONB NOT NULL,
 generation_id TEXT REFERENCES catalog.generations,
 manifest_revision BIGINT,
 expires_at TIMESTAMPTZ,
 released_at TIMESTAMPTZ,
 revision BIGINT NOT NULL DEFAULT 1,
 detail JSONB NOT NULL DEFAULT '{}'
);
DO $$
DECLARE name TEXT;
BEGIN
 FOREACH name IN ARRAY ARRAY['unit_mappings','section_mappings','unit_inventories','input_manifests','recipes',
  'generation_manifests','knowledge_receipts','extraction_receipts','derivations','publications','supersession_events',
  'knowledge_attempt_events','knowledge_operations','observations','knowledge_pending_inputs','knowledge_output_tombstones',
  'knowledge_discrepancies','knowledge_maintenance_windows','knowledge_legacy_checkpoint_applications']
 LOOP
  IF NOT EXISTS (SELECT FROM pg_trigger WHERE tgrelid=('catalog.'||name)::regclass
                 AND tgname='knowledge_append_only') THEN
   EXECUTE format('CREATE TRIGGER knowledge_append_only BEFORE UPDATE OR DELETE OR TRUNCATE ON catalog.%I '
                  'FOR EACH STATEMENT EXECUTE FUNCTION catalog.reject_history_change()',name);
  END IF;
 END LOOP;
END $$;

CREATE OR REPLACE FUNCTION catalog.guard_attempt_identity() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE mutable_columns TEXT[] := ARRAY['lease_expires_at','finished_at','outcome','error','actual_cost',
 'cost_state','accepted','sealed','receipt_id','publication_claim_id','publication_lease_expires_at'];
BEGIN
 IF (to_jsonb(OLD)-mutable_columns) IS DISTINCT FROM (to_jsonb(NEW)-mutable_columns)
    OR (OLD.sealed AND NOT NEW.sealed) OR (OLD.accepted AND NOT NEW.accepted)
    OR (OLD.receipt_id IS NOT NULL AND NEW.receipt_id IS DISTINCT FROM OLD.receipt_id)
    OR (OLD.outcome IS NOT NULL AND NEW.outcome IS DISTINCT FROM OLD.outcome) THEN
  RAISE EXCEPTION 'immutable attempt identity or terminal history cannot be changed';
 END IF;
 RETURN NEW;
END $$;
CREATE OR REPLACE FUNCTION catalog.guard_tombstone_insert() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM catalog.knowledge_attempts a JOIN catalog.jobs j ON j.current_attempt_id=a.attempt_id
   CROSS JOIN catalog.knowledge_runtime r WHERE a.namespace=NEW.namespace AND NOT a.sealed
   AND a.outcome IS NULL AND j.state='running' AND a.lease_expires_at>clock_timestamp()
   AND a.authority_epoch=r.authority_epoch AND r.authority='catalog') THEN
  RAISE EXCEPTION 'tombstone namespace sealed or execution claim invalid';
 END IF;
 RETURN NEW;
END $$;
DO $$ BEGIN
 IF NOT EXISTS(SELECT FROM pg_trigger WHERE tgrelid='catalog.knowledge_attempts'::regclass AND tgname='attempt_identity_guard') THEN
  CREATE TRIGGER attempt_identity_guard BEFORE UPDATE ON catalog.knowledge_attempts
   FOR EACH ROW EXECUTE FUNCTION catalog.guard_attempt_identity();
  CREATE TRIGGER attempt_delete_guard BEFORE DELETE OR TRUNCATE ON catalog.knowledge_attempts
   FOR EACH STATEMENT EXECUTE FUNCTION catalog.reject_history_change();
 END IF;
 IF NOT EXISTS(SELECT FROM pg_trigger WHERE tgrelid='catalog.knowledge_output_tombstones'::regclass AND tgname='tombstone_insert_guard') THEN
  CREATE TRIGGER tombstone_insert_guard BEFORE INSERT ON catalog.knowledge_output_tombstones
   FOR EACH ROW EXECUTE FUNCTION catalog.guard_tombstone_insert();
 END IF;
END $$;

CREATE OR REPLACE FUNCTION catalog.guard_unit_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF ROW(OLD.unit_id,OLD.kind,OLD.record_key,OLD.identity) IS DISTINCT FROM
    ROW(NEW.unit_id,NEW.kind,NEW.record_key,NEW.identity) THEN
  RAISE EXCEPTION 'typed unit identity is immutable';
 END IF;
 RETURN NEW;
END $$;
DO $$ BEGIN
 IF NOT EXISTS(SELECT FROM pg_trigger WHERE tgrelid='catalog.units'::regclass AND tgname='unit_identity_guard') THEN
  CREATE TRIGGER unit_identity_guard BEFORE UPDATE ON catalog.units FOR EACH ROW EXECUTE FUNCTION catalog.guard_unit_identity();
  CREATE TRIGGER unit_delete_guard BEFORE DELETE OR TRUNCATE ON catalog.units FOR EACH STATEMENT EXECUTE FUNCTION catalog.reject_history_change();
 END IF;
END $$;

CREATE OR REPLACE FUNCTION catalog.guard_generation_retrieval_contract() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF OLD.retrieval_contract IS NOT NULL AND NEW.retrieval_contract IS DISTINCT FROM OLD.retrieval_contract THEN
  RAISE EXCEPTION 'generation retrieval contract is immutable once bound';
 END IF;
 RETURN NEW;
END $$;
DO $$ BEGIN
 IF NOT EXISTS(SELECT FROM pg_trigger WHERE tgrelid='catalog.generations'::regclass AND tgname='generation_retrieval_contract_guard') THEN
  CREATE TRIGGER generation_retrieval_contract_guard BEFORE UPDATE ON catalog.generations
   FOR EACH ROW EXECUTE FUNCTION catalog.guard_generation_retrieval_contract();
 END IF;
END $$;
