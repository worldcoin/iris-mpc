-- Replace hawk_graph_mutations.id (GENERATED ALWAYS AS IDENTITY) with modification_id as the
-- natural PRIMARY KEY.
--
-- Rationale: the previous schema used a sequence-based id which drifts across parties on
-- rollback/retry. modification_id is assigned via the MAX(id)+1 trigger on the modifications
-- table, so it converges to the same value on every party regardless of retries.
-- assert_modifications_consistency (sync.rs) previously compared graph_mutation_id cross-party,
-- which would spuriously fail when sequences diverged. Using modification_id as PK eliminates
-- the drift without any loss of information.

-- Drop the now-redundant graph_mutation_id column from modifications.
-- It stored the drifting hawk_graph_mutations.id; lookups now go via modification_id directly.
ALTER TABLE modifications
    DROP COLUMN IF EXISTS graph_mutation_id;

-- Recreate hawk_graph_mutations with modification_id as the sole primary key.
-- Any existing rows are discarded; this migration is safe to apply on a fresh deployment or
-- after a coordinated shutdown because the WAL is replayed from the last checkpoint anyway.
DROP TABLE IF EXISTS hawk_graph_mutations;

CREATE TABLE IF NOT EXISTS hawk_graph_mutations (
    modification_id BIGINT PRIMARY KEY,
    -- BothEyes<Vec<GraphMutation>>
    serialized_mutations BYTEA NOT NULL,
    mutation_version INT NOT NULL DEFAULT 1
);
