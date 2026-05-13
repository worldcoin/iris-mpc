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

ALTER TABLE hawk_graph_mutations
    DROP COLUMN IF EXISTS mutation_version;
