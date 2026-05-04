-- Consolidated migration combining:
-- - add_graph_version
-- - hawk_graph_mutations
-- - add_graph_mutation_id

-- Add graph_version column and graph_mutation_id column to genesis_graph_checkpoint
ALTER TABLE genesis_graph_checkpoint
    ADD COLUMN graph_version INT NOT NULL DEFAULT 3,
    -- this can be null if the run is from Genesis
    -- this column is needed because the modifications_sync process will delete modifications,
    -- at which point they would not be available to retrieve the corresponding mutation id.
    ADD COLUMN graph_mutation_id BIGINT;

-- Create graph_mutations table to serve as a diff based write ahead log.
-- modification_id may point to a row that has been applied and is deleted.
CREATE TABLE IF NOT EXISTS hawk_graph_mutations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    modification_id BIGINT NOT NULL,
    -- BothEyes<Vec<GraphMutation>>
    serialized_mutations BYTEA NOT NULL,
    mutation_version INT NOT NULL DEFAULT 1
);

-- Add a reference to hawk_graph_mutations which stores BothEyes<Vec<GraphMutation>>.
-- NULL = no mutation recorded. The existing graph_mutation column is preserved
-- so behavior does not change until migration to id-based lookup is complete.
--
-- No foreign key constraint is added because modifications may have been deleted
-- but their associated graph mutations are retained in the hawk_graph_mutations table.
-- Application code must handle the case where graph_mutation_id references a deleted modification.
ALTER TABLE modifications
    ADD COLUMN graph_mutation_id BIGINT;
