-- Add a reference to hawk_graph_mutations which stores BothEyes<Vec<GraphMutation>>.
-- Defaults to 0 (no mutation recorded). The existing graph_mutation column is preserved
-- so behavior does not change until migration to id-based lookup is complete.
ALTER TABLE modifications
    ADD COLUMN graph_mutation_id BIGINT NOT NULL DEFAULT 0;
