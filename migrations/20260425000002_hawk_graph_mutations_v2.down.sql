-- Reverse of hawk_graph_mutationsv2: restore hawk_graph_mutations to sequence-based id schema
-- and re-add modifications.graph_mutation_id.

DROP TABLE IF EXISTS hawk_graph_mutations;

CREATE TABLE IF NOT EXISTS hawk_graph_mutations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    modification_id BIGINT NOT NULL,
    -- BothEyes<Vec<GraphMutation>>
    serialized_mutations BYTEA NOT NULL,
    mutation_version INT NOT NULL DEFAULT 1
);

ALTER TABLE modifications
    ADD COLUMN IF NOT EXISTS graph_mutation_id BIGINT;
