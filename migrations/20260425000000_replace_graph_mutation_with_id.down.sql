ALTER TABLE modifications
    DROP COLUMN graph_mutation_id;

ALTER TABLE modifications
    ADD COLUMN graph_mutation BYTEA;
