ALTER TABLE modifications
    DROP COLUMN graph_mutation;

-- Single ID referencing hawk_graph_mutations which stores BothEyes<GraphMutation>
ALTER TABLE modifications
    ADD COLUMN graph_mutation_id BIGINT;
