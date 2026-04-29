-- Add a reference to hawk_graph_mutations which stores BothEyes<Vec<GraphMutation>>.
-- NULL = no mutation recorded. The existing graph_mutation column is preserved
-- so behavior does not change until migration to id-based lookup is complete.
--
-- No foreign key constraint is added because modifications may have been deleted
-- but their associated graph mutations are retained in the hawk_graph_mutations table.
-- Application code must handle the case where graph_mutation_id references a deleted modification.
ALTER TABLE modifications
    ADD COLUMN graph_mutation_id BIGINT;
