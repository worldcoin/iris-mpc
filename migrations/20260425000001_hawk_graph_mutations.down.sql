-- Reverse of consolidated graph mutations migration

ALTER TABLE modifications
    DROP COLUMN graph_mutation_id;

DROP TABLE IF EXISTS hawk_graph_mutations;

ALTER TABLE genesis_graph_checkpoint
    DROP COLUMN graph_version;
