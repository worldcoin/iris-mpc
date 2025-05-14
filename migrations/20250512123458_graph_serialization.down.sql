-- Revert changes to use jsonb instead of bytea

TRUNCATE TABLE hawk_graph_links;
ALTER TABLE hawk_graph_links DROP COLUMN links;
ALTER TABLE hawk_graph_links ADD COLUMN links jsonb NOT NULL;

ALTER TABLE hawk_graph_entry DROP COLUMN entry_point;
ALTER TABLE hawk_graph_entry ADD COLUMN entry_point jsonb;
