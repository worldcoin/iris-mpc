-- use bytea instead of jsonb --

-- no sense having a graph table without any links. 
TRUNCATE TABLE hawk_graph_links;

ALTER TABLE hawk_graph_links DROP COLUMN links;
ALTER TABLE hawk_graph_links ADD COLUMN links bytea NOT NULL;

ALTER TABLE hawk_graph_entry DROP COLUMN entry_point;
ALTER TABLE hawk_graph_entry ADD COLUMN entry_point bytea;
