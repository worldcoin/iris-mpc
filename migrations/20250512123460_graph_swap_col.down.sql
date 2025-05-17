-- assuming that the user wants to revert from bytesa to jsonb
-- this requires another two step process.

ALTER TABLE hawk_graph_links ADD COLUMN links_j bytea NOT NULL;
ALTER TABLE hawk_graph_entry ADD COLUMN entry_point_j bytea;