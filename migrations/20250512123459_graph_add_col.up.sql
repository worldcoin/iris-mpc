-- need to convert from jsonb to bytea. a two step migration will be used for this, and a 
-- custom converter will be run between migrations.

ALTER TABLE hawk_graph_links ADD COLUMN links_b bytea NOT NULL;
ALTER TABLE hawk_graph_entry ADD COLUMN entry_point_b bytea;
