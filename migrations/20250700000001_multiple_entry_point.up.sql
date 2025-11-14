--- need the graph entry table to have multiple entry points.
ALTER TABLE hawk_graph_entry DROP CONSTRAINT hawk_graph_entry_pkey;
ALTER TABLE hawk_graph_entry ADD CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (graph_id, serial_id, layer);