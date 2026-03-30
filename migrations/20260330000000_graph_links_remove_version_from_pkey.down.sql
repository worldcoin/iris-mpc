-- Restore version_id to primary key
ALTER TABLE hawk_graph_links DROP CONSTRAINT hawk_graph_links_pkey;
ALTER TABLE hawk_graph_links ADD CONSTRAINT hawk_graph_links_pkey PRIMARY KEY (graph_id, serial_id, version_id, layer);
