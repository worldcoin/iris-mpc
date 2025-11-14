--- allow only one entry point per graph
ALTER TABLE hawk_graph_entry DROP CONSTRAINT hawk_graph_entry_pkey;
ALTER TABLE hawk_graph_entry ADD CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (graph_id);