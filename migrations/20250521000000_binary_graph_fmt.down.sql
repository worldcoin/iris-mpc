DROP TABLE IF EXISTS hawk_graph_links;
DROP TABLE IF EXISTS hawk_graph_entry;

-- Links table --

CREATE TABLE hawk_graph_links
(
    graph_id integer NOT NULL,
    source_ref text NOT NULL,
    layer integer NOT NULL,
    links jsonb NOT NULL,
    CONSTRAINT hawk_graph_links_pkey PRIMARY KEY (graph_id, source_ref, layer)
);

-- Entry point table --

CREATE TABLE hawk_graph_entry
(
    graph_id integer NOT NULL,
    entry_point jsonb,
    CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (graph_id)
);
