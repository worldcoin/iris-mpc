CREATE TABLE IF NOT EXISTS hawk_graph_links (
    source_ref text NOT NULL,
    layer integer NOT NULL,
    links jsonb NOT NULL,
    CONSTRAINT hawk_graph_pkey PRIMARY KEY (source_ref, layer)
);

CREATE TABLE IF NOT EXISTS hawk_graph_entry (
    entry_point jsonb,
    id integer NOT NULL,
    CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (id)
);
