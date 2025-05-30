DROP TABLE IF EXISTS hawk_graph_links;
DROP TABLE IF EXISTS hawk_graph_entry;

-- Links table --

CREATE TABLE hawk_graph_links
(
    graph_id smallint NOT NULL CHECK (graph_id = 0 OR graph_id = 1),
    serial_id bigint NOT NULL CHECK (serial_id BETWEEN 1 AND 4294967296),
    version_id smallint NOT NULL CHECK (version_id >= 0),
    layer smallint NOT NULL CHECK (layer >= 0),
    links bytea NOT NULL,
    CONSTRAINT hawk_graph_links_pkey PRIMARY KEY (graph_id, serial_id, version_id, layer)
);

-- Entry point table --

CREATE TABLE hawk_graph_entry
(
    graph_id smallint NOT NULL CHECK (graph_id = 0 OR graph_id = 1),
    serial_id bigint NOT NULL CHECK (serial_id BETWEEN 1 AND 4294967296),
    version_id smallint NOT NULL CHECK (version_id >= 0),
    layer smallint NOT NULL CHECK (layer >= 0),
    CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (graph_id)
);
