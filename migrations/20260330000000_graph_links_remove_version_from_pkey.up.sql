-- Remove version_id from primary key so updates overwrite instead of creating new rows

-- First, deduplicate: keep only the row with the highest version_id for each (graph_id, serial_id, layer)
DELETE FROM hawk_graph_links a
USING hawk_graph_links b
WHERE a.graph_id = b.graph_id
  AND a.serial_id = b.serial_id
  AND a.layer = b.layer
  AND a.version_id < b.version_id;

-- Now change the primary key
ALTER TABLE hawk_graph_links DROP CONSTRAINT hawk_graph_links_pkey;
ALTER TABLE hawk_graph_links ADD CONSTRAINT hawk_graph_links_pkey PRIMARY KEY (graph_id, serial_id, layer);
