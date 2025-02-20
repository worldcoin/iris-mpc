-- Down migration to revert changes made in the up migration.

-- Entry point table --

-- Remove the primary key constraint on graph_id in hawk_graph_entry
ALTER TABLE hawk_graph_entry
DROP CONSTRAINT hawk_graph_entry_pkey;

-- Re-add the id column.
ALTER TABLE hawk_graph_entry
ADD COLUMN id integer;

-- Backfill the unique id 0.
UPDATE hawk_graph_entry
SET id = 0;

-- Reinstate id as primary key.
ALTER TABLE hawk_graph_entry
ALTER COLUMN id SET NOT NULL;

ALTER TABLE hawk_graph_entry
ADD CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (id);

-- Drop the graph_id column.
ALTER TABLE hawk_graph_entry
DROP COLUMN graph_id;


-- Links table --

-- Remove the primary key constraint on graph_id, source_ref, layer from hawk_graph_links
ALTER TABLE hawk_graph_links
DROP CONSTRAINT hawk_graph_links_pkey;

-- Re-add the original primary key constraint (source_ref, layer) on hawk_graph_links
ALTER TABLE hawk_graph_links
ADD CONSTRAINT hawk_graph_pkey PRIMARY KEY (source_ref, layer);

-- Drop the graph_id column.
ALTER TABLE hawk_graph_links
DROP COLUMN graph_id;
