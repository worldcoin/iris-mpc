-- Add a graph_id field to support separate left and right graphs.

-- Links table (after) --

-- CREATE TABLE hawk_graph_links
-- (
--     source_ref text COLLATE pg_catalog."default" NOT NULL,
--     layer integer NOT NULL,
--     links jsonb NOT NULL,
--     graph_id integer NOT NULL,
--     CONSTRAINT hawk_graph_links_pkey PRIMARY KEY (graph_id, source_ref, layer)
-- )

-- Add graph_id column to hawk_graph_links.
ALTER TABLE hawk_graph_links
ADD COLUMN graph_id integer;

-- Backfill the graph_id column.
UPDATE hawk_graph_links
SET graph_id = 0
WHERE graph_id IS NULL;

-- Make graph_id mandatory.
ALTER TABLE hawk_graph_links
ALTER COLUMN graph_id SET NOT NULL;

-- Drop the existing primary key constraint on hawk_graph_links
ALTER TABLE hawk_graph_links
DROP CONSTRAINT hawk_graph_pkey;

-- Add new primary key constraint on hawk_graph_links
ALTER TABLE hawk_graph_links
ADD CONSTRAINT hawk_graph_links_pkey PRIMARY KEY (graph_id, source_ref, layer);


-- Entry point table (after) --

-- CREATE TABLE hawk_graph_entry
-- (
--     entry_point jsonb,
--     graph_id integer NOT NULL,
--     CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (graph_id)
-- )

-- Add graph_id column to hawk_graph_entry
ALTER TABLE hawk_graph_entry
ADD COLUMN graph_id integer;

-- Backfill the graph_id column.
UPDATE hawk_graph_entry
SET graph_id = 0
WHERE graph_id IS NULL;

-- Make graph_id mandatory.
ALTER TABLE hawk_graph_entry
ALTER COLUMN graph_id SET NOT NULL;

-- Drop the existing primary key constraint on hawk_graph_entry
ALTER TABLE hawk_graph_entry
DROP CONSTRAINT hawk_graph_entry_pkey;

-- Remove the id column (it was the unique 0).
ALTER TABLE hawk_graph_entry
DROP COLUMN id;

-- Add new primary key constraint on hawk_graph_entry
ALTER TABLE hawk_graph_entry
ADD CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (graph_id);

