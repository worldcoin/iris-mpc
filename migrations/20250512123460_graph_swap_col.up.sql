-- Assumes a custom program has been run to convert the data from the jsonb column to the 
-- bytea column. now the jsonb column will be dropped and the bytea column will be renamed

ALTER TABLE hawk_graph_links DROP COLUMN links;
ALTER TABLE hawk_graph_links RENAME COLUMN links_b TO links;

ALTER TABLE hawk_graph_entry DROP COLUMN entry_point;
ALTER TABLE hawk_graph_entry RENAME COLUMN entry_point_b TO entry_point;
