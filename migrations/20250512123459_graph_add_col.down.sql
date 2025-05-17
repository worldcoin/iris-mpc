-- this assumes that the add_col.up and swap_col.up migrations were ran, followed by 
-- the swap_col.down migration, followed by a custom converter to convert from bytea to jsonb

ALTER TABLE hawk_graph_links DROP COLUMN links;
ALTER TABLE hawk_graph_links RENAME COLUMN links_j TO links;

ALTER TABLE hawk_graph_entry DROP COLUMN entry_point;
ALTER TABLE hawk_graph_entry RENAME COLUMN entry_point_j TO entry_point;
