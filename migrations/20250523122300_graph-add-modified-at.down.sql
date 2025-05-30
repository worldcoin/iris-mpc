DROP TRIGGER IF EXISTS graph_links_set_last_modified_at ON hawk_graph_links;

ALTER TABLE hawk_graph_links DROP COLUMN last_modified_at;
