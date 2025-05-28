DROP TRIGGER IF EXISTS graph_links_set_last_modified_at ON hawk_graph_links;
DROP TRIGGER IF EXISTS graph_entry_set_last_modified_at ON hawk_graph_entry;

ALTER TABLE hawk_graph_links DROP COLUMN last_modified_at;
ALTER TABLE hawk_graph_entry DROP COLUMN last_modified_at;
