ALTER TABLE hawk_graph_links ADD COLUMN last_modified_at BIGINT;
ALTER TABLE hawk_graph_entry ADD COLUMN last_modified_at BIGINT;

CREATE TRIGGER graph_links_set_last_modified_at
    BEFORE INSERT OR UPDATE ON hawk_graph_links
    FOR EACH ROW
    EXECUTE FUNCTION update_last_modified_at();

CREATE TRIGGER graph_entry_set_last_modified_at
    BEFORE INSERT OR UPDATE ON hawk_graph_entry
    FOR EACH ROW
    EXECUTE FUNCTION update_last_modified_at();
