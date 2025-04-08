ALTER TABLE irises ADD COLUMN last_modified_at BIGINT;

CREATE OR REPLACE FUNCTION update_last_modified_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified_at = EXTRACT(EPOCH FROM NOW())::BIGINT;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_last_modified_at
    BEFORE INSERT OR UPDATE ON irises
    FOR EACH ROW
    EXECUTE FUNCTION update_last_modified_at();
