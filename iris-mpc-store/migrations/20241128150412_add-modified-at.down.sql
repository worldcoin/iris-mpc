DROP TRIGGER IF EXISTS set_last_modified_at ON irises;
DROP FUNCTION IF EXISTS update_last_modified_at();
ALTER TABLE irises DROP COLUMN last_modified_at;
