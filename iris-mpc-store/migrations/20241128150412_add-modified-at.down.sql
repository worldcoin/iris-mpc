DROP TRIGGER IF EXISTS set_last_modified_at ON mpc_store;
DROP FUNCTION IF EXISTS update_last_modified_at();
ALTER TABLE mpc_store DROP COLUMN last_modified_at;
