DROP TRIGGER IF EXISTS increment_version_id_trigger ON irises;
DROP FUNCTION IF EXISTS increment_version_id();
ALTER TABLE irises DROP COLUMN version_id;
