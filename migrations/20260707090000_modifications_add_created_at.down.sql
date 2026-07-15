DROP INDEX IF EXISTS idx_modifications_created_at;
ALTER TABLE modifications DROP COLUMN IF EXISTS created_at;
