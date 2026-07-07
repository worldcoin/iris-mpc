-- no-transaction
-- POP-3931: retention index, built CONCURRENTLY so writers (including the EXCLUSIVE-lock
-- id-assignment trigger) are never blocked behind the build on this hot table.
-- Caveat of CONCURRENTLY: if a build is interrupted it can leave an INVALID index that
-- IF NOT EXISTS would then skip — if this migration ever fails mid-flight, drop the
-- invalid index (`DROP INDEX idx_modifications_created_at`) before retrying.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_modifications_created_at
    ON modifications (created_at);
