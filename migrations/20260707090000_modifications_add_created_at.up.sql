-- POP-3931: retention for the append-only `modifications` table needs an age clock.
-- ADD COLUMN with a non-volatile DEFAULT is metadata-only on PG11+ (no table rewrite;
-- now() is STABLE — proven empirically on PG16.14 for the identical anon_stats
-- migration, ampc-common #121: relfilenode unchanged). Existing rows are backfilled
-- with the migration timestamp, which only makes retention MORE conservative: nothing
-- becomes deletable until 30 days after this migration lands.
--
-- Deliberately does NOT touch the PK/UNIQUE(id) — `id` is a globally-unique
-- cross-party key (sync grouping key, ON CONFLICT target); see POP-3931 for why
-- partitioning on created_at is ruled out.
ALTER TABLE modifications ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT now();
CREATE INDEX IF NOT EXISTS idx_modifications_created_at ON modifications (created_at);
