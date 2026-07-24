-- created_at: NOT NULL with a DEFAULT now() backfills every existing row with
-- the migration timestamp.
ALTER TABLE genesis_graph_checkpoint ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT now();

-- is_deleted: NOT NULL with a non-volatile DEFAULT backfills all existing rows to FALSE.
ALTER TABLE genesis_graph_checkpoint ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT FALSE;
