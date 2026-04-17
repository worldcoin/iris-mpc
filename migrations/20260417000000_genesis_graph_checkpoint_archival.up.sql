ALTER TABLE genesis_graph_checkpoint
    ADD COLUMN is_archival BOOLEAN NOT NULL DEFAULT FALSE;
