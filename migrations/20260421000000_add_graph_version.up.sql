ALTER TABLE genesis_graph_checkpoint
    ADD COLUMN graph_version INT NOT NULL DEFAULT 3;
