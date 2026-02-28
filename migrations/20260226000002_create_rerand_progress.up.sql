CREATE TABLE IF NOT EXISTS rerand_progress (
    epoch           INTEGER NOT NULL,
    chunk_id        INTEGER NOT NULL,
    staging_written BOOLEAN NOT NULL DEFAULT FALSE,
    all_confirmed   BOOLEAN NOT NULL DEFAULT FALSE,
    live_applied    BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (epoch, chunk_id)
);
