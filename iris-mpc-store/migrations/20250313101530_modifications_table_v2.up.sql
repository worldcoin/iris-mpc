-- drop the old modifications table if it exists
DROP TABLE IF EXISTS modifications;

-- create the new modifications table
CREATE TABLE IF NOT EXISTS modifications (
    id TEXT PRIMARY KEY,
    serial_id BIGINT NOT NULL,
    request_type TEXT NOT NULL,
    s3_url TEXT,
    status TEXT NOT NULL,
    persisted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW()) * 1000000)::BIGINT -- microseconds precision
);

-- create the index on created_at column to get the latest modifications faster
CREATE INDEX IF NOT EXISTS idx_modifications_created_at ON modifications(created_at DESC);
