CREATE TABLE IF NOT EXISTS anon_stats_2d_lifted (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL,
    bundle BYTEA NOT NULL,
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    origin SMALLINT NOT NULL
);

