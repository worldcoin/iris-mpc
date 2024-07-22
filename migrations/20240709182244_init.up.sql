CREATE TABLE IF NOT EXISTS irises (
    id BIGSERIAL PRIMARY KEY,
    left_code BYTEA,
    left_mask BYTEA,
    right_code BYTEA,
    right_mask BYTEA
);
