-- Derived, versioned shares for the opt-in exact CPU NTT scanner. Keep original
-- iris shares authoritative so this cache can be rebuilt by all three parties.
CREATE TABLE cpu_spectral_irises (
    serial_id BIGINT NOT NULL,
    version_id SMALLINT NOT NULL,
    side SMALLINT NOT NULL CHECK (side IN (0, 1)),
    party SMALLINT NOT NULL CHECK (party IN (0, 1, 2)),
    format_version SMALLINT NOT NULL,
    generation BYTEA NOT NULL CHECK (octet_length(generation) = 16),
    payload BYTEA NOT NULL CHECK (octet_length(payload) = 38400),
    payload_hash BYTEA NOT NULL CHECK (octet_length(payload_hash) = 32),
    PRIMARY KEY (serial_id, side)
);
-- Random shares do not compress; skip repeated TOAST compression attempts.
ALTER TABLE cpu_spectral_irises ALTER COLUMN payload SET STORAGE EXTERNAL;
