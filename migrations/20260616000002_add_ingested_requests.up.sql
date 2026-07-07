CREATE TABLE ingested_requests (
    sequence_number TEXT PRIMARY KEY,
    message_body TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    consumed_batch_id BIGINT NULL,
    persisted_at TIMESTAMPTZ NULL,
    CONSTRAINT ingested_requests_sequence_number_width CHECK (char_length(sequence_number) = 40)
);

CREATE INDEX ingested_requests_pending_sequence_number_idx
    ON ingested_requests (sequence_number)
    WHERE consumed_batch_id IS NULL;
