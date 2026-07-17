-- Durable ordered inbox for the fixed party-0 coordinator.
--
-- This table is intentionally small: request envelopes contain S3 references,
-- not iris shares. Keeping it in Postgres avoids relying on ephemeral pod disks
-- while providing one FIFO sequence for both legacy SQS and the HTTP API.
CREATE TABLE IF NOT EXISTS coordinator_requests (
    sequence_number BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    message_body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'preparing', 'processing', 'completed', 'rejected', 'failed')),
    result_body TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_coordinator_requests_pending
    ON coordinator_requests (sequence_number)
    WHERE status = 'pending';
