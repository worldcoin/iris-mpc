ALTER TABLE rerand_control
    ADD COLUMN IF NOT EXISTS worker_last_heartbeat TIMESTAMPTZ;
