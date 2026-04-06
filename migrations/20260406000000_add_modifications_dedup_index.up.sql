CREATE INDEX idx_modifications_dedup
ON modifications (serial_id, id DESC)
WHERE persisted = true AND status = 'COMPLETED';
