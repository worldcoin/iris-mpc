INSERT INTO persistent_state (domain, "key", "value")
SELECT 'db_ingest', 'completed_sequence_number',
       COALESCE(to_jsonb(MAX(sequence_number)), 'null'::jsonb)
FROM ingested_requests
WHERE persisted_at IS NOT NULL
ON CONFLICT (domain, "key") DO NOTHING;
