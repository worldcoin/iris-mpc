DELETE FROM persistent_state
WHERE domain = 'db_ingest' AND "key" = 'completed_sequence_number';
