-- Adds an `is_valid` flag to `ingested_requests`.
ALTER TABLE ingested_requests ADD COLUMN is_valid BOOLEAN DEFAULT true;
