-- all results (including uniqueness) will be replayed by modifications table --
DROP TABLE IF EXISTS results;

-- allow in progress uniqueness modifications to be inserted. they are only assigned a serial id if result is unique --
ALTER TABLE modifications ALTER COLUMN serial_id DROP NOT NULL;
