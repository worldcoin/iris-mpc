-- drop the old modifications table if it exists
DROP TABLE IF EXISTS modifications;

-- create the new modifications table
CREATE TABLE IF NOT EXISTS modifications (
    id BIGINT PRIMARY KEY,
    serial_id BIGINT NOT NULL,
    request_type TEXT NOT NULL,
    s3_url TEXT,
    status TEXT NOT NULL,
    persisted BOOLEAN NOT NULL DEFAULT FALSE
);

-- Create a function to assign the next available modification ID
CREATE OR REPLACE FUNCTION assign_modification_id()
RETURNS TRIGGER AS $$
DECLARE
    next_id BIGINT;
BEGIN
    -- Lock the table to prevent race conditions
    LOCK TABLE modifications IN EXCLUSIVE MODE;

    -- Find the next available ID (max + 1 or 1 if table is empty)
    SELECT COALESCE(MAX(id) + 1, 1) INTO next_id FROM modifications;

    -- Assign the ID to the new row
    NEW.id := next_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically assign modification IDs before insert if not specified explicitly
CREATE TRIGGER before_insert_modifications
BEFORE INSERT ON modifications
FOR EACH ROW
WHEN (NEW.id IS NULL)
EXECUTE FUNCTION assign_modification_id();
