-- Add version_id column with a default value of 0
ALTER TABLE irises ADD COLUMN IF NOT EXISTS version_id SMALLINT DEFAULT 0 CHECK (version_id >= 0);

-- Create a function that will be executed by the trigger
CREATE OR REPLACE FUNCTION increment_version_id()
RETURNS TRIGGER AS $$
BEGIN
    -- Only increment version_id if actual data columns changed
    IF (OLD.left_code IS DISTINCT FROM NEW.left_code OR
        OLD.left_mask IS DISTINCT FROM NEW.left_mask OR
        OLD.right_code IS DISTINCT FROM NEW.right_code OR
        OLD.right_mask IS DISTINCT FROM NEW.right_mask) THEN
        NEW.version_id = COALESCE(OLD.version_id, 0) + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger that calls the function before updates
CREATE TRIGGER increment_version_id_trigger
    BEFORE UPDATE ON irises
    FOR EACH ROW
    EXECUTE FUNCTION increment_version_id();
