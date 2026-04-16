ALTER TABLE irises ADD COLUMN IF NOT EXISTS rerand_epoch INTEGER NOT NULL DEFAULT 0;

CREATE OR REPLACE FUNCTION increment_version_id()
RETURNS TRIGGER AS $$
BEGIN
    IF (OLD.left_code IS DISTINCT FROM NEW.left_code OR
        OLD.left_mask IS DISTINCT FROM NEW.left_mask OR
        OLD.right_code IS DISTINCT FROM NEW.right_code OR
        OLD.right_mask IS DISTINCT FROM NEW.right_mask)
       AND NEW.rerand_epoch IS NOT DISTINCT FROM OLD.rerand_epoch THEN
        NEW.version_id = COALESCE(OLD.version_id, 0) + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
