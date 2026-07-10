-- An explicitly changed version_id takes precedence; otherwise content changes auto-increment it.
CREATE OR REPLACE FUNCTION increment_version_id()
RETURNS TRIGGER AS $$
BEGIN
    IF (NEW.version_id IS NOT DISTINCT FROM OLD.version_id) AND
       (OLD.left_code IS DISTINCT FROM NEW.left_code OR
        OLD.left_mask IS DISTINCT FROM NEW.left_mask OR
        OLD.right_code IS DISTINCT FROM NEW.right_code OR
        OLD.right_mask IS DISTINCT FROM NEW.right_mask) THEN
        NEW.version_id = COALESCE(OLD.version_id, 0) + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
