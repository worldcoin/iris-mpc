-- version_id is trigger-managed: content changes auto-increment it. A hand-set
-- version_id is rejected unless the caller opts in for the transaction via
-- `SET LOCAL app.explicit_version_id = 'on'` (the ExplicitVersion store API),
-- in which case the value is written verbatim.
CREATE OR REPLACE FUNCTION increment_version_id()
RETURNS TRIGGER AS $$
BEGIN
    -- Authoritative write: honor version_id verbatim (any value).
    IF current_setting('app.explicit_version_id', true) = 'on' THEN
        RETURN NEW;
    END IF;

    -- Flag off: version_id must not be set by hand.
    IF NEW.version_id IS DISTINCT FROM OLD.version_id THEN
        RAISE EXCEPTION 'version_id changed (% -> %) without the explicit-version flag',
            OLD.version_id, NEW.version_id
            USING HINT = 'SET LOCAL app.explicit_version_id = ''on''; first, or use the ExplicitVersion store API';
    END IF;

    -- Default: auto-increment on content change.
    IF (OLD.left_code IS DISTINCT FROM NEW.left_code OR
        OLD.left_mask IS DISTINCT FROM NEW.left_mask OR
        OLD.right_code IS DISTINCT FROM NEW.right_code OR
        OLD.right_mask IS DISTINCT FROM NEW.right_mask) THEN
        NEW.version_id = COALESCE(OLD.version_id, 0) + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
