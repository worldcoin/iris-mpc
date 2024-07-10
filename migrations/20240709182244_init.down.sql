-- Delete the table if it is empty.
DO $$
BEGIN
    IF (SELECT count(*) FROM irises) > 0 THEN
        RAISE EXCEPTION 'Not dropping the table "irises" because it is not empty.';
    ELSE
        EXECUTE 'DROP TABLE irises';
    END IF;
END $$;