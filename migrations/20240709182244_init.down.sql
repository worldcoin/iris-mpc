-- Delete the table if it is empty.
DO $$
BEGIN
    IF (SELECT count(*) FROM irises) = 0 THEN
        EXECUTE 'DROP TABLE irises';
    END IF;
END $$;
