DO $$
DECLARE
    staging_schema TEXT;
BEGIN
    staging_schema := current_schema() || '_rerand_staging';
    EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE', staging_schema);
END $$;
