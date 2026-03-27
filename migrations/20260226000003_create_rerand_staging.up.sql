DO $$
DECLARE
    staging_schema TEXT;
BEGIN
    staging_schema := current_schema() || '_rerand_staging';
    EXECUTE format('CREATE SCHEMA IF NOT EXISTS %I', staging_schema);
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I.irises (
        epoch               INTEGER NOT NULL,
        id                  BIGINT NOT NULL,
        chunk_id            INTEGER NOT NULL,
        left_code           BYTEA,
        left_mask           BYTEA,
        right_code          BYTEA,
        right_mask          BYTEA,
        original_version_id SMALLINT,
        rerand_epoch        INTEGER,
        PRIMARY KEY (epoch, id)
    )', staging_schema);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_staging_irises_epoch_chunk ON %I.irises (epoch, chunk_id)',
        staging_schema
    );
END $$;
