SELECT pg_catalog.set_config(
    'search_path',
    pg_catalog.quote_ident(pg_catalog.current_schema()) || ',pg_catalog,pg_temp',
    true
);

LOCK TABLE irises IN ACCESS EXCLUSIVE MODE;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM rerand_control WHERE active_epoch IS NOT NULL)
       OR EXISTS (SELECT 1 FROM irises WHERE rerand_epoch <> 0) THEN
        RAISE EXCEPTION 'normalize all rows to epoch 0 and stop the sweeper before rollback';
    END IF;
END;
$$;

DO $$
DECLARE
    role_name NAME;
    writer_had_schema_usage BOOLEAN;
    writer_had_iris_select BOOLEAN;
BEGIN
    SELECT pg_catalog.pg_get_userbyid(control.writer_role),
           control.writer_had_schema_usage,
           control.writer_had_iris_select
      INTO role_name, writer_had_schema_usage, writer_had_iris_select
      FROM rerand_control control
     WHERE control.singleton AND control.writer_role IS NOT NULL;
    IF role_name IS NOT NULL THEN
        IF writer_had_iris_select IS FALSE THEN
            EXECUTE pg_catalog.format('REVOKE SELECT ON irises FROM %I', role_name);
        END IF;
        IF writer_had_schema_usage IS FALSE THEN
            EXECUTE pg_catalog.format(
                'REVOKE USAGE ON SCHEMA %I FROM %I',
                pg_catalog.current_schema(),
                role_name
            );
        END IF;
    END IF;
END;
$$;

DROP TRIGGER protect_iris_truncate_trigger ON irises;
DROP FUNCTION protect_iris_truncate();
DROP TRIGGER update_iris_metadata_trigger ON irises;
DROP FUNCTION update_iris_metadata();
DROP TRIGGER protect_rerand_control_trigger ON rerand_control;
DROP FUNCTION protect_rerand_control();
DROP FUNCTION begin_rerand_pass(INTEGER, BIGINT, BYTEA);
DROP FUNCTION advance_rerand_pass(INTEGER, BIGINT);
DROP FUNCTION complete_rerand_pass(INTEGER, INTEGER);
DROP FUNCTION apply_rerand_updates(
    BIGINT[], SMALLINT[], TEXT[], INTEGER[], BYTEA[], BYTEA[], BYTEA[], BYTEA[], INTEGER
);
DROP FUNCTION get_rerand_store_state();
DROP FUNCTION try_rerand_pass_lock();
DROP FUNCTION unlock_rerand_pass_lock();
DROP FUNCTION rerand_pass_lock_held();
DROP FUNCTION initialize_rerand_store(TEXT, TEXT, TEXT, SMALLINT, TEXT, NAME);
DROP TABLE rerand_control;

CREATE FUNCTION update_last_modified_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified_at = EXTRACT(EPOCH FROM NOW())::BIGINT;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_last_modified_at
    BEFORE INSERT OR UPDATE ON irises
    FOR EACH ROW EXECUTE FUNCTION update_last_modified_at();

CREATE FUNCTION increment_version_id()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.left_code IS DISTINCT FROM NEW.left_code OR
       OLD.left_mask IS DISTINCT FROM NEW.left_mask OR
       OLD.right_code IS DISTINCT FROM NEW.right_code OR
       OLD.right_mask IS DISTINCT FROM NEW.right_mask THEN
        NEW.version_id = COALESCE(OLD.version_id, 0) + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER increment_version_id_trigger
    BEFORE UPDATE ON irises
    FOR EACH ROW EXECUTE FUNCTION increment_version_id();

ALTER TABLE irises DROP COLUMN rerand_epoch, DROP COLUMN semantic_id;
