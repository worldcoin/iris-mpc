-- Self-describing rerandomized rows and the minimal local pass state.

-- Make the intended schema explicit and put the temporary schema last before
-- resolving any relation name in this migration.
SELECT pg_catalog.set_config(
    'search_path',
    pg_catalog.quote_ident(pg_catalog.current_schema()) || ',pg_catalog,pg_temp',
    true
);

ALTER TABLE irises
    ADD COLUMN rerand_epoch INTEGER NOT NULL DEFAULT 0,
    -- Nullable during the first-pass bootstrap so this migration remains a
    -- metadata-only change on the existing large table.
    ADD COLUMN semantic_id UUID;

-- New rows are checked immediately, while existing rows are known to read as
-- the constant fast default. Deferring validation avoids a heap scan while the
-- migration holds ACCESS EXCLUSIVE.
ALTER TABLE irises ADD CONSTRAINT irises_rerand_epoch_nonnegative
    CHECK (rerand_epoch >= 0) NOT VALID;

DROP TRIGGER IF EXISTS increment_version_id_trigger ON irises;
DROP TRIGGER IF EXISTS set_last_modified_at ON irises;
DROP FUNCTION IF EXISTS increment_version_id();
DROP FUNCTION IF EXISTS update_last_modified_at();

CREATE TABLE rerand_control (
    singleton               BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    store_id                TEXT UNIQUE,
    environment             TEXT,
    coordination_id         TEXT,
    party_id                SMALLINT,
    store_kind              TEXT,
    writer_role             OID,
    writer_had_schema_usage BOOLEAN,
    writer_had_iris_select  BOOLEAN,
    last_completed_epoch    INTEGER NOT NULL DEFAULT 0 CHECK (last_completed_epoch >= 0),
    last_seed_commitment    BYTEA,
    active_epoch            INTEGER,
    active_seed_commitment  BYTEA,
    next_id                 BIGINT,
    max_id                  BIGINT,
    CHECK ((store_id IS NULL) = (writer_role IS NULL)
       AND (store_id IS NULL) = (environment IS NULL)
       AND (store_id IS NULL) = (coordination_id IS NULL)
       AND (store_id IS NULL) = (party_id IS NULL)
       AND (store_id IS NULL) = (store_kind IS NULL)
       AND (store_id IS NULL) = (writer_had_schema_usage IS NULL)
       AND (store_id IS NULL) = (writer_had_iris_select IS NULL)),
    CHECK (party_id IS NULL OR party_id BETWEEN 0 AND 2),
    CHECK (store_kind IS NULL OR store_kind IN ('gpu', 'hnsw')),
    CHECK ((last_completed_epoch = 0) = (last_seed_commitment IS NULL)),
    CHECK (last_seed_commitment IS NULL OR octet_length(last_seed_commitment) = 32),
    CHECK (
        (active_epoch IS NULL AND active_seed_commitment IS NULL
         AND next_id IS NULL AND max_id IS NULL)
        OR
        (active_epoch IS NOT NULL
         AND active_epoch = last_completed_epoch + 1
         AND active_seed_commitment IS NOT NULL
         AND octet_length(active_seed_commitment) = 32
         AND max_id IS NOT NULL
         AND max_id BETWEEN 0 AND 9223372036854775806
         AND next_id IS NOT NULL
         AND next_id BETWEEN 1 AND max_id + 1)
    )
);

INSERT INTO rerand_control DEFAULT VALUES;

-- One owner-only, idempotent initialization binds this physical store to a
-- deployment identity and a dedicated LOGIN role. The identity is immutable.
CREATE FUNCTION initialize_rerand_store(
    p_store_id TEXT,
    p_environment TEXT,
    p_coordination_id TEXT,
    p_party_id SMALLINT,
    p_store_kind TEXT,
    p_writer_role NAME
)
RETURNS VOID AS $$
DECLARE
    owner_oid OID;
    schema_name NAME;
    role_oid OID;
    role_is_login BOOLEAN;
    role_is_super BOOLEAN;
    role_bypasses_rls BOOLEAN;
    role_can_create_roles BOOLEAN;
    role_can_create_db BOOLEAN;
    role_replication BOOLEAN;
    role_had_schema_usage BOOLEAN;
    role_had_iris_select BOOLEAN;
BEGIN
    SELECT c.relowner, n.nspname INTO owner_oid, schema_name
      FROM pg_catalog.pg_class c
      JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
     WHERE c.oid = 'irises'::regclass;
    IF session_user::regrole::oid <> owner_oid THEN
        RAISE EXCEPTION 'initialize_rerand_store must be called by the irises owner';
    END IF;
    IF p_store_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$'
       OR p_environment !~ '^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$'
       OR p_coordination_id !~ '^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$'
       OR p_party_id NOT BETWEEN 0 AND 2
       OR p_store_kind NOT IN ('gpu', 'hnsw') THEN
        RAISE EXCEPTION 'invalid rerandomization store identity';
    END IF;

    -- Serialize the identity transition with both the sweeper and legacy
    -- exporter. The transaction-level lock is held through the control-row
    -- update and grants, so initialization cannot race legacy marker
    -- publication after that exporter observed an uninitialized store.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        1381126734,
        (SELECT c.relnamespace::integer
           FROM pg_catalog.pg_class c
          WHERE c.oid = 'irises'::regclass)
    );

    SELECT oid, rolcanlogin, rolsuper, rolbypassrls, rolcreaterole,
           rolcreatedb, rolreplication
      INTO role_oid, role_is_login, role_is_super, role_bypasses_rls,
           role_can_create_roles, role_can_create_db, role_replication
      FROM pg_catalog.pg_roles WHERE rolname = p_writer_role;
    IF role_oid IS NULL OR NOT role_is_login OR role_is_super OR role_bypasses_rls
       OR role_can_create_roles OR role_can_create_db OR role_replication
       OR role_oid = owner_oid
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_auth_members membership
            WHERE membership.member = role_oid
       )
       OR pg_catalog.has_schema_privilege(role_oid, schema_name, 'CREATE')
       OR pg_catalog.has_table_privilege(role_oid, 'irises', 'INSERT')
       OR pg_catalog.has_table_privilege(role_oid, 'irises', 'UPDATE')
       OR pg_catalog.has_table_privilege(role_oid, 'irises', 'DELETE')
       OR pg_catalog.has_table_privilege(role_oid, 'irises', 'TRUNCATE')
       OR pg_catalog.has_table_privilege(role_oid, 'irises', 'TRIGGER') THEN
        RAISE EXCEPTION 'rerandomization writer must be a distinct, non-privileged LOGIN role';
    END IF;

    role_had_schema_usage := pg_catalog.has_schema_privilege(
        role_oid, schema_name, 'USAGE'
    );
    role_had_iris_select := pg_catalog.has_table_privilege(
        role_oid, 'irises', 'SELECT'
    );

    UPDATE rerand_control
       SET store_id = p_store_id,
           environment = p_environment,
           coordination_id = p_coordination_id,
           party_id = p_party_id,
           store_kind = p_store_kind,
           writer_role = role_oid,
           writer_had_schema_usage = role_had_schema_usage,
           writer_had_iris_select = role_had_iris_select
     WHERE singleton AND store_id IS NULL;
    IF NOT FOUND AND NOT EXISTS (
        SELECT 1 FROM rerand_control
         WHERE singleton AND store_id = p_store_id
           AND environment = p_environment AND party_id = p_party_id
           AND coordination_id = p_coordination_id
           AND store_kind = p_store_kind AND writer_role = role_oid
    ) THEN
        RAISE EXCEPTION 'rerandomization store identity is already initialized differently';
    END IF;

    IF NOT role_had_schema_usage THEN
        EXECUTE pg_catalog.format(
            'GRANT USAGE ON SCHEMA %I TO %I', schema_name, p_writer_role
        );
    END IF;
    IF NOT role_had_iris_select THEN
        EXECUTE pg_catalog.format(
            'GRANT SELECT ON TABLE %I.irises TO %I', schema_name, p_writer_role
        );
    END IF;
    EXECUTE pg_catalog.format(
        'GRANT EXECUTE ON FUNCTION %I.apply_rerand_updates(bigint[], smallint[], text[], integer[], bytea[], bytea[], bytea[], bytea[], integer) TO %I',
        schema_name,
        p_writer_role
    );
    EXECUTE pg_catalog.format('GRANT EXECUTE ON FUNCTION %I.try_rerand_pass_lock() TO %I', schema_name, p_writer_role);
    EXECUTE pg_catalog.format('GRANT EXECUTE ON FUNCTION %I.unlock_rerand_pass_lock() TO %I', schema_name, p_writer_role);
    EXECUTE pg_catalog.format('GRANT EXECUTE ON FUNCTION %I.begin_rerand_pass(integer, bigint, bytea) TO %I', schema_name, p_writer_role);
    EXECUTE pg_catalog.format('GRANT EXECUTE ON FUNCTION %I.advance_rerand_pass(integer, bigint) TO %I', schema_name, p_writer_role);
    EXECUTE pg_catalog.format('GRANT EXECUTE ON FUNCTION %I.complete_rerand_pass(integer, integer) TO %I', schema_name, p_writer_role);
END;
$$ LANGUAGE plpgsql;

-- The lock is schema-specific and session-scoped. A pass must keep the same
-- backend connection for control updates and row CAS writes.
CREATE FUNCTION rerand_pass_lock_held()
RETURNS BOOLEAN AS $$
    SELECT EXISTS (
        SELECT 1 FROM pg_catalog.pg_locks l
         WHERE l.locktype = 'advisory'
           AND l.pid = pg_catalog.pg_backend_pid()
           AND l.classid = 1381126734::oid
           AND l.objid = (
               SELECT relnamespace FROM pg_catalog.pg_class
                WHERE oid = 'irises'::regclass
           )
           AND l.objsubid = 2
           AND l.granted
    );
$$ LANGUAGE sql STABLE;

CREATE FUNCTION try_rerand_pass_lock()
RETURNS BOOLEAN AS $$
DECLARE
    configured_writer OID;
BEGIN
    SELECT writer_role INTO configured_writer FROM rerand_control WHERE singleton;
    IF configured_writer IS NULL OR session_user::regrole::oid <> configured_writer THEN
        RAISE EXCEPTION 'rerandomization lock requires the configured writer role';
    END IF;
    RETURN pg_catalog.pg_try_advisory_lock(
        1381126734,
        (SELECT relnamespace::integer FROM pg_catalog.pg_class
          WHERE oid = 'irises'::regclass)
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE FUNCTION unlock_rerand_pass_lock()
RETURNS BOOLEAN AS $$
DECLARE
    configured_writer OID;
BEGIN
    SELECT writer_role INTO configured_writer FROM rerand_control WHERE singleton;
    IF configured_writer IS NULL OR session_user::regrole::oid <> configured_writer THEN
        RAISE EXCEPTION 'rerandomization unlock requires the configured writer role';
    END IF;
    RETURN pg_catalog.pg_advisory_unlock(
        1381126734,
        (SELECT relnamespace::integer FROM pg_catalog.pg_class
          WHERE oid = 'irises'::regclass)
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE FUNCTION begin_rerand_pass(p_epoch INTEGER, p_max_id BIGINT, p_commitment BYTEA)
RETURNS VOID AS $$
BEGIN
    IF p_epoch IS NULL OR p_max_id IS NULL OR p_commitment IS NULL THEN
        RAISE EXCEPTION 'rerandomization pass start arguments must be non-null';
    END IF;
    UPDATE rerand_control
       SET active_epoch = p_epoch,
           active_seed_commitment = p_commitment,
           next_id = 1,
           max_id = p_max_id
     WHERE singleton;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE FUNCTION advance_rerand_pass(p_epoch INTEGER, p_next_id BIGINT)
RETURNS VOID AS $$
BEGIN
    IF p_epoch IS NULL OR p_next_id IS NULL THEN
        RAISE EXCEPTION 'rerandomization cursor arguments must be non-null';
    END IF;
    UPDATE rerand_control SET next_id = p_next_id
     WHERE singleton AND active_epoch = p_epoch;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'rerandomization pass epoch changed';
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Protocol version 1 is the persisted end-of-epoch masked linear check. The
-- immutable common success marker is validated by the caller before this
-- transition. Keeping the version in the signature makes old binaries fail
-- closed instead of completing an unchecked pass.
CREATE FUNCTION complete_rerand_pass(p_epoch INTEGER, p_check_protocol_version INTEGER)
RETURNS VOID AS $$
BEGIN
    IF p_epoch IS NULL OR p_check_protocol_version IS DISTINCT FROM 1 THEN
        RAISE EXCEPTION 'invalid rerandomization completion check binding';
    END IF;
    UPDATE rerand_control
       SET last_completed_epoch = active_epoch,
           last_seed_commitment = active_seed_commitment,
           active_epoch = NULL,
           active_seed_commitment = NULL,
           next_id = NULL,
           max_id = NULL
     WHERE singleton AND active_epoch = p_epoch;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'rerandomization pass epoch changed';
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

REVOKE ALL ON FUNCTION try_rerand_pass_lock() FROM PUBLIC;
REVOKE ALL ON FUNCTION unlock_rerand_pass_lock() FROM PUBLIC;
REVOKE ALL ON FUNCTION begin_rerand_pass(INTEGER, BIGINT, BYTEA) FROM PUBLIC;
REVOKE ALL ON FUNCTION advance_rerand_pass(INTEGER, BIGINT) FROM PUBLIC;
REVOKE ALL ON FUNCTION complete_rerand_pass(INTEGER, INTEGER) FROM PUBLIC;

-- The writer has no direct UPDATE privilege on irises. This schema-bound
-- function is the only representation-write surface and makes the
-- (id, version_id, semantic_id, old_epoch) CAS structurally unavoidable.
CREATE FUNCTION apply_rerand_updates(
    p_ids BIGINT[],
    p_versions SMALLINT[],
    p_semantic_ids TEXT[],
    p_from_epochs INTEGER[],
    p_left_codes BYTEA[],
    p_left_masks BYTEA[],
    p_right_codes BYTEA[],
    p_right_masks BYTEA[],
    p_to_epoch INTEGER
)
RETURNS BIGINT AS $$
DECLARE
    updated BIGINT;
    control rerand_control%ROWTYPE;
BEGIN
    SELECT * INTO STRICT control FROM rerand_control WHERE singleton;
    IF control.store_id IS NULL
       OR session_user::regrole::oid <> control.writer_role
       OR control.active_epoch IS DISTINCT FROM p_to_epoch
       OR NOT rerand_pass_lock_held() THEN
        RAISE EXCEPTION 'unauthorized rerandomization write';
    END IF;
    IF p_to_epoch IS NULL
       OR cardinality(p_ids) IS NULL
       OR cardinality(p_ids) = 0
       OR cardinality(p_ids) IS DISTINCT FROM cardinality(p_versions)
       OR cardinality(p_ids) IS DISTINCT FROM cardinality(p_semantic_ids)
       OR cardinality(p_ids) IS DISTINCT FROM cardinality(p_from_epochs)
       OR cardinality(p_ids) IS DISTINCT FROM cardinality(p_left_codes)
       OR cardinality(p_ids) IS DISTINCT FROM cardinality(p_left_masks)
       OR cardinality(p_ids) IS DISTINCT FROM cardinality(p_right_codes)
       OR cardinality(p_ids) IS DISTINCT FROM cardinality(p_right_masks)
       OR EXISTS (SELECT 1 FROM pg_catalog.unnest(p_ids) id WHERE id IS NULL)
       OR EXISTS (SELECT 1 FROM pg_catalog.unnest(p_versions) v WHERE v IS NULL)
       OR EXISTS (
           SELECT 1 FROM pg_catalog.unnest(p_from_epochs) e
            WHERE e IS NULL OR e < 0 OR e >= p_to_epoch
       )
       OR EXISTS (SELECT 1 FROM pg_catalog.unnest(p_left_codes) v WHERE v IS NULL)
       OR EXISTS (SELECT 1 FROM pg_catalog.unnest(p_left_masks) v WHERE v IS NULL)
       OR EXISTS (SELECT 1 FROM pg_catalog.unnest(p_right_codes) v WHERE v IS NULL)
       OR EXISTS (SELECT 1 FROM pg_catalog.unnest(p_right_masks) v WHERE v IS NULL)
       OR (SELECT COUNT(*) FROM pg_catalog.unnest(p_ids) id) <>
          (SELECT COUNT(DISTINCT id) FROM pg_catalog.unnest(p_ids) id) THEN
        RAISE EXCEPTION 'invalid rerandomization CAS batch';
    END IF;

    UPDATE irises AS i SET
      left_code = u.left_code, left_mask = u.left_mask,
      right_code = u.right_code, right_mask = u.right_mask,
      rerand_epoch = p_to_epoch
    FROM ROWS FROM (
        pg_catalog.unnest(p_ids),
        pg_catalog.unnest(p_versions),
        pg_catalog.unnest(p_semantic_ids),
        pg_catalog.unnest(p_from_epochs),
        pg_catalog.unnest(p_left_codes),
        pg_catalog.unnest(p_left_masks),
        pg_catalog.unnest(p_right_codes),
        pg_catalog.unnest(p_right_masks)
    ) AS u(id, version_id, semantic_id, from_epoch,
           left_code, left_mask, right_code, right_mask)
    WHERE i.id = u.id AND i.version_id = u.version_id
      AND i.semantic_id::text IS NOT DISTINCT FROM u.semantic_id
      AND i.rerand_epoch = u.from_epoch;
    GET DIAGNOSTICS updated = ROW_COUNT;
    RETURN updated;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT;

REVOKE ALL ON FUNCTION apply_rerand_updates(
    BIGINT[], SMALLINT[], TEXT[], INTEGER[], BYTEA[], BYTEA[], BYTEA[], BYTEA[], INTEGER
) FROM PUBLIC;

-- Serving roles need to verify identity but need no direct access to the
-- control table.
CREATE FUNCTION get_rerand_store_state()
RETURNS TABLE (
    store_id TEXT,
    environment TEXT,
    coordination_id TEXT,
    party_id SMALLINT,
    store_kind TEXT,
    writer_role TEXT,
    last_completed_epoch INTEGER,
    last_seed_commitment BYTEA,
    active_epoch INTEGER,
    active_seed_commitment BYTEA,
    next_id BIGINT,
    max_id BIGINT
) AS $$
    SELECT store_id, environment, coordination_id, party_id, store_kind,
           pg_catalog.pg_get_userbyid(writer_role), last_completed_epoch,
           last_seed_commitment, active_epoch, active_seed_commitment,
           next_id, max_id
      FROM rerand_control WHERE singleton
$$ LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT;

REVOKE ALL ON FUNCTION get_rerand_store_state() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION get_rerand_store_state() TO PUBLIC;

CREATE FUNCTION protect_rerand_control()
RETURNS TRIGGER AS $$
DECLARE
    owner_oid OID;
BEGIN
    SELECT relowner INTO owner_oid FROM pg_class WHERE oid = 'irises'::regclass;

    IF OLD.store_id IS DISTINCT FROM NEW.store_id
       OR OLD.environment IS DISTINCT FROM NEW.environment
       OR OLD.coordination_id IS DISTINCT FROM NEW.coordination_id
       OR OLD.party_id IS DISTINCT FROM NEW.party_id
       OR OLD.store_kind IS DISTINCT FROM NEW.store_kind
       OR OLD.writer_role IS DISTINCT FROM NEW.writer_role
       OR OLD.writer_had_schema_usage IS DISTINCT FROM NEW.writer_had_schema_usage
       OR OLD.writer_had_iris_select IS DISTINCT FROM NEW.writer_had_iris_select THEN
        IF OLD.store_id IS NOT NULL OR session_user::regrole::oid <> owner_oid
           OR OLD.last_completed_epoch <> 0 OR OLD.active_epoch IS NOT NULL THEN
            RAISE EXCEPTION 'rerandomization store identity is immutable';
        END IF;
        RETURN NEW;
    END IF;

    IF NEW.store_id IS NULL OR session_user::regrole::oid <> NEW.writer_role THEN
        RAISE EXCEPTION 'rerandomization control update requires the configured writer role';
    END IF;
    IF NOT rerand_pass_lock_held() THEN
        RAISE EXCEPTION 'rerandomization pass advisory lock is not held by this session';
    END IF;

    IF OLD.active_epoch IS NULL AND NEW.active_epoch IS NOT NULL THEN
        IF NEW.last_completed_epoch IS DISTINCT FROM OLD.last_completed_epoch
           OR NEW.last_seed_commitment IS DISTINCT FROM OLD.last_seed_commitment
           OR NEW.active_epoch IS DISTINCT FROM OLD.last_completed_epoch + 1
           OR NEW.active_seed_commitment IS NULL
           OR pg_catalog.octet_length(NEW.active_seed_commitment) IS DISTINCT FROM 32
           OR NEW.max_id IS NULL
           OR NEW.next_id IS DISTINCT FROM 1 THEN
            RAISE EXCEPTION 'invalid rerandomization pass start';
        END IF;
    ELSIF OLD.active_epoch IS NOT NULL AND NEW.active_epoch IS NOT NULL THEN
        IF NEW.active_epoch IS DISTINCT FROM OLD.active_epoch
           OR NEW.max_id IS DISTINCT FROM OLD.max_id
           OR NEW.active_seed_commitment IS DISTINCT FROM OLD.active_seed_commitment
           OR NEW.last_seed_commitment IS DISTINCT FROM OLD.last_seed_commitment
           OR NEW.last_completed_epoch IS DISTINCT FROM OLD.last_completed_epoch
           OR NEW.next_id IS NULL OR OLD.next_id IS NULL
           OR NEW.next_id < OLD.next_id THEN
            RAISE EXCEPTION 'rerandomization cursor may only advance within the active pass';
        END IF;
    ELSIF OLD.active_epoch IS NOT NULL AND NEW.active_epoch IS NULL THEN
        IF OLD.next_id IS NULL OR OLD.max_id IS NULL
           OR OLD.next_id IS DISTINCT FROM OLD.max_id + 1
           OR NEW.last_completed_epoch IS DISTINCT FROM OLD.active_epoch
           OR NEW.last_seed_commitment IS DISTINCT FROM OLD.active_seed_commitment
           OR EXISTS (SELECT 1 FROM irises WHERE semantic_id IS NULL)
           OR EXISTS (
               SELECT 1 FROM irises
                WHERE rerand_epoch <> 0 AND rerand_epoch <> OLD.active_epoch
           ) THEN
            RAISE EXCEPTION 'cannot complete an unfinished rerandomization pass';
        END IF;
    ELSE
        RAISE EXCEPTION 'invalid rerandomization control transition';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER protect_rerand_control_trigger
    BEFORE UPDATE ON rerand_control
    FOR EACH ROW EXECUTE FUNCTION protect_rerand_control();
ALTER TABLE rerand_control ENABLE ALWAYS TRIGGER protect_rerand_control_trigger;

CREATE FUNCTION update_iris_metadata()
RETURNS TRIGGER AS $$
DECLARE
    shares_changed BOOLEAN;
    configured_store_id TEXT;
    configured_writer OID;
    configured_active_epoch INTEGER;
    owner_oid OID;
BEGIN
    SELECT store_id, writer_role, active_epoch
      INTO STRICT configured_store_id, configured_writer, configured_active_epoch
      FROM rerand_control WHERE singleton;
    SELECT relowner INTO owner_oid FROM pg_class WHERE oid = 'irises'::regclass;
    IF TG_OP = 'DELETE' THEN
        IF configured_writer IS NOT NULL
           AND session_user::regrole::oid = configured_writer THEN
            RAISE EXCEPTION 'rerandomization writer cannot delete iris rows';
        END IF;
        RETURN OLD;
    END IF;
    IF TG_OP = 'INSERT' THEN
        IF configured_writer IS NOT NULL
           AND session_user::regrole::oid = configured_writer THEN
            RAISE EXCEPTION 'rerandomization writer cannot insert iris rows';
        END IF;
        IF NEW.rerand_epoch <> 0 THEN
            RAISE EXCEPTION 'new iris rows must have rerand_epoch 0';
        END IF;
        NEW.semantic_id = gen_random_uuid();
        NEW.last_modified_at = EXTRACT(EPOCH FROM NOW())::BIGINT;
        RETURN NEW;
    END IF;

    IF NEW.id <> OLD.id THEN
        RAISE EXCEPTION 'iris id is immutable';
    END IF;
    shares_changed :=
        OLD.left_code IS DISTINCT FROM NEW.left_code OR
        OLD.left_mask IS DISTINCT FROM NEW.left_mask OR
        OLD.right_code IS DISTINCT FROM NEW.right_code OR
        OLD.right_mask IS DISTINCT FROM NEW.right_mask;

    IF shares_changed AND NEW.rerand_epoch > OLD.rerand_epoch THEN
        IF configured_store_id IS NULL
           OR session_user::regrole::oid <> configured_writer
           OR current_user::regrole::oid <> owner_oid
           OR configured_active_epoch IS DISTINCT FROM NEW.rerand_epoch
           OR NOT rerand_pass_lock_held() THEN
            RAISE EXCEPTION 'unauthorized rerandomization write';
        END IF;
        NEW.version_id = OLD.version_id;
        NEW.semantic_id = COALESCE(OLD.semantic_id, gen_random_uuid());
        NEW.last_modified_at = OLD.last_modified_at;
    ELSIF shares_changed THEN
        IF session_user::regrole::oid = configured_writer THEN
            RAISE EXCEPTION 'rerandomization writer may only use the guarded CAS function';
        END IF;
        -- Every semantic write contains raw shares, regardless of what epoch
        -- the writer copied from the old row.
        NEW.version_id = OLD.version_id + 1;
        NEW.rerand_epoch = 0;
        NEW.semantic_id = gen_random_uuid();
        NEW.last_modified_at = EXTRACT(EPOCH FROM NOW())::BIGINT;
    ELSE
        IF session_user::regrole::oid = configured_writer THEN
            RAISE EXCEPTION 'rerandomization writer may only use the guarded CAS function';
        END IF;
        IF NEW.rerand_epoch IS DISTINCT FROM OLD.rerand_epoch
           OR NEW.version_id IS DISTINCT FROM OLD.version_id
           OR NEW.semantic_id IS DISTINCT FROM OLD.semantic_id THEN
            RAISE EXCEPTION 'share metadata cannot change without share bytes';
        END IF;
        NEW.last_modified_at = EXTRACT(EPOCH FROM NOW())::BIGINT;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_iris_metadata_trigger
    BEFORE INSERT OR UPDATE OR DELETE ON irises
    FOR EACH ROW EXECUTE FUNCTION update_iris_metadata();
ALTER TABLE irises ENABLE ALWAYS TRIGGER update_iris_metadata_trigger;

CREATE FUNCTION protect_iris_truncate()
RETURNS TRIGGER AS $$
DECLARE
    configured_writer OID;
BEGIN
    SELECT writer_role INTO configured_writer FROM rerand_control WHERE singleton;
    IF configured_writer IS NOT NULL
       AND session_user::regrole::oid = configured_writer THEN
        RAISE EXCEPTION 'rerandomization writer cannot truncate iris rows';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

REVOKE ALL ON FUNCTION protect_iris_truncate() FROM PUBLIC;

CREATE TRIGGER protect_iris_truncate_trigger
    BEFORE TRUNCATE ON irises
    FOR EACH STATEMENT EXECUTE FUNCTION protect_iris_truncate();
ALTER TABLE irises ENABLE ALWAYS TRIGGER protect_iris_truncate_trigger;

-- Pin every function to the migration's actual schema. Explicitly placing
-- pg_temp last prevents a caller-created temporary table from shadowing the
-- authoritative control or iris tables.
DO $$
DECLARE
    schema_name NAME := pg_catalog.current_schema();
    iris_owner NAME;
BEGIN
    SELECT pg_catalog.pg_get_userbyid(c.relowner) INTO iris_owner
      FROM pg_catalog.pg_class c WHERE c.oid = 'irises'::regclass;

    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.initialize_rerand_store(TEXT, TEXT, TEXT, SMALLINT, TEXT, NAME) SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.rerand_pass_lock_held() SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.try_rerand_pass_lock() SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.unlock_rerand_pass_lock() SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.begin_rerand_pass(INTEGER, BIGINT, BYTEA) SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.advance_rerand_pass(INTEGER, BIGINT) SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.complete_rerand_pass(INTEGER, INTEGER) SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.apply_rerand_updates(BIGINT[], SMALLINT[], TEXT[], INTEGER[], BYTEA[], BYTEA[], BYTEA[], BYTEA[], INTEGER) SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.get_rerand_store_state() SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.protect_rerand_control() SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.update_iris_metadata() SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.protect_iris_truncate() SET search_path TO pg_catalog, %1$I, pg_temp', schema_name);

    -- SECURITY DEFINER execution and the trigger's current_user check must
    -- agree even when an administrative migration role ran this migration.
    EXECUTE pg_catalog.format('ALTER TABLE %I.rerand_control OWNER TO %I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.initialize_rerand_store(TEXT, TEXT, TEXT, SMALLINT, TEXT, NAME) OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.rerand_pass_lock_held() OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.try_rerand_pass_lock() OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.unlock_rerand_pass_lock() OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.begin_rerand_pass(INTEGER, BIGINT, BYTEA) OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.advance_rerand_pass(INTEGER, BIGINT) OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.complete_rerand_pass(INTEGER, INTEGER) OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.apply_rerand_updates(BIGINT[], SMALLINT[], TEXT[], INTEGER[], BYTEA[], BYTEA[], BYTEA[], BYTEA[], INTEGER) OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.get_rerand_store_state() OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.protect_rerand_control() OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.update_iris_metadata() OWNER TO %2$I', schema_name, iris_owner);
    EXECUTE pg_catalog.format('ALTER FUNCTION %1$I.protect_iris_truncate() OWNER TO %2$I', schema_name, iris_owner);
END;
$$;

COMMENT ON TABLE rerand_control IS
    'Owner-initialized rerandomization identity and bounded pass cursor. Epoch seeds are intentionally never deleted by this implementation.';
