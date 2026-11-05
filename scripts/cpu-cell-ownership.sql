-- Installed atomically by cpu_cell::bootstrap, using a separately credentialed
-- provisioner. __SCHEMA__ is replaced with a quoted identifier, never input SQL.
-- Control objects are owned by a dedicated NOLOGIN authority role; the canonical
-- schema remains separately provisioned and trusted. Migration credentials stay
-- separate; runtime roles never receive DDL privileges.
CREATE TABLE __SCHEMA__.cpu_cell_authority (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    incarnation uuid NOT NULL,
    revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0),
    party_id integer NOT NULL CHECK (party_id BETWEEN 0 AND 2),
    intent_login name NOT NULL,
    recovery_login name NOT NULL CHECK (recovery_login <> intent_login),
    current_owner uuid,
    current_attempt uuid,
    recovery_blocked boolean NOT NULL DEFAULT true,
    recovery_id uuid,
    fencing_ack text
);
CREATE TABLE __SCHEMA__.cpu_cell_generations (
    generation uuid PRIMARY KEY,
    slot text NOT NULL CHECK (slot IN ('a','b')),
    login name UNIQUE NOT NULL,
    image text NOT NULL CHECK (image ~ '^sha256:[0-9a-f]{64}$'),
    schema_version text NOT NULL CHECK (schema_version <> ''),
    protocol text NOT NULL CHECK (protocol <> ''),
    configuration text NOT NULL CHECK (configuration <> ''),
    process_id uuid UNIQUE,
    secret_hash bytea CHECK (octet_length(secret_hash) = 32),
    loaded boolean NOT NULL DEFAULT false,
    sealed boolean NOT NULL DEFAULT false,
    retired boolean NOT NULL DEFAULT false,
    CHECK ((process_id IS NULL) = (secret_hash IS NULL)),
    CHECK (NOT retired OR sealed)
);
CREATE TABLE __SCHEMA__.cpu_cell_intents (
    intent uuid PRIMARY KEY,
    source uuid NOT NULL REFERENCES __SCHEMA__.cpu_cell_generations,
    destination uuid NOT NULL REFERENCES __SCHEMA__.cpu_cell_generations,
    automatic boolean NOT NULL,
    CHECK (source <> destination)
);
CREATE TABLE __SCHEMA__.cpu_cell_attempts (
    attempt uuid PRIMARY KEY,
    intent uuid UNIQUE NOT NULL REFERENCES __SCHEMA__.cpu_cell_intents,
    source_processes uuid[] NOT NULL CHECK (array_ndims(source_processes) = 1 AND array_length(source_processes, 1) = 3 AND array_lower(source_processes, 1) = 1),
    destination_processes uuid[] NOT NULL CHECK (array_ndims(destination_processes) = 1 AND array_length(destination_processes, 1) = 3 AND array_lower(destination_processes, 1) = 1),
    watermark text,
    proof bytea CHECK (octet_length(proof) = 32),
    reconciled boolean NOT NULL DEFAULT false,
    source_retired boolean NOT NULL DEFAULT false,
    retirement_certificate boolean NOT NULL DEFAULT false,
    acquired boolean NOT NULL DEFAULT false,
    superseded boolean NOT NULL DEFAULT false,
    CHECK (NOT reconciled OR proof IS NOT NULL),
    CHECK (NOT source_retired OR reconciled),
    CHECK (NOT retirement_certificate OR source_retired),
    CHECK (NOT acquired OR retirement_certificate)
);
ALTER TABLE __SCHEMA__.cpu_cell_authority
    ADD FOREIGN KEY (current_owner) REFERENCES __SCHEMA__.cpu_cell_generations(generation),
    ADD FOREIGN KEY (current_attempt) REFERENCES __SCHEMA__.cpu_cell_attempts(attempt);

CREATE FUNCTION __SCHEMA__.cpu_cell_lock(i uuid)
RETURNS __SCHEMA__.cpu_cell_authority LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority;
BEGIN
    SELECT * INTO STRICT a FROM __SCHEMA__.cpu_cell_authority WHERE singleton FOR UPDATE;
    IF i IS DISTINCT FROM a.incarnation THEN RAISE EXCEPTION 'CPU authority incarnation mismatch'; END IF;
    RETURN a;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_auth(i uuid, g uuid, p uuid, secret bytea)
RETURNS __SCHEMA__.cpu_cell_generations LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority; r __SCHEMA__.cpu_cell_generations;
BEGIN
    a := __SCHEMA__.cpu_cell_lock(i);
    SELECT * INTO STRICT r FROM __SCHEMA__.cpu_cell_generations WHERE generation = g;
    IF r.login <> session_user OR r.process_id IS DISTINCT FROM p
       OR octet_length(secret) IS DISTINCT FROM 32 OR r.secret_hash IS DISTINCT FROM sha256(secret)
    THEN RAISE EXCEPTION 'CPU process binding rejected'; END IF;
    IF a.recovery_blocked THEN RAISE EXCEPTION 'CPU recovery blocks runtime operations'; END IF;
    RETURN r;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_runtime_role(role_name name) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE r pg_catalog.pg_roles;
BEGIN
    SELECT * INTO STRICT r FROM pg_catalog.pg_roles WHERE rolname = role_name;
    IF NOT r.rolcanlogin OR r.rolsuper OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication OR r.rolbypassrls
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE member = r.oid)
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspowner = r.oid)
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relowner = r.oid)
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_proc WHERE proowner = r.oid)
    THEN RAISE EXCEPTION 'CPU runtime role must be unprivileged, without membership or owned objects'; END IF;
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_attribute att
        JOIN pg_catalog.pg_class c ON c.oid=att.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        CROSS JOIN LATERAL pg_catalog.aclexplode(att.attacl) acl
        WHERE n.nspname='__SCHEMA_NAME__' AND acl.grantee IN (0,r.oid)
          AND acl.privilege_type IN ('INSERT','UPDATE','REFERENCES'))
    THEN RAISE EXCEPTION 'CPU runtime role has forbidden column write privileges'; END IF;
    IF has_database_privilege(role_name, current_database(), 'CREATE')
       OR has_schema_privilege(role_name, '__SCHEMA_NAME__', 'CREATE')
    THEN RAISE EXCEPTION 'CPU runtime role must not have CREATE privileges'; END IF;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_register_generation(i uuid, g uuid, s text, role_name name, img text, sv text, proto text, cfg text)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority; r __SCHEMA__.cpu_cell_generations; f record;
BEGIN
    a := __SCHEMA__.cpu_cell_lock(i);
    IF session_user NOT IN (a.intent_login, a.recovery_login) THEN RAISE EXCEPTION 'CPU provisioning permission denied'; END IF;
    IF role_name IN (a.intent_login, a.recovery_login, current_user) THEN RAISE EXCEPTION 'CPU runtime and administrative roles must differ'; END IF;
    PERFORM __SCHEMA__.cpu_cell_runtime_role(role_name);
    SELECT * INTO r FROM __SCHEMA__.cpu_cell_generations WHERE generation = g;
    IF FOUND THEN
        IF (r.slot, r.login, r.image, r.schema_version, r.protocol, r.configuration) IS DISTINCT FROM (s, role_name, img, sv, proto, cfg)
        THEN RAISE EXCEPTION 'CPU generation identity is immutable'; END IF;
        RETURN;
    END IF;
    -- A grant option does not permit revoking another grantor's legacy grants.
    -- Start clean so every later canonical write grant belongs to this authority.
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='__SCHEMA_NAME__' AND c.relkind IN ('r','p','v','f')
          AND has_table_privilege(role_name,c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'))
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='__SCHEMA_NAME__' AND CASE WHEN c.relkind='S' THEN has_sequence_privilege(role_name,c.oid,'USAGE,UPDATE') ELSE false END)
    THEN RAISE EXCEPTION 'CPU runtime role must have no pre-existing canonical write grants'; END IF;
    INSERT INTO __SCHEMA__.cpu_cell_generations(generation, slot, login, image, schema_version, protocol, configuration)
        VALUES (g, s, role_name, img, sv, proto, cfg);
    EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA %I FROM %I', '__SCHEMA_NAME__', role_name);
    EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA %I FROM %I', '__SCHEMA_NAME__', role_name);
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO %I', '__SCHEMA_NAME__', role_name);
    -- Control rows (especially the secret hash) are never exposed by table grants.
    FOR f IN SELECT c.relname FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='__SCHEMA_NAME__' AND c.relkind IN ('r','p','v','m','f') AND left(c.relname,9) <> 'cpu_cell_'
    LOOP EXECUTE format('GRANT SELECT ON %I.%I TO %I', '__SCHEMA_NAME__', f.relname, role_name); END LOOP;
    FOR f IN SELECT p.oid::regprocedure AS signature FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
        WHERE n.nspname='__SCHEMA_NAME__' AND p.proname IN ('cpu_cell_status','cpu_cell_inspect','cpu_cell_bind_process','cpu_cell_assert_owner','cpu_cell_assert_settling_owner','cpu_cell_assert_candidate','cpu_cell_mark_loaded','cpu_cell_freeze_attempt','cpu_cell_seal_source','cpu_cell_mark_reconciled','cpu_cell_retire_source','cpu_cell_record_retirement_certificate','cpu_cell_acquire')
    LOOP EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO %I', f.signature, role_name); END LOOP;
    UPDATE __SCHEMA__.cpu_cell_authority SET revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_canonical_writes(role_name name, enable boolean) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE r record;
BEGIN
    FOR r IN SELECT c.relname FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='__SCHEMA_NAME__' AND c.relkind IN ('r','p') AND left(c.relname,9) <> 'cpu_cell_'
    LOOP
        IF enable AND r.relname <> '_sqlx_migrations' THEN
            EXECUTE format('GRANT INSERT, UPDATE, DELETE ON %I.%I TO %I', '__SCHEMA_NAME__', r.relname, role_name);
        ELSE
            EXECUTE format('REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON %I.%I FROM %I', '__SCHEMA_NAME__', r.relname, role_name);
        END IF;
    END LOOP;
    IF enable THEN EXECUTE format('GRANT USAGE ON ALL SEQUENCES IN SCHEMA %I TO %I', '__SCHEMA_NAME__', role_name);
    ELSE EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA %I FROM %I', '__SCHEMA_NAME__', role_name); END IF;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_bind_process(i uuid, g uuid, p uuid, secret bytea, s text, img text, sv text, proto text, cfg text)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority; r __SCHEMA__.cpu_cell_generations;
BEGIN
    a := __SCHEMA__.cpu_cell_lock(i);
    SELECT * INTO STRICT r FROM __SCHEMA__.cpu_cell_generations WHERE generation=g;
    IF a.recovery_blocked OR r.sealed OR r.retired THEN RAISE EXCEPTION 'CPU generation blocked, sealed or retired'; END IF;
    IF r.login <> session_user OR p IS NULL OR octet_length(secret) IS DISTINCT FROM 32
       OR (r.slot,r.image,r.schema_version,r.protocol,r.configuration) IS DISTINCT FROM (s,img,sv,proto,cfg)
    THEN RAISE EXCEPTION 'CPU generation login or identity mismatch'; END IF;
    PERFORM __SCHEMA__.cpu_cell_runtime_role(r.login);
    IF r.process_id IS NOT NULL THEN
        IF r.process_id IS DISTINCT FROM p OR r.secret_hash IS DISTINCT FROM sha256(secret)
        THEN RAISE EXCEPTION 'CPU generation already bound; authorized recovery required'; END IF;
        RETURN;
    END IF;
    UPDATE __SCHEMA__.cpu_cell_generations SET process_id=p, secret_hash=sha256(secret) WHERE generation=g;
    IF a.current_owner=g THEN PERFORM __SCHEMA__.cpu_cell_canonical_writes(r.login,true); END IF;
    UPDATE __SCHEMA__.cpu_cell_authority SET revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_assert_owner(i uuid,g uuid,p uuid,secret bytea) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE r __SCHEMA__.cpu_cell_generations; a __SCHEMA__.cpu_cell_authority;
BEGIN
    r := __SCHEMA__.cpu_cell_auth(i,g,p,secret);
    SELECT * INTO STRICT a FROM __SCHEMA__.cpu_cell_authority WHERE singleton;
    IF a.current_owner IS DISTINCT FROM g OR r.sealed OR r.retired THEN RAISE EXCEPTION 'CPU process is not an unsealed owner'; END IF;
END $$;
CREATE FUNCTION __SCHEMA__.cpu_cell_assert_settling_owner(i uuid,g uuid,p uuid,secret bytea) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE r __SCHEMA__.cpu_cell_generations; a __SCHEMA__.cpu_cell_authority;
BEGIN
    r := __SCHEMA__.cpu_cell_auth(i,g,p,secret);
    SELECT * INTO STRICT a FROM __SCHEMA__.cpu_cell_authority WHERE singleton;
    IF a.current_owner IS DISTINCT FROM g OR r.retired THEN RAISE EXCEPTION 'CPU process cannot settle canonical effects'; END IF;
END $$;
CREATE FUNCTION __SCHEMA__.cpu_cell_assert_candidate(i uuid,g uuid,p uuid,secret bytea) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE r __SCHEMA__.cpu_cell_generations; a __SCHEMA__.cpu_cell_authority;
BEGIN
    r := __SCHEMA__.cpu_cell_auth(i,g,p,secret);
    SELECT * INTO STRICT a FROM __SCHEMA__.cpu_cell_authority WHERE singleton;
    IF a.current_owner=g OR r.sealed OR r.retired THEN RAISE EXCEPTION 'CPU process is not a passive candidate'; END IF;
END $$;
CREATE FUNCTION __SCHEMA__.cpu_cell_mark_loaded(i uuid,g uuid,p uuid,secret bytea) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
BEGIN
    PERFORM __SCHEMA__.cpu_cell_assert_candidate(i,g,p,secret);
    UPDATE __SCHEMA__.cpu_cell_generations SET loaded=true WHERE generation=g;
    UPDATE __SCHEMA__.cpu_cell_authority SET revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_write_intent(i uuid,id uuid,src uuid,dst uuid,auto boolean) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority; old __SCHEMA__.cpu_cell_intents; s __SCHEMA__.cpu_cell_generations; d __SCHEMA__.cpu_cell_generations;
BEGIN
    a := __SCHEMA__.cpu_cell_lock(i);
    IF session_user <> a.intent_login THEN RAISE EXCEPTION 'CPU deployment intent permission denied'; END IF;
    SELECT * INTO old FROM __SCHEMA__.cpu_cell_intents WHERE intent=id;
    IF FOUND THEN
        IF (old.source,old.destination,old.automatic) IS DISTINCT FROM (src,dst,auto) THEN RAISE EXCEPTION 'CPU deployment intent is immutable'; END IF;
        RETURN;
    END IF;
    SELECT * INTO STRICT s FROM __SCHEMA__.cpu_cell_generations WHERE generation=src;
    SELECT * INTO STRICT d FROM __SCHEMA__.cpu_cell_generations WHERE generation=dst;
    IF a.recovery_blocked OR a.current_owner IS DISTINCT FROM src OR s.sealed OR s.retired OR d.sealed OR d.retired
       OR s.slot=d.slot OR src=dst OR s.schema_version<>d.schema_version OR s.protocol<>d.protocol OR s.configuration<>d.configuration
    THEN RAISE EXCEPTION 'CPU deployment intent is stale or incompatible'; END IF;
    INSERT INTO __SCHEMA__.cpu_cell_intents VALUES(id,src,dst,auto);
    UPDATE __SCHEMA__.cpu_cell_authority SET revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_freeze_attempt(i uuid,g uuid,p uuid,secret bytea,id uuid,intent_id uuid,sources uuid[],destinations uuid[]) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority; intent_row __SCHEMA__.cpu_cell_intents; d __SCHEMA__.cpu_cell_generations; old __SCHEMA__.cpu_cell_attempts;
BEGIN
    PERFORM __SCHEMA__.cpu_cell_auth(i,g,p,secret);
    SELECT * INTO STRICT a FROM __SCHEMA__.cpu_cell_authority WHERE singleton;
    SELECT * INTO STRICT intent_row FROM __SCHEMA__.cpu_cell_intents WHERE intent=intent_id;
    IF intent_row.source<>g OR a.current_owner IS DISTINCT FROM g OR NOT intent_row.automatic THEN RAISE EXCEPTION 'CPU source or automatic intent rejected'; END IF;
    SELECT * INTO STRICT d FROM __SCHEMA__.cpu_cell_generations WHERE generation=intent_row.destination;
    IF cardinality(sources) IS DISTINCT FROM 3 OR cardinality(destinations) IS DISTINCT FROM 3
       OR array_position(sources,NULL) IS NOT NULL OR array_position(destinations,NULL) IS NOT NULL
       OR (SELECT count(DISTINCT x) FROM unnest(sources || destinations) x)<>6
       OR sources[a.party_id+1] IS DISTINCT FROM p OR destinations[a.party_id+1] IS DISTINCT FROM d.process_id
       OR NOT d.loaded OR d.sealed OR d.retired
    THEN RAISE EXCEPTION 'CPU exact cohorts or loaded preflight rejected'; END IF;
    SELECT * INTO old FROM __SCHEMA__.cpu_cell_attempts WHERE attempt=id;
    IF FOUND THEN
        IF a.current_attempt IS DISTINCT FROM id OR old.superseded OR (old.intent,old.source_processes,old.destination_processes) IS DISTINCT FROM (intent_id,sources,destinations)
        THEN RAISE EXCEPTION 'CPU attempt is stale or immutable'; END IF;
        RETURN;
    END IF;
    IF a.current_attempt IS NOT NULL AND NOT EXISTS (SELECT 1 FROM __SCHEMA__.cpu_cell_attempts WHERE attempt=a.current_attempt AND acquired AND NOT superseded)
    THEN RAISE EXCEPTION 'CPU competing attempt rejected'; END IF;
    PERFORM __SCHEMA__.cpu_cell_assert_owner(i,g,p,secret);
    UPDATE __SCHEMA__.cpu_cell_attempts SET superseded=true WHERE attempt=a.current_attempt;
    INSERT INTO __SCHEMA__.cpu_cell_attempts(attempt,intent,source_processes,destination_processes) VALUES(id,intent_id,sources,destinations);
    UPDATE __SCHEMA__.cpu_cell_authority SET current_attempt=id,revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_attempt(i uuid,g uuid,p uuid,secret bytea,id uuid,is_source boolean)
RETURNS __SCHEMA__.cpu_cell_attempts LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority; t __SCHEMA__.cpu_cell_attempts; approval __SCHEMA__.cpu_cell_intents;
BEGIN
    PERFORM __SCHEMA__.cpu_cell_auth(i,g,p,secret);
    SELECT * INTO STRICT a FROM __SCHEMA__.cpu_cell_authority WHERE singleton;
    SELECT * INTO STRICT t FROM __SCHEMA__.cpu_cell_attempts WHERE attempt=id;
    SELECT * INTO STRICT approval FROM __SCHEMA__.cpu_cell_intents WHERE intent=t.intent;
    IF a.current_attempt IS DISTINCT FROM id OR t.superseded
       OR g IS DISTINCT FROM (CASE WHEN is_source THEN approval.source ELSE approval.destination END)
       OR p IS DISTINCT FROM (CASE WHEN is_source THEN t.source_processes[a.party_id+1] ELSE t.destination_processes[a.party_id+1] END)
    THEN RAISE EXCEPTION 'CPU stale attempt or wrong cohort'; END IF;
    RETURN t;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_seal_source(i uuid,g uuid,p uuid,secret bytea,id uuid) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE t __SCHEMA__.cpu_cell_attempts;
BEGIN
    t := __SCHEMA__.cpu_cell_attempt(i,g,p,secret,id,true);
    IF t.acquired THEN RAISE EXCEPTION 'CPU source already replaced'; END IF;
    UPDATE __SCHEMA__.cpu_cell_generations SET sealed=true WHERE generation=g;
    -- Admitted work must still settle. Canonical writes are revoked at retirement.
    UPDATE __SCHEMA__.cpu_cell_authority SET revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_mark_reconciled(i uuid,g uuid,p uuid,secret bytea,id uuid,w text,digest bytea) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE t __SCHEMA__.cpu_cell_attempts; source_sealed boolean;
BEGIN
    t := __SCHEMA__.cpu_cell_attempt(i,g,p,secret,id,false);
    SELECT s.sealed INTO STRICT source_sealed FROM __SCHEMA__.cpu_cell_generations s
        JOIN __SCHEMA__.cpu_cell_intents a ON a.source=s.generation WHERE a.intent=t.intent;
    IF NOT source_sealed OR octet_length(digest) IS DISTINCT FROM 32
       OR (w IS NOT NULL AND (w !~ '^[0-9]{40}$' OR w::numeric > 340282366920938463463374607431768211455)) THEN RAISE EXCEPTION 'CPU reconciliation requires seal and complete proof'; END IF;
    IF t.reconciled AND (t.watermark,t.proof) IS DISTINCT FROM (w,digest) THEN RAISE EXCEPTION 'CPU reconciliation proof is immutable'; END IF;
    UPDATE __SCHEMA__.cpu_cell_attempts SET watermark=w,proof=digest,reconciled=true WHERE attempt=id;
    UPDATE __SCHEMA__.cpu_cell_authority SET revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_retire_source(i uuid,g uuid,p uuid,secret bytea,id uuid,w text,digest bytea) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE t __SCHEMA__.cpu_cell_attempts; r __SCHEMA__.cpu_cell_generations;
BEGIN
    t := __SCHEMA__.cpu_cell_attempt(i,g,p,secret,id,true);
    SELECT * INTO STRICT r FROM __SCHEMA__.cpu_cell_generations WHERE generation=g;
    IF NOT r.sealed OR NOT t.reconciled OR (t.watermark,t.proof) IS DISTINCT FROM (w,digest) THEN RAISE EXCEPTION 'CPU retirement requires matching settled proof'; END IF;
    UPDATE __SCHEMA__.cpu_cell_generations SET retired=true WHERE generation=g;
    UPDATE __SCHEMA__.cpu_cell_attempts SET source_retired=true WHERE attempt=id;
    PERFORM __SCHEMA__.cpu_cell_canonical_writes(r.login,false);
    UPDATE __SCHEMA__.cpu_cell_authority SET revision=revision+1 WHERE singleton;
END $$;

-- This source-only attestation assumes the bound source authenticated all three
-- peer retirement statements. Cross-party authentication is POP-4454; keep
-- deployment handoff disabled until that protocol and activation barrier exist.
CREATE FUNCTION __SCHEMA__.cpu_cell_record_retirement_certificate(i uuid,g uuid,p uuid,secret bytea,id uuid,w text,digest bytea,retired_processes uuid[]) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE t __SCHEMA__.cpu_cell_attempts;
BEGIN
    t := __SCHEMA__.cpu_cell_attempt(i,g,p,secret,id,true);
    IF NOT t.source_retired OR (t.watermark,t.proof,t.source_processes) IS DISTINCT FROM (w,digest,retired_processes)
    THEN RAISE EXCEPTION 'CPU complete retirement certificate rejected'; END IF;
    UPDATE __SCHEMA__.cpu_cell_attempts SET retirement_certificate=true WHERE attempt=id;
    UPDATE __SCHEMA__.cpu_cell_authority SET revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_acquire(i uuid,g uuid,p uuid,secret bytea,id uuid,w text,digest bytea) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE t __SCHEMA__.cpu_cell_attempts; r __SCHEMA__.cpu_cell_generations; a __SCHEMA__.cpu_cell_authority; src uuid;
BEGIN
    t := __SCHEMA__.cpu_cell_attempt(i,g,p,secret,id,false);
    SELECT * INTO STRICT r FROM __SCHEMA__.cpu_cell_generations WHERE generation=g;
    SELECT * INTO STRICT a FROM __SCHEMA__.cpu_cell_authority WHERE singleton;
    SELECT source INTO STRICT src FROM __SCHEMA__.cpu_cell_intents WHERE intent=t.intent;
    IF NOT t.retirement_certificate OR r.retired OR r.sealed OR (t.watermark,t.proof) IS DISTINCT FROM (w,digest)
       OR (NOT t.acquired AND a.current_owner IS DISTINCT FROM src) OR (t.acquired AND a.current_owner IS DISTINCT FROM g)
    THEN RAISE EXCEPTION 'CPU acquisition requires current complete retirement certificate'; END IF;
    PERFORM __SCHEMA__.cpu_cell_canonical_writes(r.login,true);
    UPDATE __SCHEMA__.cpu_cell_attempts SET acquired=true WHERE attempt=id;
    UPDATE __SCHEMA__.cpu_cell_authority SET current_owner=g,revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_begin_recovery(i uuid,id uuid,expected_revision bigint) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority; r record;
BEGIN
    a := __SCHEMA__.cpu_cell_lock(i);
    IF session_user<>a.recovery_login THEN RAISE EXCEPTION 'CPU recovery permission denied'; END IF;
    IF id IS NULL THEN RAISE EXCEPTION 'CPU recovery ID required'; END IF;
    IF a.recovery_id=id THEN
        IF NOT a.recovery_blocked THEN RAISE EXCEPTION 'CPU completed recovery cannot reopen'; END IF;
        RETURN;
    END IF;
    IF expected_revision IS DISTINCT FROM a.revision THEN RAISE EXCEPTION 'CPU recovery authority revision changed'; END IF;
    IF a.recovery_blocked AND a.recovery_id IS NOT NULL THEN RAISE EXCEPTION 'CPU competing recovery rejected'; END IF;
    UPDATE __SCHEMA__.cpu_cell_attempts SET superseded=true WHERE NOT superseded;
    UPDATE __SCHEMA__.cpu_cell_generations SET sealed=true WHERE process_id IS NOT NULL;
    -- A deleted historical login has no remaining privileges to revoke.
    FOR r IN SELECT g.login FROM __SCHEMA__.cpu_cell_generations g
        JOIN pg_catalog.pg_roles p ON p.rolname=g.login
    LOOP PERFORM __SCHEMA__.cpu_cell_canonical_writes(r.login,false); END LOOP;
    UPDATE __SCHEMA__.cpu_cell_authority SET recovery_blocked=true,recovery_id=id,fencing_ack=NULL,current_attempt=NULL,revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_finish_recovery(i uuid,id uuid,fresh_generation uuid,external_fencing_ack text,new_incarnation uuid) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
DECLARE a __SCHEMA__.cpu_cell_authority; r __SCHEMA__.cpu_cell_generations;
BEGIN
    a := __SCHEMA__.cpu_cell_lock(i);
    IF session_user<>a.recovery_login THEN RAISE EXCEPTION 'CPU recovery permission denied'; END IF;
    IF NOT a.recovery_blocked OR a.recovery_id IS DISTINCT FROM id OR external_fencing_ack IS NULL OR btrim(external_fencing_ack)='' OR new_incarnation IS NULL
    THEN RAISE EXCEPTION 'CPU recovery requires current ID and explicit external fencing acknowledgement'; END IF;
    SELECT * INTO STRICT r FROM __SCHEMA__.cpu_cell_generations WHERE generation=fresh_generation;
    IF r.process_id IS NOT NULL OR r.sealed OR r.retired OR a.current_owner=fresh_generation
    THEN RAISE EXCEPTION 'CPU recovery requires a fresh unbound generation'; END IF;
    UPDATE __SCHEMA__.cpu_cell_generations SET sealed=true,retired=true WHERE process_id IS NOT NULL;
    UPDATE __SCHEMA__.cpu_cell_authority SET current_owner=fresh_generation,current_attempt=NULL,recovery_blocked=false,
        fencing_ack=external_fencing_ack,incarnation=new_incarnation,revision=revision+1 WHERE singleton;
END $$;

CREATE FUNCTION __SCHEMA__.cpu_cell_status()
RETURNS TABLE(incarnation uuid,revision bigint,party_id integer,current_owner uuid,current_process uuid,
    current_attempt uuid,recovery_blocked boolean,recovery_id uuid,fencing_ack text)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
    SELECT a.incarnation,a.revision,a.party_id,a.current_owner,g.process_id,
        a.current_attempt,a.recovery_blocked,a.recovery_id,a.fencing_ack
    FROM __SCHEMA__.cpu_cell_authority a LEFT JOIN __SCHEMA__.cpu_cell_generations g ON g.generation=a.current_owner
    WHERE a.singleton AND (session_user IN (a.intent_login,a.recovery_login)
        OR EXISTS (SELECT 1 FROM __SCHEMA__.cpu_cell_generations r WHERE r.login=session_user))
$$;

-- Evidence remains readable after the bound source process exits. Secrets are
-- excluded, and historical attempts never confer current activation authority.
CREATE FUNCTION __SCHEMA__.cpu_cell_inspect(i uuid,id uuid)
RETURNS TABLE(incarnation uuid,revision bigint,current_owner uuid,current_attempt uuid,recovery_blocked boolean,
    attempt uuid,intent uuid,source_generation uuid,destination_generation uuid,
    source_processes uuid[],destination_processes uuid[],watermark text,proof bytea,
    source_retired boolean,retirement_certificate boolean,acquired boolean,superseded boolean)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, __SCHEMA__, pg_temp AS $$
    SELECT a.incarnation,a.revision,a.current_owner,a.current_attempt,a.recovery_blocked,
        t.attempt,t.intent,d.source,d.destination,t.source_processes,t.destination_processes,
        t.watermark,t.proof,t.source_retired,t.retirement_certificate,t.acquired,t.superseded
    FROM __SCHEMA__.cpu_cell_authority a CROSS JOIN __SCHEMA__.cpu_cell_attempts t
    JOIN __SCHEMA__.cpu_cell_intents d ON d.intent=t.intent
    WHERE a.singleton AND a.incarnation=i AND t.attempt=id
      AND (session_user IN (a.intent_login,a.recovery_login)
           OR EXISTS (SELECT 1 FROM __SCHEMA__.cpu_cell_generations g WHERE g.login=session_user))
$$;

-- No runtime or PUBLIC table DML, sequence access, helper execution or schema DDL.
REVOKE ALL ON __SCHEMA__.cpu_cell_authority, __SCHEMA__.cpu_cell_generations, __SCHEMA__.cpu_cell_intents, __SCHEMA__.cpu_cell_attempts FROM PUBLIC;
DO $$ DECLARE f record; BEGIN
    FOR f IN SELECT p.oid::regprocedure AS signature FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
        WHERE n.nspname='__SCHEMA_NAME__' AND left(p.proname,9) = 'cpu_cell_'
    LOOP EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', f.signature); END LOOP;
END $$;
