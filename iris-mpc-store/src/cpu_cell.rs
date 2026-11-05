//! Party-local CPU ownership authority. Install before enabling participating
//! binaries, using separate migration credentials. Peer authentication and the
//! exact-process activation barrier are prerequisites for deployment enablement.
use eyre::{ensure, Result, WrapErr};
use sqlx::PgPool;

const SCHEMA_SQL: &str = include_str!("../../scripts/cpu-cell-ownership.sql");

fn identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

/// Credentials for exactly one process lifetime. Never log or persist `secret`.
/// A restart must use a new process ID and secret and obtain authorized recovery.
#[derive(Clone)]
pub struct ProcessBinding {
    pub incarnation: String,
    pub generation: String,
    pub process_id: String,
    pub secret: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct GenerationIdentity {
    pub slot: String,
    pub image: String,
    pub schema_version: String,
    pub protocol: String,
    pub configuration: String,
}

/// Bootstrap is deliberately separate from normal Store/migration construction.
/// Provision roles beforehand; the authority owner must be NOLOGIN, without
/// privileged attributes or role memberships. Runtime roles are generation-unique.
pub struct BootstrapConfig<'a> {
    pub schema: &'a str,
    pub authority_owner: &'a str,
    pub incarnation: &'a str,
    pub party_id: i32,
    pub intent_login: &'a str,
    pub recovery_login: &'a str,
}

/// Install once, atomically, with PUBLIC function execution revoked before commit.
/// Requires a provisioner able to grant the owner canonical-table privileges and
/// SET ROLE to it. Existing runtime credentials must be fenced before bootstrap.
pub async fn bootstrap(pool: &PgPool, config: &BootstrapConfig<'_>) -> Result<()> {
    ensure!(
        !config.schema.is_empty() && config.schema.len() <= 63
            && config.schema.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
        "CPU authority schema must be a PostgreSQL identifier containing only ASCII letters, digits, underscores or hyphens"
    );
    ensure!(
        config.authority_owner != config.intent_login
            && config.authority_owner != config.recovery_login
            && config.intent_login != config.recovery_login,
        "CPU administrative roles must differ"
    );
    for role in [
        config.authority_owner,
        config.intent_login,
        config.recovery_login,
    ] {
        ensure!(
            !role.is_empty() && role.len() <= 63 && !role.contains('\0'),
            "CPU role names must be nonempty PostgreSQL identifiers of at most 63 bytes"
        );
    }
    let schema = identifier(config.schema);
    let owner = identifier(config.authority_owner);
    let intent = identifier(config.intent_login);
    let recovery = identifier(config.recovery_login);
    let mut tx = pool.begin().await?;
    sqlx::query("SET LOCAL lock_timeout = '5s';")
        .execute(&mut *tx)
        .await?;
    sqlx::query("SET LOCAL statement_timeout = '30s';")
        .execute(&mut *tx)
        .await?;
    let valid_owner: bool = sqlx::query_scalar(
        "SELECT NOT (rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls) \
         AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE member = r.oid) \
         FROM pg_catalog.pg_roles r WHERE rolname=$1",
    )
    .bind(config.authority_owner)
    .fetch_one(&mut *tx)
    .await?;
    ensure!(
        valid_owner,
        "CPU authority owner must be an unprivileged NOLOGIN role"
    );
    let has_default_grants: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_default_acl d JOIN pg_catalog.pg_roles r ON r.oid=d.defaclrole WHERE r.rolname=$1)"
    ).bind(config.authority_owner).fetch_one(&mut *tx).await?;
    ensure!(
        !has_default_grants,
        "CPU authority owner must not have custom default grants"
    );
    for role in [config.intent_login, config.recovery_login] {
        let safe: bool = sqlx::query_scalar(
            "SELECT rolcanlogin AND NOT (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls) \
             AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE member = r.oid) \
             AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspowner = r.oid) \
             AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relowner = r.oid) \
             AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_proc WHERE proowner = r.oid) \
             AND NOT has_database_privilege(r.oid, current_database(), 'CREATE') \
             AND NOT has_schema_privilege(r.oid, $2, 'CREATE') \
             AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace \
                 WHERE n.nspname=$2 AND c.relkind IN ('r','p','v','f') \
                   AND has_table_privilege(r.oid,c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')) \
             AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace \
                 WHERE n.nspname=$2 AND CASE WHEN c.relkind='S' THEN has_sequence_privilege(r.oid,c.oid,'USAGE,UPDATE') ELSE false END) \
             AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid \
                 JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace CROSS JOIN LATERAL pg_catalog.aclexplode(a.attacl) acl \
                 WHERE n.nspname=$2 AND acl.grantee IN (0,r.oid) AND acl.privilege_type IN ('INSERT','UPDATE','REFERENCES')) \
             FROM pg_catalog.pg_roles r WHERE rolname=$1",
        ).bind(role).bind(config.schema).fetch_one(&mut *tx).await?;
        ensure!(
            safe,
            "CPU administrative login must be an unprivileged dedicated role"
        );
    }
    sqlx::raw_sql(&format!(
        "REVOKE CREATE ON SCHEMA {schema} FROM PUBLIC; \
         REVOKE ALL ON ALL TABLES IN SCHEMA {schema} FROM PUBLIC; \
         REVOKE ALL ON ALL SEQUENCES IN SCHEMA {schema} FROM PUBLIC; \
         GRANT USAGE, CREATE ON SCHEMA {schema} TO {owner} WITH GRANT OPTION; \
         GRANT ALL ON ALL TABLES IN SCHEMA {schema} TO {owner} WITH GRANT OPTION; \
         GRANT ALL ON ALL SEQUENCES IN SCHEMA {schema} TO {owner} WITH GRANT OPTION; \
         SET LOCAL ROLE {owner};"
    ))
    .execute(&mut *tx)
    .await?;
    // Substitute identifier and literal placeholders in one pass; inserted names
    // are never interpreted as template syntax.
    let sql = SCHEMA_SQL
        .split("__SCHEMA_NAME__")
        .map(|part| part.replace("__SCHEMA__", &schema))
        .collect::<Vec<_>>()
        .join(&config.schema.replace(char::from(39), "''"));
    sqlx::raw_sql(&sql)
        .execute(&mut *tx)
        .await
        .wrap_err("install CPU authority schema")?;
    sqlx::query(&format!(
        "INSERT INTO {schema}.cpu_cell_authority(incarnation,party_id,intent_login,recovery_login) VALUES($1::uuid,$2,$3::name,$4::name)"
    ))
    .bind(config.incarnation)
    .bind(config.party_id)
    .bind(config.intent_login)
    .bind(config.recovery_login)
    .execute(&mut *tx)
    .await?;
    sqlx::raw_sql(&format!(
        "GRANT USAGE ON SCHEMA {schema} TO {intent}, {recovery}; \
         GRANT EXECUTE ON FUNCTION {schema}.cpu_cell_status(), {schema}.cpu_cell_inspect(uuid,uuid) TO {intent}, {recovery}; \
         GRANT EXECUTE ON FUNCTION {schema}.cpu_cell_register_generation(uuid,uuid,text,name,text,text,text,text) TO {intent}, {recovery}; \
         GRANT EXECUTE ON FUNCTION {schema}.cpu_cell_write_intent(uuid,uuid,uuid,uuid,boolean) TO {intent}; \
         GRANT EXECUTE ON FUNCTION {schema}.cpu_cell_begin_recovery(uuid,uuid,bigint), {schema}.cpu_cell_finish_recovery(uuid,uuid,uuid,text,uuid) TO {recovery};"
    ))
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(())
}

/// Persisted local evidence, including whether recovery or a later deployment
/// superseded it. Possessing this record does not authorize acquisition.
#[derive(Debug, sqlx::FromRow)]
pub struct AttemptEvidence {
    pub incarnation: String,
    pub revision: i64,
    pub current_owner: Option<String>,
    pub current_attempt: Option<String>,
    pub recovery_blocked: bool,
    pub attempt: String,
    pub intent: String,
    pub source_generation: String,
    pub destination_generation: String,
    pub source_processes: Vec<String>,
    pub destination_processes: Vec<String>,
    pub watermark: Option<String>,
    pub proof: Option<Vec<u8>>,
    pub source_retired: bool,
    pub retirement_certificate: bool,
    pub acquired: bool,
    pub superseded: bool,
}

/// Secret-free administrative readback, including ordinary crashes with no attempt.
#[derive(Debug, sqlx::FromRow)]
pub struct AuthorityStatus {
    pub incarnation: String,
    pub revision: i64,
    pub party_id: i32,
    pub current_owner: Option<String>,
    pub current_process: Option<String>,
    pub current_attempt: Option<String>,
    pub recovery_blocked: bool,
    pub recovery_id: Option<String>,
    pub fencing_ack: Option<String>,
}

#[derive(Clone)]
pub struct CpuCellAuthority {
    pool: PgPool,
    schema: String,
}

impl CpuCellAuthority {
    /// Use a separate pool without read-only session defaults, with the same
    /// generation login as the data pool. SQL permissions enforce passive reads.
    pub fn new(pool: PgPool, schema: &str) -> Self {
        Self {
            pool,
            schema: identifier(schema),
        }
    }

    pub async fn status(&self) -> Result<AuthorityStatus> {
        sqlx::query_as(&format!("SELECT incarnation::text, revision, party_id, current_owner::text, current_process::text, current_attempt::text, recovery_blocked, recovery_id::text, fencing_ack FROM {}.cpu_cell_status()", self.schema))
            .fetch_one(&self.pool).await.wrap_err("read CPU authority status")
    }

    pub async fn inspect(&self, incarnation: &str, attempt: &str) -> Result<AttemptEvidence> {
        sqlx::query_as(&format!("SELECT incarnation::text, revision, current_owner::text, current_attempt::text, recovery_blocked, attempt::text, intent::text, source_generation::text, destination_generation::text, source_processes::text[], destination_processes::text[], watermark, proof, source_retired, retirement_certificate, acquired, superseded FROM {}.cpu_cell_inspect($1::uuid,$2::uuid)", self.schema))
            .bind(incarnation).bind(attempt).fetch_one(&self.pool).await.wrap_err("inspect CPU authority evidence")
    }

    pub async fn bind_process(
        &self,
        b: &ProcessBinding,
        identity: &GenerationIdentity,
    ) -> Result<()> {
        sqlx::query(&format!(
            "SELECT {}.cpu_cell_bind_process($1::uuid,$2::uuid,$3::uuid,$4,$5,$6,$7,$8,$9)",
            self.schema
        ))
        .bind(&b.incarnation)
        .bind(&b.generation)
        .bind(&b.process_id)
        .bind(&b.secret)
        .bind(&identity.slot)
        .bind(&identity.image)
        .bind(&identity.schema_version)
        .bind(&identity.protocol)
        .bind(&identity.configuration)
        .execute(&self.pool)
        .await
        .wrap_err("bind CPU process")?;
        Ok(())
    }

    async fn binding_operation(&self, name: &str, b: &ProcessBinding) -> Result<()> {
        sqlx::query(&format!(
            "SELECT {}.{name}($1::uuid,$2::uuid,$3::uuid,$4)",
            self.schema
        ))
        .bind(&b.incarnation)
        .bind(&b.generation)
        .bind(&b.process_id)
        .bind(&b.secret)
        .execute(&self.pool)
        .await
        .wrap_err("check CPU authority")?;
        Ok(())
    }

    pub async fn assert_owner(&self, b: &ProcessBinding) -> Result<()> {
        self.binding_operation("cpu_cell_assert_owner", b).await
    }
    pub async fn assert_settling_owner(&self, b: &ProcessBinding) -> Result<()> {
        self.binding_operation("cpu_cell_assert_settling_owner", b)
            .await
    }
    pub async fn assert_candidate(&self, b: &ProcessBinding) -> Result<()> {
        self.binding_operation("cpu_cell_assert_candidate", b).await
    }
    pub async fn mark_loaded(&self, b: &ProcessBinding) -> Result<()> {
        self.binding_operation("cpu_cell_mark_loaded", b).await
    }

    pub async fn seal_source(&self, b: &ProcessBinding, attempt: &str) -> Result<()> {
        sqlx::query(&format!(
            "SELECT {}.cpu_cell_seal_source($1::uuid,$2::uuid,$3::uuid,$4,$5::uuid)",
            self.schema
        ))
        .bind(&b.incarnation)
        .bind(&b.generation)
        .bind(&b.process_id)
        .bind(&b.secret)
        .bind(attempt)
        .execute(&self.pool)
        .await
        .wrap_err("durably seal CPU source")?;
        Ok(())
    }

    pub async fn freeze_attempt(
        &self,
        b: &ProcessBinding,
        attempt: &str,
        intent: &str,
        sources: &[String; 3],
        destinations: &[String; 3],
    ) -> Result<()> {
        sqlx::query(&format!("SELECT {}.cpu_cell_freeze_attempt($1::uuid,$2::uuid,$3::uuid,$4,$5::uuid,$6::uuid,$7::uuid[],$8::uuid[])", self.schema))
            .bind(&b.incarnation).bind(&b.generation).bind(&b.process_id).bind(&b.secret)
            .bind(attempt).bind(intent).bind(sources.as_slice()).bind(destinations.as_slice())
            .execute(&self.pool).await.wrap_err("freeze CPU exact cohorts")?;
        Ok(())
    }

    async fn proof_operation(
        &self,
        name: &str,
        b: &ProcessBinding,
        attempt: &str,
        watermark: Option<&str>,
        proof: &[u8],
    ) -> Result<()> {
        sqlx::query(&format!(
            "SELECT {}.{name}($1::uuid,$2::uuid,$3::uuid,$4,$5::uuid,$6,$7)",
            self.schema
        ))
        .bind(&b.incarnation)
        .bind(&b.generation)
        .bind(&b.process_id)
        .bind(&b.secret)
        .bind(attempt)
        .bind(watermark)
        .bind(proof)
        .execute(&self.pool)
        .await
        .wrap_err("persist CPU handoff proof")?;
        Ok(())
    }

    pub async fn mark_reconciled(
        &self,
        b: &ProcessBinding,
        attempt: &str,
        watermark: Option<&str>,
        proof: &[u8],
    ) -> Result<()> {
        self.proof_operation("cpu_cell_mark_reconciled", b, attempt, watermark, proof)
            .await
    }
    pub async fn retire_source(
        &self,
        b: &ProcessBinding,
        attempt: &str,
        watermark: Option<&str>,
        proof: &[u8],
    ) -> Result<()> {
        self.proof_operation("cpu_cell_retire_source", b, attempt, watermark, proof)
            .await
    }
    pub async fn acquire(
        &self,
        b: &ProcessBinding,
        attempt: &str,
        watermark: Option<&str>,
        proof: &[u8],
    ) -> Result<()> {
        self.proof_operation("cpu_cell_acquire", b, attempt, watermark, proof)
            .await
    }
    /// Only the bound source may attest after authenticating all three durable
    /// retirement statements. This method does not implement peer authentication.
    pub async fn record_retirement_certificate(
        &self,
        b: &ProcessBinding,
        attempt: &str,
        watermark: Option<&str>,
        proof: &[u8],
        sources: &[String; 3],
    ) -> Result<()> {
        sqlx::query(&format!("SELECT {}.cpu_cell_record_retirement_certificate($1::uuid,$2::uuid,$3::uuid,$4,$5::uuid,$6,$7,$8::uuid[])", self.schema))
            .bind(&b.incarnation).bind(&b.generation).bind(&b.process_id).bind(&b.secret)
            .bind(attempt).bind(watermark).bind(proof).bind(sources.as_slice())
            .execute(&self.pool).await.wrap_err("record CPU retirement certificate")?;
        Ok(())
    }

    pub async fn register_generation(
        &self,
        incarnation: &str,
        generation: &str,
        login: &str,
        identity: &GenerationIdentity,
    ) -> Result<()> {
        ensure!(
            !login.is_empty() && login.len() <= 63 && !login.contains('\0'),
            "CPU runtime login must be a PostgreSQL identifier of at most 63 bytes"
        );
        sqlx::query(&format!(
            "SELECT {}.cpu_cell_register_generation($1::uuid,$2::uuid,$3,$4::name,$5,$6,$7,$8)",
            self.schema
        ))
        .bind(incarnation)
        .bind(generation)
        .bind(&identity.slot)
        .bind(login)
        .bind(&identity.image)
        .bind(&identity.schema_version)
        .bind(&identity.protocol)
        .bind(&identity.configuration)
        .execute(&self.pool)
        .await
        .wrap_err("register CPU generation")?;
        Ok(())
    }
    pub async fn write_intent(
        &self,
        incarnation: &str,
        intent: &str,
        source: &str,
        destination: &str,
        automatic: bool,
    ) -> Result<()> {
        sqlx::query(&format!(
            "SELECT {}.cpu_cell_write_intent($1::uuid,$2::uuid,$3::uuid,$4::uuid,$5)",
            self.schema
        ))
        .bind(incarnation)
        .bind(intent)
        .bind(source)
        .bind(destination)
        .bind(automatic)
        .execute(&self.pool)
        .await
        .wrap_err("write CPU deployment intent")?;
        Ok(())
    }
    pub async fn begin_recovery(
        &self,
        incarnation: &str,
        recovery: &str,
        expected_revision: i64,
    ) -> Result<()> {
        sqlx::query(&format!(
            "SELECT {}.cpu_cell_begin_recovery($1::uuid,$2::uuid,$3)",
            self.schema
        ))
        .bind(incarnation)
        .bind(recovery)
        .bind(expected_revision)
        .execute(&self.pool)
        .await
        .wrap_err("supersede CPU authority for recovery")?;
        Ok(())
    }
    /// The acknowledgment records an external fact, never proves fencing itself.
    /// After database restore, supply an externally fresh incarnation: database
    /// state alone cannot detect rollback of its own authority history.
    pub async fn finish_recovery(
        &self,
        incarnation: &str,
        recovery: &str,
        fresh_generation: &str,
        fencing_ack: &str,
        new_incarnation: &str,
    ) -> Result<()> {
        sqlx::query(&format!(
            "SELECT {}.cpu_cell_finish_recovery($1::uuid,$2::uuid,$3::uuid,$4,$5::uuid)",
            self.schema
        ))
        .bind(incarnation)
        .bind(recovery)
        .bind(fresh_generation)
        .bind(fencing_ack)
        .bind(new_incarnation)
        .execute(&self.pool)
        .await
        .wrap_err("finish fenced CPU recovery")?;
        Ok(())
    }
}

#[cfg(all(test, feature = "db_dependent"))]
#[path = "cpu_cell_tests.rs"]
mod tests;
