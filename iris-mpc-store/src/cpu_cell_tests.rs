use super::*;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use std::{str::FromStr, time::Duration};

const INC: &str = "10000000-0000-0000-0000-000000000001";
const SOURCE: &str = "20000000-0000-0000-0000-000000000001";
const DEST: &str = "20000000-0000-0000-0000-000000000002";
const FRESH: &str = "20000000-0000-0000-0000-000000000003";
const INTENT: &str = "30000000-0000-0000-0000-000000000001";
const ATTEMPT: &str = "40000000-0000-0000-0000-000000000001";
const OTHER_ATTEMPT: &str = "40000000-0000-0000-0000-000000000002";
const RECOVERY: &str = "50000000-0000-0000-0000-000000000001";
const W: &str = "0000000000000000000000000000000000000009";
const PROOF: [u8; 32] = [9; 32];

fn processes(prefix: u8) -> [String; 3] {
    std::array::from_fn(|party| format!("60000000-0000-0000-0000-0000000000{prefix}{party}"))
}
fn identity(slot: &str) -> GenerationIdentity {
    GenerationIdentity {
        slot: slot.into(),
        image: format!(
            "sha256:{}",
            if slot == "a" {
                "a".repeat(64)
            } else {
                "b".repeat(64)
            }
        ),
        schema_version: "1".into(),
        protocol: "1".into(),
        configuration: "config-digest".into(),
    }
}
fn binding(generation: &str, prefix: u8) -> ProcessBinding {
    ProcessBinding {
        incarnation: INC.into(),
        generation: generation.into(),
        process_id: processes(prefix)[0].clone(),
        secret: vec![prefix; 32],
    }
}

struct Fixture {
    admin: PgPool,
    schema: String,
    roles: Vec<String>,
    deploy: CpuCellAuthority,
    recovery: CpuCellAuthority,
    source: CpuCellAuthority,
    destination: CpuCellAuthority,
    fresh: CpuCellAuthority,
    a: ProcessBinding,
    b: ProcessBinding,
}
impl Fixture {
    async fn new() -> Result<Self> {
        let url = std::env::var("CPU_CELL_TEST_DATABASE_URL")
            .wrap_err("CPU_CELL_TEST_DATABASE_URL must point to a disposable PostgreSQL database with role provisioning rights")?;
        let admin = PgPoolOptions::new()
            .max_connections(3)
            .acquire_timeout(Duration::from_secs(5))
            .connect(&url)
            .await?;
        let schema = format!("cpu_test_{:x}", rand::random::<u64>());
        let roles: Vec<_> = ["owner", "deploy", "recovery", "source", "dest", "fresh"]
            .into_iter()
            .map(|r| format!("{schema}_{r}"))
            .collect();
        sqlx::raw_sql(&format!(
            "CREATE SCHEMA {}; CREATE TABLE {}.canonical(id bigint PRIMARY KEY); CREATE TABLE {}._sqlx_migrations(version bigint);",
            identifier(&schema),
            identifier(&schema),
            identifier(&schema)
        ))
        .execute(&admin)
        .await?;
        for (index, role) in roles.iter().enumerate() {
            let login = if index == 0 {
                "NOLOGIN"
            } else {
                "LOGIN PASSWORD 'local-test-only'"
            };
            sqlx::raw_sql(&format!("CREATE ROLE {} {login} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS", identifier(role)))
                .execute(&admin).await?;
        }
        bootstrap(
            &admin,
            &BootstrapConfig {
                schema: &schema,
                authority_owner: &roles[0],
                incarnation: INC,
                party_id: 0,
                intent_login: &roles[1],
                recovery_login: &roles[2],
            },
        )
        .await?;
        let mut clients = Vec::new();
        for role in roles.iter().skip(1) {
            let options = PgConnectOptions::from_str(&url)?
                .username(role)
                .password("local-test-only")
                .options([("statement_timeout", "3000"), ("lock_timeout", "2000")]);
            let pool = PgPoolOptions::new()
                .max_connections(3)
                .acquire_timeout(Duration::from_secs(5))
                .connect_with(options)
                .await?;
            clients.push(CpuCellAuthority::new(pool, &schema));
        }
        let fixture = Self {
            admin,
            schema,
            roles,
            deploy: clients[0].clone(),
            recovery: clients[1].clone(),
            source: clients[2].clone(),
            destination: clients[3].clone(),
            fresh: clients[4].clone(),
            a: binding(SOURCE, 1),
            b: binding(DEST, 2),
        };
        fixture
            .recovery
            .register_generation(INC, SOURCE, &fixture.roles[3], &identity("a"))
            .await?;
        fixture
            .deploy
            .register_generation(INC, DEST, &fixture.roles[4], &identity("b"))
            .await?;
        let revision = fixture.recovery.status().await?.revision;
        fixture
            .recovery
            .begin_recovery(INC, RECOVERY, revision)
            .await?;
        fixture
            .recovery
            .finish_recovery(
                INC,
                RECOVERY,
                SOURCE,
                "all previous processes externally fenced",
                INC,
            )
            .await?;
        fixture
            .source
            .bind_process(&fixture.a, &identity("a"))
            .await?;
        fixture
            .destination
            .bind_process(&fixture.b, &identity("b"))
            .await?;
        Ok(fixture)
    }
    async fn prepare(&self) -> Result<()> {
        self.deploy
            .write_intent(INC, INTENT, SOURCE, DEST, true)
            .await?;
        self.destination.mark_loaded(&self.b).await?;
        self.source
            .freeze_attempt(&self.a, ATTEMPT, INTENT, &processes(1), &processes(2))
            .await
    }
    async fn certificate(&self) -> Result<()> {
        self.prepare().await?;
        self.source.seal_source(&self.a, ATTEMPT).await?;
        self.destination
            .mark_reconciled(&self.b, ATTEMPT, Some(W), &PROOF)
            .await?;
        self.source
            .retire_source(&self.a, ATTEMPT, Some(W), &PROOF)
            .await?;
        self.source
            .record_retirement_certificate(&self.a, ATTEMPT, Some(W), &PROOF, &processes(1))
            .await
    }
    async fn insert(&self, actor: &CpuCellAuthority, id: i64) -> Result<()> {
        sqlx::query(&format!(
            "INSERT INTO {}.canonical VALUES($1)",
            identifier(&self.schema)
        ))
        .bind(id)
        .execute(&actor.pool)
        .await?;
        Ok(())
    }
    async fn cleanup(self) -> Result<()> {
        for actor in [
            &self.deploy,
            &self.recovery,
            &self.source,
            &self.destination,
            &self.fresh,
        ] {
            actor.pool.close().await;
        }
        sqlx::raw_sql(&format!("DROP SCHEMA {} CASCADE", identifier(&self.schema)))
            .execute(&self.admin)
            .await?;
        for role in &self.roles {
            sqlx::raw_sql(&format!("DROP ROLE IF EXISTS {}", identifier(role)))
                .execute(&self.admin)
                .await?;
        }
        self.admin.close().await;
        Ok(())
    }
}

#[tokio::test]
async fn permissions_and_one_lifetime_process_binding() -> Result<()> {
    let f = Fixture::new().await?;
    f.source.assert_owner(&f.a).await?;
    f.destination.assert_candidate(&f.b).await?;
    f.source.bind_process(&f.a, &identity("a")).await?;
    f.insert(&f.source, 1).await?;
    assert!(sqlx::query(&format!(
        "INSERT INTO {}._sqlx_migrations VALUES(1)",
        identifier(&f.schema)
    ))
    .execute(&f.source.pool)
    .await
    .is_err());
    sqlx::query("SET default_transaction_read_only=off")
        .execute(&f.destination.pool)
        .await?;
    assert!(f.insert(&f.destination, 2).await.is_err());
    assert!(f
        .destination
        .write_intent(INC, INTENT, SOURCE, DEST, true)
        .await
        .is_err());
    assert!(f
        .destination
        .begin_recovery(INC, OTHER_ATTEMPT, 0)
        .await
        .is_err());
    assert!(f
        .deploy
        .begin_recovery(INC, OTHER_ATTEMPT, 0)
        .await
        .is_err());
    assert!(f
        .recovery
        .write_intent(INC, INTENT, SOURCE, DEST, true)
        .await
        .is_err());
    for sql in [
        format!(
            "UPDATE {}.cpu_cell_authority SET recovery_blocked=false",
            identifier(&f.schema)
        ),
        format!("CREATE TABLE {}.escape(id int)", identifier(&f.schema)),
        format!(
            "SELECT {}.cpu_cell_canonical_writes(current_user,true)",
            identifier(&f.schema)
        ),
        format!("SET ROLE {}", identifier(&f.roles[0])),
    ] {
        assert!(
            sqlx::raw_sql(&sql)
                .execute(&f.destination.pool)
                .await
                .is_err(),
            "unexpected permission: {sql}"
        );
    }
    assert!(f.insert(&f.deploy, 3).await.is_err());
    assert!(f.insert(&f.recovery, 4).await.is_err());
    sqlx::raw_sql(&format!(
        "GRANT UPDATE(id) ON {}.canonical TO {}",
        identifier(&f.schema),
        identifier(&f.roles[5])
    ))
    .execute(&f.admin)
    .await?;
    assert!(f
        .deploy
        .register_generation(INC, FRESH, &f.roles[5], &identity("a"))
        .await
        .is_err());
    sqlx::raw_sql(&format!(
        "REVOKE UPDATE(id) ON {}.canonical FROM {}",
        identifier(&f.schema),
        identifier(&f.roles[5])
    ))
    .execute(&f.admin)
    .await?;
    let mut mutable_image = identity("a");
    mutable_image.image = "latest".into();
    assert!(f
        .deploy
        .register_generation(INC, FRESH, &f.roles[5], &mutable_image)
        .await
        .is_err());
    sqlx::raw_sql(&format!(
        "GRANT INSERT ON {}.canonical TO {}",
        identifier(&f.schema),
        identifier(&f.roles[5])
    ))
    .execute(&f.admin)
    .await?;
    assert!(f
        .deploy
        .register_generation(INC, FRESH, &f.roles[5], &identity("a"))
        .await
        .is_err());
    sqlx::raw_sql(&format!(
        "REVOKE INSERT ON {}.canonical FROM {}",
        identifier(&f.schema),
        identifier(&f.roles[5])
    ))
    .execute(&f.admin)
    .await?;
    let mut spoof = f.a.clone();
    spoof.secret = vec![99; 32];
    assert!(f.source.bind_process(&spoof, &identity("a")).await.is_err());
    assert!(f.source.assert_owner(&spoof).await.is_err());
    spoof = f.a.clone();
    spoof.process_id = processes(3)[0].clone();
    assert!(f.source.bind_process(&spoof, &identity("a")).await.is_err());
    assert!(f.destination.assert_owner(&f.a).await.is_err());
    assert!(f
        .deploy
        .register_generation(INC, FRESH, &f.roles[3], &identity("a"))
        .await
        .is_err());
    assert!(bootstrap(
        &f.admin,
        &BootstrapConfig {
            schema: &f.schema,
            authority_owner: &f.roles[0],
            incarnation: INC,
            party_id: 0,
            intent_login: &f.roles[1],
            recovery_login: &f.roles[2]
        }
    )
    .await
    .is_err());
    f.source.assert_owner(&f.a).await?;
    sqlx::raw_sql(&format!(
        "GRANT UPDATE ON {}.canonical TO {}",
        identifier(&f.schema),
        identifier(&f.roles[1])
    ))
    .execute(&f.admin)
    .await?;
    let error = bootstrap(
        &f.admin,
        &BootstrapConfig {
            schema: &f.schema,
            authority_owner: &f.roles[0],
            incarnation: INC,
            party_id: 0,
            intent_login: &f.roles[1],
            recovery_login: &f.roles[2],
        },
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("administrative login"));
    sqlx::raw_sql(&format!(
        "REVOKE UPDATE ON {}.canonical FROM {}",
        identifier(&f.schema),
        identifier(&f.roles[1])
    ))
    .execute(&f.admin)
    .await?;
    f.cleanup().await
}

#[tokio::test]
async fn serialized_handoff_preserves_settlement_and_requires_certificate() -> Result<()> {
    let f = Fixture::new().await?;
    f.deploy
        .write_intent(INC, INTENT, SOURCE, DEST, true)
        .await?;
    assert!(f
        .source
        .freeze_attempt(&f.a, ATTEMPT, INTENT, &processes(1), &processes(2))
        .await
        .is_err());
    f.destination.mark_loaded(&f.b).await?;
    let sources = processes(1);
    let destinations = processes(2);
    let (first, second) = tokio::join!(
        f.source
            .freeze_attempt(&f.a, ATTEMPT, INTENT, &sources, &destinations),
        f.source
            .freeze_attempt(&f.a, OTHER_ATTEMPT, INTENT, &sources, &destinations)
    );
    assert_ne!(first.is_ok(), second.is_ok());
    let attempt = if first.is_ok() {
        ATTEMPT
    } else {
        OTHER_ATTEMPT
    };
    assert!(f
        .destination
        .mark_reconciled(&f.b, attempt, Some(W), &PROOF)
        .await
        .is_err());
    f.source.seal_source(&f.a, attempt).await?;
    f.source.seal_source(&f.a, attempt).await?;
    assert!(f.source.assert_owner(&f.a).await.is_err());
    assert!(f.source.bind_process(&f.a, &identity("a")).await.is_err());
    f.source.assert_settling_owner(&f.a).await?;
    f.insert(&f.source, 1).await?; // Previously admitted canonical effects still settle.
    assert!(f
        .source
        .retire_source(&f.a, attempt, Some(W), &PROOF)
        .await
        .is_err());
    f.destination
        .mark_reconciled(&f.b, attempt, Some(W), &PROOF)
        .await?;
    assert!(f
        .source
        .retire_source(&f.a, attempt, Some("wrong-watermark"), &PROOF)
        .await
        .is_err());
    assert!(f
        .destination
        .retire_source(&f.b, attempt, Some(W), &PROOF)
        .await
        .is_err());
    f.source
        .retire_source(&f.a, attempt, Some(W), &PROOF)
        .await?;
    assert!(f.insert(&f.source, 2).await.is_err());
    assert!(f.source.assert_settling_owner(&f.a).await.is_err());
    assert!(f
        .destination
        .acquire(&f.b, attempt, Some(W), &PROOF)
        .await
        .is_err());
    assert!(f
        .destination
        .record_retirement_certificate(&f.b, attempt, Some(W), &PROOF, &sources)
        .await
        .is_err());
    assert!(f
        .source
        .record_retirement_certificate(&f.a, attempt, Some(W), &PROOF, &processes(3))
        .await
        .is_err());
    f.source
        .record_retirement_certificate(&f.a, attempt, Some(W), &PROOF, &sources)
        .await?;
    let (first, second) = tokio::join!(
        f.destination.acquire(&f.b, attempt, Some(W), &PROOF),
        f.destination.acquire(&f.b, attempt, Some(W), &PROOF)
    );
    first?;
    second?; // Same exact process retry is idempotent.
    f.destination.assert_owner(&f.b).await?;
    assert!(f.source.assert_owner(&f.a).await.is_err());
    assert!(f.source.bind_process(&f.a, &identity("a")).await.is_err());
    f.insert(&f.destination, 2).await?;
    f.cleanup().await
}

#[tokio::test]
async fn recovery_supersedes_acquisition_and_requires_fresh_fenced_owner() -> Result<()> {
    let f = Fixture::new().await?;
    f.certificate().await?;
    f.destination
        .acquire(&f.b, ATTEMPT, Some(W), &PROOF)
        .await?;
    f.source.pool.close().await;
    let mut cleanup = f.admin.begin().await?;
    sqlx::raw_sql(&format!(
        "SET LOCAL ROLE {}; REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM {}; \
         REVOKE ALL ON ALL FUNCTIONS IN SCHEMA {} FROM {}; REVOKE USAGE ON SCHEMA {} FROM {}; \
         RESET ROLE; DROP OWNED BY {}; DROP ROLE {}",
        identifier(&f.roles[0]),
        identifier(&f.schema),
        identifier(&f.roles[3]),
        identifier(&f.schema),
        identifier(&f.roles[3]),
        identifier(&f.schema),
        identifier(&f.roles[3]),
        identifier(&f.roles[3]),
        identifier(&f.roles[3])
    ))
    .execute(&mut *cleanup)
    .await?;
    cleanup.commit().await?;
    let recovery = "50000000-0000-0000-0000-000000000002";
    let revision = f.recovery.status().await?.revision;
    f.recovery.begin_recovery(INC, recovery, revision).await?;
    assert!(f
        .destination
        .acquire(&f.b, ATTEMPT, Some(W), &PROOF)
        .await
        .is_err());
    assert!(f.destination.assert_owner(&f.b).await.is_err());
    assert!(f.insert(&f.destination, 1).await.is_err());
    assert!(f
        .recovery
        .finish_recovery(INC, recovery, DEST, "fenced", INC)
        .await
        .is_err());
    f.recovery
        .register_generation(INC, FRESH, &f.roles[5], &identity("a"))
        .await?;
    assert!(f
        .recovery
        .finish_recovery(INC, recovery, FRESH, "", INC)
        .await
        .is_err());
    let fresh_incarnation = "10000000-0000-0000-0000-000000000002";
    f.recovery
        .finish_recovery(
            INC,
            recovery,
            FRESH,
            "all A/B processes externally fenced",
            fresh_incarnation,
        )
        .await?;
    let mut c = binding(FRESH, 3);
    c.incarnation = fresh_incarnation.into();
    f.fresh.bind_process(&c, &identity("a")).await?;
    f.fresh.assert_owner(&c).await?;
    assert!(f
        .destination
        .acquire(&f.b, ATTEMPT, Some(W), &PROOF)
        .await
        .is_err());
    let mut stale = f.b.clone();
    stale.incarnation = fresh_incarnation.into();
    assert!(f
        .destination
        .acquire(&stale, ATTEMPT, Some(W), &PROOF)
        .await
        .is_err());
    let rows: i64 = sqlx::query_scalar(&format!(
        "SELECT count(*) FROM {}.cpu_cell_generations WHERE sealed AND retired",
        identifier(&f.schema)
    ))
    .fetch_one(&f.admin)
    .await?;
    assert_eq!(rows, 2);
    f.fresh.pool.close().await;
    assert!(f.fresh.assert_owner(&c).await.is_err()); // Database unavailable never implies ownership.
    f.cleanup().await
}

#[tokio::test]
async fn empty_frontier_and_repeated_slot_reuse_preserve_history() -> Result<()> {
    let f = Fixture::new().await?;
    f.certificate().await?;
    f.destination
        .acquire(&f.b, ATTEMPT, Some(W), &PROOF)
        .await?;
    f.deploy
        .register_generation(INC, FRESH, &f.roles[5], &identity("a"))
        .await?;
    let c = binding(FRESH, 3);
    f.fresh.bind_process(&c, &identity("a")).await?;
    f.fresh.mark_loaded(&c).await?;
    let second_intent = "30000000-0000-0000-0000-000000000002";
    f.deploy
        .write_intent(INC, second_intent, DEST, FRESH, true)
        .await?;
    f.destination
        .freeze_attempt(
            &f.b,
            OTHER_ATTEMPT,
            second_intent,
            &processes(2),
            &processes(3),
        )
        .await?;
    assert!(f
        .destination
        .acquire(&f.b, ATTEMPT, Some(W), &PROOF)
        .await
        .is_err());
    f.destination.seal_source(&f.b, OTHER_ATTEMPT).await?;
    assert!(f
        .fresh
        .mark_reconciled(&c, OTHER_ATTEMPT, Some("bad"), &PROOF)
        .await
        .is_err());
    f.fresh
        .mark_reconciled(&c, OTHER_ATTEMPT, None, &PROOF)
        .await?;
    f.destination
        .retire_source(&f.b, OTHER_ATTEMPT, None, &PROOF)
        .await?;
    f.destination
        .record_retirement_certificate(&f.b, OTHER_ATTEMPT, None, &PROOF, &processes(2))
        .await?;
    f.fresh.acquire(&c, OTHER_ATTEMPT, None, &PROOF).await?;
    f.fresh.assert_owner(&c).await?;
    f.insert(&f.fresh, 1).await?;
    let history:bool=sqlx::query_scalar(&format!("SELECT source_retired AND retirement_certificate AND acquired AND superseded FROM {}.cpu_cell_inspect($1::uuid,$2::uuid)",identifier(&f.schema)))
        .bind(INC).bind(ATTEMPT).fetch_one(&f.fresh.pool).await?;
    assert!(history);
    let historical = f.fresh.inspect(INC, ATTEMPT).await?;
    assert!(historical.source_retired && historical.superseded);
    assert_eq!(historical.source_processes, processes(1));
    f.cleanup().await
}

#[tokio::test]
async fn ordinary_crash_recovery_cannot_replay_an_older_begin() -> Result<()> {
    let f = Fixture::new().await?;
    let first_revision = f.recovery.status().await?.revision;
    assert!(f.recovery.status().await?.current_attempt.is_none());
    f.recovery
        .register_generation(INC, FRESH, &f.roles[5], &identity("a"))
        .await?;
    let recovery = "50000000-0000-0000-0000-000000000002";
    assert!(f
        .recovery
        .begin_recovery(INC, recovery, first_revision)
        .await
        .is_err());
    let revision = f.recovery.status().await?.revision;
    f.recovery.begin_recovery(INC, recovery, revision).await?;
    f.recovery.begin_recovery(INC, recovery, revision).await?;
    f.recovery
        .finish_recovery(INC, recovery, FRESH, "ordinary crashed owners fenced", INC)
        .await?;
    let c = binding(FRESH, 3);
    f.fresh.bind_process(&c, &identity("a")).await?;
    assert!(f.recovery.begin_recovery(INC, RECOVERY, 2).await.is_err());
    assert!(f
        .recovery
        .begin_recovery(INC, recovery, revision)
        .await
        .is_err());
    f.fresh.assert_owner(&c).await?;
    let status = f.recovery.status().await?;
    assert_eq!(status.recovery_id.as_deref(), Some(recovery));
    assert_eq!(
        status.fencing_ack.as_deref(),
        Some("ordinary crashed owners fenced")
    );
    assert_eq!(status.current_owner.as_deref(), Some(FRESH));
    assert!(!status.recovery_blocked);
    f.cleanup().await
}
