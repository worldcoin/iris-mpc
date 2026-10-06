use clap::Parser;
use eyre::{ensure, Result, WrapErr};
use iris_mpc_common::postgres::{run_migrations, PostgresClient};
use iris_mpc_store::Store;
use sqlx::{postgres::PgPoolOptions, Executor};
use std::time::Duration;

const MAX_TARGET_SIZE: usize = 17_000_000;

/// Seed a fresh synthetic fixture schema, then exit without starting a server.
/// Interrupted loads leave their partial schema intact; retries require a new attempt name.
#[derive(Parser)]
struct Args {
    #[clap(long)]
    party_id: usize,
    /// Must be POP4401_17M_<attempt>_stage_<party>, with an ASCII alphanumeric attempt.
    #[clap(long)]
    db_schema: String,
    #[clap(long)]
    target_db_size: usize,
}

impl Args {
    fn validate(&self) -> Result<()> {
        ensure!(self.party_id < 3, "party-id must be 0, 1, or 2");
        ensure!(
            (1..=MAX_TARGET_SIZE).contains(&self.target_db_size),
            "target-db-size must be within 1..=17000000"
        );
        let suffix = format!("_stage_{}", self.party_id);
        let attempt = self
            .db_schema
            .strip_prefix("POP4401_17M_")
            .and_then(|rest| rest.strip_suffix(&suffix));
        ensure!(
            self.db_schema.len() <= 63
                && attempt.is_some_and(|value| {
                    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_alphanumeric())
                }),
            "db-schema must be POP4401_17M_<ASCII alphanumeric attempt>_stage_<party-id>, at most 63 bytes"
        );
        Ok(())
    }
}

fn require_empty(count: usize, max_serial_id: usize) -> Result<()> {
    ensure!(
        count == 0 && max_serial_id == 0,
        "fixture must be empty; found count={count}, max_serial_id={max_serial_id}"
    );
    Ok(())
}

async fn create_store(db_url: &str, schema: &str) -> Result<Store> {
    // Exclude public: migrations and unqualified iris writes must stay in this fixture.
    let connection_sql = format!(
        "SET search_path TO \"{schema}\"; SET statement_timeout = '60s'; SET lock_timeout = '10s';"
    );
    let pool = PgPoolOptions::new()
        .max_connections(2)
        .acquire_timeout(Duration::from_secs(30))
        .after_connect(move |connection, _| {
            let sql = connection_sql.clone();
            Box::pin(async move {
                connection.execute(sql.as_str()).await?;
                Ok(())
            })
        })
        .connect(db_url)
        .await
        .wrap_err("connecting to fixture database")?;

    // Atomic creation rejects previous attempts and competing loaders, even when empty.
    pool.execute(format!("CREATE SCHEMA \"{schema}\"").as_str())
        .await
        .wrap_err("creating fresh fixture schema; use a new attempt name if it already exists")?;
    run_migrations(&pool, false).await?;
    Store::new(&PostgresClient {
        pool,
        schema_name: schema.to_owned(),
    })
    .await
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    args.validate()?;
    let db_url = std::env::var("INIT_RANDOM_DB_URL")
        .wrap_err("INIT_RANDOM_DB_URL must contain the fixture database connection URL")?;
    tracing_subscriber::fmt().init();
    let store = create_store(&db_url, &args.db_schema).await?;
    require_empty(
        store.count_irises().await?,
        store.get_max_serial_id().await?,
    )?;
    store
        .init_db_with_random_shares(42, args.party_id, args.target_db_size, false)
        .await?;
    let count = store.count_irises().await?;
    let max_serial_id = store.get_max_serial_id().await?;
    ensure!(
        count == args.target_db_size && max_serial_id == args.target_db_size,
        "fixture verification failed: target={}, count={count}, max_serial_id={max_serial_id}",
        args.target_db_size
    );
    tracing::info!(schema = %args.db_schema, party = args.party_id, count, "Fixture initialization complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(schema: &str, party: usize, target: usize) -> Args {
        Args {
            party_id: party,
            db_schema: schema.to_owned(),
            target_db_size: target,
        }
    }

    #[test]
    fn only_matching_fixture_schemas_are_allowed() {
        for party in 0..3 {
            assert!(args(
                &format!("POP4401_17M_attempt1_stage_{party}"),
                party,
                17_000_000
            )
            .validate()
            .is_ok());
        }
        for schema in [
            "SMPC_stage_0",
            "public",
            "POP4401_17M_stage_0",
            "POP4401_17M__stage_0",
            "POP4401_17M_attempt_stage_1",
            "POP4401_17M_attempt\";DROP SCHEMA public;--_stage_0",
            "POP4401_17M_ä_stage_0",
        ] {
            assert!(args(schema, 0, 1).validate().is_err(), "{schema}");
        }
        let long_schema = format!("POP4401_17M_{}_stage_0", "a".repeat(64));
        assert!(args(&long_schema, 0, 1).validate().is_err());
    }

    #[test]
    fn generation_bounds_are_enforced_before_connection() {
        for (party, target) in [(3, 1), (usize::MAX, 1), (0, 0), (0, 17_000_001)] {
            assert!(args("POP4401_17M_attempt1_stage_0", party, target)
                .validate()
                .is_err());
        }
    }

    #[test]
    fn partial_or_inconsistent_fixtures_are_rejected() {
        assert!(require_empty(0, 0).is_ok());
        for (count, max) in [(1, 1), (0, 1), (1, 0), (271001, 271001)] {
            assert!(require_empty(count, max).is_err());
        }
    }

    #[tokio::test]
    #[ignore = "requires INIT_RANDOM_DB_TEST_URL pointing to a disposable local Postgres database"]
    async fn fresh_schema_is_isolated_and_previous_attempts_are_refused() -> Result<()> {
        use sqlx::postgres::PgConnectOptions;
        use std::str::FromStr;

        let url = std::env::var("INIT_RANDOM_DB_TEST_URL")?;
        let options = PgConnectOptions::from_str(&url)?;
        ensure!(
            ["localhost", "127.0.0.1", "::1"].contains(&options.get_host()),
            "integration test requires a disposable local database"
        );
        let control = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await?;
        // CREATE without IF NOT EXISTS protects any pre-existing public iris table.
        control
            .execute("CREATE TABLE public.irises (id bigint PRIMARY KEY); INSERT INTO public.irises VALUES (987654);")
            .await?;
        let attempt = uuid::Uuid::new_v4().simple().to_string();
        let mut fixtures = Vec::new();
        for party in 0..3 {
            let fresh = format!("POP4401_17M_{attempt}_stage_{party}");
            args(&fresh, party, 3).validate()?;
            let store = create_store(&url, &fresh).await?;
            require_empty(
                store.count_irises().await?,
                store.get_max_serial_id().await?,
            )?;
            store
                .init_db_with_random_shares(42, party, 3, false)
                .await?;
            assert_eq!(store.count_irises().await?, 3);
            assert_eq!(store.get_max_serial_id().await?, 3);
            assert!(create_store(&url, &fresh).await.is_err());
            fixtures.push(fresh);
        }

        let previous = format!("POP4401_17M_{attempt}previous_stage_0");
        control
            .execute(format!("CREATE SCHEMA \"{previous}\"").as_str())
            .await?;
        assert!(create_store(&url, &previous).await.is_err());
        let migrated: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = $1)",
        )
        .bind(&previous)
        .fetch_one(&control)
        .await?;
        assert!(
            !migrated,
            "an existing schema must be refused before migrations"
        );
        let sentinel: Vec<i64> = sqlx::query_scalar("SELECT id FROM public.irises")
            .fetch_all(&control)
            .await?;
        assert_eq!(sentinel, vec![987654]);
        for fixture in fixtures {
            control
                .execute(format!("DROP SCHEMA \"{fixture}\" CASCADE").as_str())
                .await?;
        }
        control
            .execute(format!("DROP SCHEMA \"{previous}\"; DROP TABLE public.irises;").as_str())
            .await?;
        Ok(())
    }
}
