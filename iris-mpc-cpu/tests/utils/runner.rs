use super::CpuConfigs;
use tokio_util::sync::CancellationToken;

/// Lifecycle trait implemented by each `wal_NNN` test struct.
///
/// Mirrors the `TestRun` trait from iris-mpc-upgrade-hawk/tests/utils/runner.rs.
///
/// `run()` calls all phases in order; only `exec` and `exec_assert` are required —
/// the rest default to `Ok(())`.
#[allow(async_fn_in_trait)]
pub trait TestRun {
    /// Orchestrates all lifecycle phases in order.
    /// Propagates the first error encountered; teardown runs even on exec failure.
    /// All errors are logged for visibility.
    async fn run(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let test_id = format!("{}:{}", ctx.kind, ctx.idx);

        // Capture the first failure from setup → exec → exec_assert without
        // short-circuiting out of `run`, so teardown always gets a chance to run.
        let setup_res = async {
            tracing::info!("[{}] Starting phase: setup", test_id);
            self.setup(ctx).await?;
            tracing::info!("[{}] Starting phase: setup_assert", test_id);
            self.setup_assert(ctx).await
        }
        .await;

        let phase_res = if let Err(ref e) = setup_res {
            tracing::error!("[{}] Setup error: {}", test_id, e);
            setup_res
        } else {
            let exec_res = async {
                tracing::info!("[{}] Starting phase: exec", test_id);
                self.exec(ctx).await?;
                tracing::info!("[{}] Starting phase: exec_assert", test_id);
                self.exec_assert(ctx).await
            }
            .await;

            if let Err(ref e) = exec_res {
                tracing::error!("[{}] Exec error: {}", test_id, e);
            }
            exec_res
        };

        // Teardown always runs, even if an earlier phase failed.
        let teardown_res = self.teardown(ctx).await;
        if let Err(ref e) = teardown_res {
            tracing::error!(
                "[{}] Teardown error (after phase failure={}): {}",
                test_id,
                phase_res.is_err(),
                e
            );
        }

        tracing::info!(
            "[{}] Starting phase: teardown_assert (phase_failed={})",
            test_id,
            phase_res.is_err()
        );
        let teardown_assert_res = self.teardown_assert(ctx).await;
        if let Err(ref e) = teardown_assert_res {
            tracing::error!(
                "[{}] Teardown assert error (after phase failure={}): {}",
                test_id,
                phase_res.is_err(),
                e
            );
        }

        // Surface errors in phase order: setup/exec first, then teardown.
        phase_res?;
        teardown_res?;
        teardown_assert_res
    }

    /// Prepare DB state: truncate tables, seed WAL mutations, seed checkpoint.
    async fn setup(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        Ok(())
    }

    /// Verify preconditions before running services (e.g. WAL row count = N).
    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        Ok(())
    }

    /// REQUIRED — Spawn services and wait for the termination condition.
    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()>;

    /// REQUIRED — Verify post-conditions: checkpoint rows, S3 objects, WAL HWM.
    async fn exec_assert(&mut self, ctx: &CpuTestContext) -> eyre::Result<()>;

    /// Cancel services, truncate tables, clean up S3 objects.
    async fn teardown(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        Ok(())
    }

    /// Optional final invariant check (runs after teardown).
    async fn teardown_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        Ok(())
    }
}

/// Context passed to every lifecycle method.
pub struct CpuTestContext {
    pub configs: CpuConfigs,
    pub env: TestEnvironment,
    /// Test number (100–104) for log tagging and config selection.
    pub kind: usize,
    /// Test run index (usually 1) for multi-run scenarios.
    pub idx: usize,
    /// Cancelled when Ctrl+C is received; services observe this to shut down cleanly.
    pub abort: CancellationToken,
}

impl CpuTestContext {
    pub fn new(kind: usize, idx: usize) -> Self {
        let env = if std::path::Path::new("/.dockerenv").exists() {
            TestEnvironment::Docker
        } else {
            TestEnvironment::Local
        };
        Self {
            configs: Self::load_configs(&env),
            env,
            kind,
            idx,
            abort: CancellationToken::new(),
        }
    }

    fn load_configs(env: &TestEnvironment) -> CpuConfigs {
        crate::utils::configs::hardcoded_configs(env)
    }
}

/// Execution environment — controls addresses and S3 endpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum TestEnvironment {
    /// Running locally; LocalStack at http://localhost:4566.
    Local,
    /// Running in Docker network; LocalStack at http://localstack:4566.
    Docker,
}

impl TestEnvironment {
    pub fn s3_endpoint(&self) -> &'static str {
        match self {
            Self::Local => "http://localhost:4566",
            Self::Docker => "http://localstack:4566",
        }
    }

    pub fn public_key_base_url(&self) -> &'static str {
        match self {
            Self::Local => "http://localhost:4566/wf-dev-public-keys",
            Self::Docker => "http://localstack:4566/wf-dev-public-keys",
        }
    }
}
