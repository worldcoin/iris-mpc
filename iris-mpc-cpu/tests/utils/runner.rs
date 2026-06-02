use super::CpuConfigs;

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
    async fn run(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        self.setup(ctx).await?;
        self.setup_assert(ctx).await?;
        let exec_res = self.exec(ctx).await;
        // Always attempt teardown regardless of exec outcome.
        let teardown_res = self.teardown(ctx).await;
        exec_res?;
        teardown_res?;
        self.exec_assert(ctx).await?;
        self.teardown_assert(ctx).await
    }

    /// Prepare DB state: truncate tables, seed WAL mutations, seed checkpoint.
    async fn setup(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        Ok(())
    }

    /// Verify preconditions before running services (e.g. WAL row count = N).
    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        Ok(())
    }

    /// REQUIRED — Spawn services, wait for termination condition (TC-1 or TC-2).
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
        }
    }

    fn load_configs(env: &TestEnvironment) -> CpuConfigs {
        // TODO (open question #1): load per-party CpuNodeConfig from TOML files at
        //   tests/resources/node-config/{local,docker}/
        // mirroring the genesis test config-loading pattern.
        let _ = env;
        todo!("load per-party CpuNodeConfig from TOML files")
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
}
