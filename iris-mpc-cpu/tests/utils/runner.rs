use super::CpuConfigs;

/// Lifecycle trait implemented by each `wal_NNN` test struct.
///
/// Mirrors the `TestRun` trait from iris-mpc-upgrade-hawk/tests/utils/runner.rs.
/// All methods are async and return `eyre::Result<()>`.
///
/// Execution order:
///   setup → setup_assert → exec → exec_assert → teardown
#[allow(async_fn_in_trait)]
pub trait TestRun {
    /// Prepare DB state: seed WAL mutations, seed base checkpoint, etc.
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()>;

    /// Verify preconditions before running services (e.g. WAL row count = N).
    async fn setup_assert(&self, ctx: &CpuTestContext) -> eyre::Result<()>;

    /// Spawn services (hawk_main / sidecar_main) and wait for the appropriate
    /// termination condition (TC-1 ready endpoint or TC-2 S3 checkpoint).
    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()>;

    /// Verify post-conditions: checkpoint rows, S3 objects, WAL high-water mark.
    async fn exec_assert(&self, ctx: &CpuTestContext) -> eyre::Result<()>;

    /// Cancel services, delete S3 checkpoint objects, truncate WAL table.
    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()>;
}

/// Context passed to every lifecycle method.
pub struct CpuTestContext {
    pub configs: CpuConfigs,
    pub env: TestEnvironment,
    /// Test number (100, 101, ...) for log tagging.
    pub kind: usize,
}

impl CpuTestContext {
    /// Detect environment from the presence of `/.dockerenv`.
    pub fn from_env(kind: usize) -> Self {
        let env = if std::path::Path::new("/.dockerenv").exists() {
            TestEnvironment::Docker
        } else {
            TestEnvironment::Local
        };
        Self {
            configs: Self::load_configs(&env),
            env,
            kind,
        }
    }

    fn load_configs(env: &TestEnvironment) -> CpuConfigs {
        // TODO: load from tests/resources/node-config/{local,docker}/ TOML files,
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
