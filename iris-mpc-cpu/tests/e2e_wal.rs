// Integration tests for the iris-mpc-cpu WAL pipeline.
//
// See tests/e2e_wal_readme.md for full design documentation.
//
// Run with:
//   cargo test --test e2e_wal -- --nocapture
//
// Requires:
//   - PostgreSQL running (via docker-compose) with per-party schemas
//   - LocalStack at http://localhost:4566

mod utils;
mod workflows;

// TODO: wire up a single serial runner (see open question #10 in readme —
// decide between one #[tokio::test] iterating all cases vs separate functions).
//
// The genesis test pattern uses a single test function that calls each workflow
// in sequence so that DB state is deterministically ordered.  Example:
//
// #[tokio::test]
// async fn e2e_wal_tests() -> eyre::Result<()> {
//     let ctx = utils::runner::CpuTestContext::from_env();
//     workflows::run_wal_100(&ctx).await?;
//     workflows::run_wal_101(&ctx).await?;
//     workflows::run_wal_102(&ctx).await?;
//     workflows::run_wal_103(&ctx).await?;
//     workflows::run_wal_104(&ctx).await?;
//     Ok(())
// }
