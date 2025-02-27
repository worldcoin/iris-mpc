use eyre::Result;
use iris_mpc_common::test::{load_test_db, TestCase, TestCaseGenerator};
use iris_mpc_cpu::execution::hawk_main::{HawkActor, HawkArgs, HawkHandle};
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const DB_SIZE: usize = 8 * 1000;
const DB_RNG_SEED: u64 = 0xdeadbeef;
const INTERNAL_RNG_SEED: u64 = 0xdeadbeef;
const NUM_BATCHES: usize = 30;
const MAX_BATCH_SIZE: usize = 64;
const HAWK_REQUEST_PARALLELISM: usize = 1;
const MAX_DELETIONS_PER_BATCH: usize = 0; // TODO: set back to 10 or so once deletions are supported

fn install_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

async fn start_hawk_node(args: &HawkArgs) -> Result<HawkHandle> {
    tracing::info!("🦅 Starting Hawk node {}", args.party_index);
    let mut hawk_actor = HawkActor::from_cli(args).await?;
    {
        let (mut iris_loader, _graph_loader) = hawk_actor.as_iris_loader().await;
        load_test_db(args.party_index, DB_SIZE, DB_RNG_SEED, &mut iris_loader)?;
        // TODO: built the graph for the test db...
    }

    let handle = HawkHandle::new(hawk_actor, args.request_parallelism).await?;
    Ok(handle)
}

#[ignore = "Expected to fail for now"]
#[tokio::test]
async fn e2e_test() -> Result<()> {
    install_tracing();

    let addresses = ["127.0.0.1:16000", "127.0.0.1:16100", "127.0.0.1:16200"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let args0 = HawkArgs {
        party_index: 0,
        addresses,
        request_parallelism: HAWK_REQUEST_PARALLELISM,
        disable_persistence: false,
    };
    let args1 = HawkArgs {
        party_index: 1,
        ..args0.clone()
    };
    let args2 = HawkArgs {
        party_index: 2,
        ..args0.clone()
    };
    let (handle0, handle1, handle2) = tokio::join!(
        start_hawk_node(&args0),
        start_hawk_node(&args1),
        start_hawk_node(&args2)
    );
    let mut handle0 = handle0?;
    let mut handle1 = handle1?;
    let mut handle2 = handle2?;

    let mut test_case_generator =
        TestCaseGenerator::new_seeded(DB_SIZE, DB_RNG_SEED, INTERNAL_RNG_SEED);

    // Disable test cases that are not yet supported
    // TODO: enable these once supported
    test_case_generator.disable_test_case(TestCase::PreviouslyDeleted);
    test_case_generator.disable_test_case(TestCase::WithOrRuleSet);
    test_case_generator.disable_test_case(TestCase::ReauthMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthNonMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthOrRuleMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthOrRuleNonMatchingTarget);

    // TODO: enable this once supported
    // test_case_generator.enable_bucket_statistic_checks(
    //     N_BUCKETS,
    //     num_devices,
    //     MATCH_DISTANCES_BUFFER_SIZE,
    // );

    test_case_generator
        .run_n_batches(NUM_BATCHES, MAX_BATCH_SIZE, MAX_DELETIONS_PER_BATCH, [
            &mut handle0,
            &mut handle1,
            &mut handle2,
        ])
        .await?;

    drop(handle0);
    drop(handle1);
    drop(handle2);

    // TODO: ATM we have no real way to wait for the actors to finish, so just sleep
    // a bit for now
    tokio::time::sleep(Duration::from_secs(5)).await;

    Ok(())
}
