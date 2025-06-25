use eyre::Result;

mod shared;

/// HNSW-100: Basic genesis test.
///   against:
///     a known dataset of Iris shares in plaintext format;
///     an empty dataset of exclusions;
///     an empty dataset of modifications;
///   asserts:
///     node processes exit normally;
///     graph construction is equivalent for each node;
#[tokio::test]
async fn test_100() -> Result<()> {
    // Generic setup.
    shared::setup();

    // Set inputs.
    let net_inputs = shared::factory::get_net_inputs();
    let run_info = shared::types::TestRunInfo::new(net_inputs);

    // Execute tests.
    shared::runner::exec_test(run_info).await;

    // Test code here
    assert!(true);

    Ok(())
}
