use eyre::Result;

mod utils;

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
    utils::setup();

    // Set run network inputs.
    let net_inputs = utils::get_net_inputs();

    // Set run context info.
    let run_info = utils::get_test_info(net_inputs);

    // Execute tests.
    utils::runner::exec_test(run_info).await;

    // Test code here
    assert!(true);

    Ok(())
}
