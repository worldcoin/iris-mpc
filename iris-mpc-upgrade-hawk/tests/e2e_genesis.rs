use eyre::Result;
use utils::runner::{Test, TestContextInfo};

mod utils;
mod workflows;

/// HNSW-Genesis-100
///   against:
///     a known set of 100 Iris shares in plaintext format;
///     an empty set of exclusions;
///     an empty set of modifications;
///   asserts:
///     node processes exit normally;
///     graph construction is equivalent for each node;
#[tokio::test]
async fn test_100() -> Result<()> {
    use workflows::genesis_100::{TestRunner, TestState};

    let run_kind: usize = 100;
    let run_idx: usize = 1;

    let ctx = TestContextInfo::<TestState>::new(100, 1);
    TestRunner::do_test(&self, ctx).await?;

    Ok(())
}

// #[tokio::test]
// async fn test_100_1() -> Result<()> {
//     // Set network inputs.
//     let net_inputs = utils::get_net_inputs();

//     // Set context info.
//     let run_info = utils::get_test_info(net_inputs);

//     // Execute tests.
//     utils::runner::exec_test(run_info).await;

//     // Perform asserts
//     // TODO: implement assertions

//     panic!("TODO");

//     Ok(())
// }
