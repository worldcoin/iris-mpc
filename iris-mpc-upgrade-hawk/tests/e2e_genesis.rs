use eyre::Result;

mod utils;
mod workflows;

#[tokio::test]
async fn test_100() -> Result<()> {
    let workflow = workflows::Genesis_100::new();
    workflow.run().await?;

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
