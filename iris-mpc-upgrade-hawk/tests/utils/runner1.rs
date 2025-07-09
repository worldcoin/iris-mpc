use super::{logger, types::TestRunInfo};
use iris_mpc_upgrade_hawk::genesis::exec as exec_genesis;

pub async fn exec_test(info: TestRunInfo) {
    logger::log_info(format!("{}", info).as_str(), "Test execution starts");

    // Set collected futures.
    let exec_futures: Vec<_> = info
        .node_inputs()
        .into_iter()
        .map(|node_input| {
            let args = node_input.args;
            let config = node_input.config;
            exec_genesis(args, config)
        })
        .collect();

    // Await all futures to complete.
    let exec_results = futures::future::join_all(exec_futures).await;

    for result in exec_results {
        if let Err(e) = result {
            eprintln!("Error during execution: {}", e);
        }
    }

    logger::log_info(format!("{}", info).as_str(), "Test execution ends");
}
