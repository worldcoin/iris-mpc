use super::misc::get_check_addresses;
use axum::http::StatusCode;
use eyre::{Report, Result, WrapErr};
use iris_mpc_common::config::Config;
use std::time::Duration;
use tokio;

pub(crate) async fn do_unreadiness_check(config: &Config) -> Result<(), Report> {
    let node_urls = get_check_addresses(config, "ready");
    let party_id = config.party_id;
    let task_handle = async move {
        let node_urls = [
            &node_urls[(party_id + 1) % 3],
            &node_urls[(party_id + 2) % 3],
        ];
        let mut is_unready = [false, false];

        loop {
            for (idx, url) in node_urls.iter().enumerate() {
                let response = reqwest::get(url.as_str()).await;
                if response.is_ok() && response.unwrap().status() == StatusCode::SERVICE_UNAVAILABLE
                {
                    is_unready[idx] = true;
                    if is_unready.iter().all(|&c| c) {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    };

    tracing::info!("Waiting for all nodes to be unready...");
    tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        task_handle,
    )
    .await
    .wrap_err("Timeout waiting for all nodes to be unready")
}
