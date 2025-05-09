use crate::config::Config;
use crate::helpers::sqs::get_next_sns_seq_num;
use crate::server_coordination::get_check_addresses;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSyncState {
    pub next_sns_sequence_num: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSyncResult {
    pub my_state: BatchSyncState,
    pub all_states: Vec<BatchSyncState>,
}

impl BatchSyncResult {
    pub fn new(my_state: BatchSyncState, all_states: Vec<BatchSyncState>) -> Self {
        Self {
            my_state,
            all_states,
        }
    }
    pub fn max_sns_sequence_num(&self) -> u128 {
        self.all_states
            .iter()
            .map(|s| s.next_sns_sequence_num)
            .max()
            .expect("can get max u128 value")
    }
}

pub async fn get_own_batch_sync_state(
    config: &Config,
    sqs_client: &aws_sdk_sqs::Client,
) -> eyre::Result<BatchSyncState> {
    let next_sns_sequence_num = get_next_sns_seq_num(config, &sqs_client.clone())
        .await?
        .unwrap_or(0);
    let batch_sync_state = BatchSyncState {
        next_sns_sequence_num,
    };
    Ok(batch_sync_state)
}

pub async fn get_batch_sync_states(
    config: &Config,
    sqs_client: &aws_sdk_sqs::Client,
) -> Vec<BatchSyncState> {
    let all_batch_size_sync_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "batch-sync-state",
    );
    let own_sync_state = get_own_batch_sync_state(config, sqs_client).await.unwrap();
    let next_node = &all_batch_size_sync_addresses[(config.party_id + 1) % 3];
    let prev_node = &all_batch_size_sync_addresses[(config.party_id + 2) % 3];

    let mut states = vec![own_sync_state.clone()];

    for host in [next_node, prev_node].iter() {
        let res = reqwest::get(host.as_str()).await;
        match res {
            Ok(res) => {
                let state: BatchSyncState = match res.json().await {
                    Ok(state) => state,
                    Err(e) => {
                        tracing::error!(
                            "Failed to parse batch size sync state from party {}: {:?}",
                            host,
                            e
                        );
                        panic!(
                            "could not get batch size sync state from party {}, trying to restart",
                            host
                        );
                    }
                };
                states.push(state);
            }
            Err(e) => {
                tracing::error!(
                    "Failed to fetch batch sync state from party {}: {:?}",
                    host,
                    e
                );
                panic!(
                    "could not get batch sync state from party {}, trying to restart",
                    host
                );
            }
        }
    }
    states
}
