use crate::config::Config;
use crate::helpers::sqs::get_next_sns_seq_num;
use crate::server_coordination::get_check_addresses;
use eyre::{Context, Result};
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
) -> Result<BatchSyncState> {
    let next_sns_sequence_num = match get_next_sns_seq_num(config, &sqs_client.clone()).await? {
        Some(seq) => {
            tracing::info!("fetching next_sns_sequence_num: {}", seq);
            seq
        }
        None => {
            tracing::info!("fetching next_sns_sequence_num: None (queue was empty)");
            0
        }
    };
    let batch_sync_state = BatchSyncState {
        next_sns_sequence_num,
    };
    Ok(batch_sync_state)
}

pub async fn get_batch_sync_states(
    config: &Config,
    sqs_client: &aws_sdk_sqs::Client,
    own_state: Option<&BatchSyncState>,
) -> Result<Vec<BatchSyncState>> {
    let all_batch_size_sync_addresses = get_check_addresses(
        &config.node_hostnames,
        &config.healthcheck_ports,
        "batch-sync-state",
    );

    let own_sync_state = match own_state {
        Some(state) => state.clone(),
        None => get_own_batch_sync_state(config, sqs_client).await?,
    };

    let next_node = &all_batch_size_sync_addresses[(config.party_id + 1) % 3];
    let prev_node = &all_batch_size_sync_addresses[(config.party_id + 2) % 3];

    let mut states = Vec::with_capacity(3);
    states.push(own_sync_state.clone());

    for host in [next_node, prev_node].iter() {
        let res = reqwest::get(host.as_str())
            .await
            .with_context(|| format!("Failed to fetch batch sync state from party {}", host))?;
        let state: BatchSyncState = res
            .json()
            .await
            .with_context(|| format!("Failed to parse batch sync state from party {}", host))?;
        states.push(state);
    }
    Ok(states)
}
