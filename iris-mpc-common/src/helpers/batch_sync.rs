use crate::config::Config;
use crate::helpers::sqs::get_approximate_number_of_messages;
use crate::job::{InflightBatchMap, INFLIGHT_BATCHES};
use crate::server_coordination::get_check_addresses;
use eyre::{eyre, Context, Result};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tokio::time::{timeout, Duration};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSyncState {
    pub approximate_visible_messages: u32,
    pub batch_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSyncEntriesResult {
    pub my_state: InflightBatchMap,
    pub all_states: Vec<InflightBatchMap>,
}

impl BatchSyncEntriesResult {
    pub fn new(my_state: InflightBatchMap, all_states: Vec<InflightBatchMap>) -> Self {
        Self {
            my_state,
            all_states,
        }
    }
    pub fn sha_matches(&self, current_batch_sha: String) -> bool {
        self.all_states
            .iter()
            .all(|s| s.contains_key(&current_batch_sha))
    }

    pub fn valid_entries_for_sha(&self, batch_sha: String) -> Vec<bool> {
        let valid_entries = self
            .all_states
            .iter()
            .map(|s| {
                s.get(&batch_sha).unwrap_or_else(|| {
                    panic!("Invalid batch sha: {}", hex::encode(&batch_sha[0..4]))
                })
            })
            .collect_vec();

        valid_entries.iter().fold(
            vec![true; self.my_state.get(&batch_sha).iter().len()],
            |mut acc, entries| {
                for (i, &entry) in entries.iter().enumerate() {
                    if !entry {
                        acc[i] = false;
                    }
                }
                acc
            },
        )
    }
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

    pub fn max_approximate_visible_messages(&self) -> u32 {
        self.all_states
            .iter()
            .map(|s| s.approximate_visible_messages)
            .max()
            .unwrap_or(0)
    }
}

pub async fn get_own_batch_sync_entries() -> InflightBatchMap {
    let current_batch_hash = INFLIGHT_BATCHES
        .lock()
        .expect("Failed to lock CURRENT_BATCH_SHA");
    current_batch_hash.to_owned()
}

pub async fn get_own_batch_sync_state(
    config: &Config,
    sqs_client: &aws_sdk_sqs::Client,
    current_batch_id: u64,
) -> Result<BatchSyncState> {
    let approximate_visible_messages =
        get_approximate_number_of_messages(config, &sqs_client.clone()).await?;
    tracing::info!(
        "fetching approximate_visible_messages: {}",
        approximate_visible_messages
    );

    let batch_sync_state = BatchSyncState {
        approximate_visible_messages,
        batch_id: current_batch_id,
    };
    Ok(batch_sync_state)
}

pub async fn get_batch_sync_entries(
    config: &Config,
    own_state: Option<InflightBatchMap>,
    current_batch_sha: String,
) -> Result<Vec<InflightBatchMap>> {
    let all_batch_size_sync_entries_addresses = get_check_addresses(
        &config.node_hostnames,
        &config.healthcheck_ports,
        "batch-sync-entries",
    );

    let own_sync_state = match own_state {
        Some(state) => state.clone(),
        None => get_own_batch_sync_entries().await,
    };

    let next_node = &all_batch_size_sync_entries_addresses[(config.party_id + 1) % 3];
    let prev_node = &all_batch_size_sync_entries_addresses[(config.party_id + 2) % 3];

    let mut states = Vec::with_capacity(3);
    states.push(own_sync_state.clone());

    let polling_timeout_duration = Duration::from_secs(20);

    for host in [next_node, prev_node].iter() {
        let mut fetched_state: Option<InflightBatchMap> = None;

        match timeout(polling_timeout_duration, async {
            loop {
                let res = reqwest::get(host.as_str()).await.with_context(|| {
                    format!("Failed to fetch batch sync entries from party {}", host)
                })?;
                let state: InflightBatchMap = res.json().await.with_context(|| {
                    format!("Failed to parse batch sync entries from party {}", host)
                })?;

                if !state.contains_key(&current_batch_sha.clone()) {
                    let state_keys = state
                        .keys()
                        .map(|s| s[0..8].to_string())
                        .collect::<Vec<_>>()
                        .join(", ");

                    tracing::info!(
                        "Party {} ({}) does not have current batch_sha {}. Retrying in 1 second...",
                        host,
                        state_keys,
                        &current_batch_sha.clone()[0..8],
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                } else {
                    fetched_state = Some(state);
                    break;
                }
            }
            Ok::<(), eyre::Error>(())
        })
        .await
        {
            Ok(Ok(_)) => {
                if let Some(state) = fetched_state {
                    states.push(state);
                } else {
                    tracing::error!("Fetched_state is None after successful polling loop from party {}. This is a bug.", host);
                    return Err(eyre!("Internal logic error fetching state from {}", host));
                }
            }
            Ok(Err(e)) => {
                tracing::error!(
                    "Error polling party {}: {:?}. Using potentially stale or default sync entries.",
                    host,
                    e
                );
                return Err(eyre!(
                    "Failed to get a consistent batch_hash from party {} due to: {:?}",
                    host,
                    e
                ));
            }
            Err(_) => {
                tracing::error!(
                    "Timeout polling party {} for batch sync entries with hash {}",
                    host,
                    own_sync_state
                        .keys()
                        .map(|sha| hex::encode(&sha[0..4]))
                        .join(", ")
                );
                return Err(eyre!(
                    "Timeout waiting for party {} to reach batch hash {}",
                    host,
                    own_sync_state
                        .keys()
                        .map(|sha| hex::encode(&sha[0..4]))
                        .join(", ")
                ));
            }
        }
    }
    Ok(states)
}

pub async fn get_batch_sync_states(
    config: &Config,
    sqs_client: &aws_sdk_sqs::Client,
    own_state: Option<&BatchSyncState>,
    current_batch_id: u64,
) -> Result<Vec<BatchSyncState>> {
    let all_batch_size_sync_addresses = get_check_addresses(
        &config.node_hostnames,
        &config.healthcheck_ports,
        "batch-sync-state",
    );

    let own_sync_state = match own_state {
        Some(state) => state.clone(),
        None => get_own_batch_sync_state(config, sqs_client, current_batch_id).await?,
    };

    let reference_batch_id = own_sync_state.batch_id;

    let next_node = &all_batch_size_sync_addresses[(config.party_id + 1) % 3];
    let prev_node = &all_batch_size_sync_addresses[(config.party_id + 2) % 3];

    let mut states = Vec::with_capacity(3);
    states.push(own_sync_state.clone());

    let polling_timeout_duration = Duration::from_secs(config.batch_sync_polling_timeout_secs);

    for host in [next_node, prev_node].iter() {
        let mut fetched_state: Option<BatchSyncState> = None;

        match timeout(polling_timeout_duration, async {
            loop {
                let res = reqwest::get(host.as_str()).await.with_context(|| {
                    format!("Failed to fetch batch sync state from party {}", host)
                })?;
                let state: BatchSyncState = res.json().await.with_context(|| {
                    format!("Failed to parse batch sync state from party {}", host)
                })?;

                if state.batch_id < reference_batch_id {
                    tracing::info!(
                        "Party {} (batch_id {}) is behind own batch_id {}. Retrying in 1 second...",
                        host,
                        state.batch_id,
                        reference_batch_id
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                } else {
                    fetched_state = Some(state);
                    break;
                }
            }
            Ok::<(), eyre::Error>(())
        })
        .await
        {
            Ok(Ok(_)) => {
                if let Some(state) = fetched_state {
                    match state.batch_id.cmp(&reference_batch_id) {
                        std::cmp::Ordering::Greater => {
                            tracing::warn!(
                                "Received batch sync state from party {} for a future batch_id {} (own is {}). This might indicate this node is behind.",
                                host, state.batch_id, reference_batch_id
                            );
                        }
                        std::cmp::Ordering::Less => {
                            tracing::error!(
                                "Party {} (batch_id {}) is still behind own batch_id {} after polling loop. This is unexpected.",
                                host, state.batch_id, reference_batch_id
                            );
                        }
                        std::cmp::Ordering::Equal => {}
                    }
                    states.push(state);
                } else {
                    tracing::error!("Fetched_state is None after successful polling loop from party {}. This is a bug.", host);
                    return Err(eyre!("Internal logic error fetching state from {}", host));
                }
            }
            Ok(Err(e)) => {
                tracing::error!(
                    "Error polling party {}: {:?}. Using potentially stale or default state.",
                    host,
                    e
                );
                return Err(eyre!(
                    "Failed to get a consistent batch_id from party {} due to: {:?}",
                    host,
                    e
                ));
            }
            Err(_) => {
                tracing::error!("Timeout polling party {} for batch_id {}. Using potentially stale or default state.", host, reference_batch_id);
                return Err(eyre!(
                    "Timeout waiting for party {} to reach batch_id {}",
                    host,
                    reference_batch_id
                ));
            }
        }
    }
    Ok(states)
}
