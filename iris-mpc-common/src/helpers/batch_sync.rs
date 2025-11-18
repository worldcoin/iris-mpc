use crate::config::Config;
use crate::job::{CURRENT_BATCH_SHA, CURRENT_BATCH_VALID_ENTRIES};
use ampc_server_utils::{get_approximate_number_of_messages, get_check_addresses};
use eyre::{eyre, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::time::{timeout, Duration};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSyncState {
    pub messages_to_poll: u32,
    pub batch_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSyncEntries {
    pub valid_entries: Vec<bool>,
    pub batch_sha: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSyncEntriesResult {
    pub my_state: BatchSyncEntries,
    pub all_states: Vec<BatchSyncEntries>,
}

impl BatchSyncEntriesResult {
    pub fn new(my_state: BatchSyncEntries, all_states: Vec<BatchSyncEntries>) -> Self {
        Self {
            my_state,
            all_states,
        }
    }
    pub fn sha_matches(&self) -> bool {
        self.all_states
            .iter()
            .all(|s| s.batch_sha == self.my_state.batch_sha)
    }

    pub fn valid_entries(&self) -> Vec<bool> {
        self.all_states
            .iter()
            .map(|s| s.valid_entries.clone())
            .fold(
                vec![true; self.my_state.valid_entries.len()],
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

    pub fn own_sha_pretty(&self) -> String {
        hex::encode(&self.my_state.batch_sha[0..4])
    }

    pub fn all_shas_pretty(&self) -> String {
        self.all_states
            .iter()
            .map(|s| hex::encode(&s.batch_sha[0..4]))
            .collect::<Vec<_>>()
            .join(", ")
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

    pub fn messages_to_poll(&self) -> u32 {
        self.all_states
            .iter()
            .map(|s| s.messages_to_poll)
            .min()
            .unwrap_or(0)
    }
}

pub async fn get_own_batch_sync_entries() -> BatchSyncEntries {
    let current_batch_valid_entries = CURRENT_BATCH_VALID_ENTRIES
        .lock()
        .expect("Failed to lock CURRENT_BATCH_VALID_ENTRIES");
    let current_batch_hash = CURRENT_BATCH_SHA
        .lock()
        .expect("Failed to lock CURRENT_BATCH_SHA");

    BatchSyncEntries {
        valid_entries: current_batch_valid_entries.clone(),
        batch_sha: current_batch_hash.to_owned(),
    }
}

pub async fn get_own_batch_sync_state(
    config: &Config,
    sqs_client: &aws_sdk_sqs::Client,
    current_batch_id: u64,
) -> Result<BatchSyncState> {
    let approximate_visible_messages =
        get_approximate_number_of_messages(&sqs_client.clone(), &config.requests_queue_url).await?;
    tracing::info!(
        "fetching approximate_visible_messages: {}",
        approximate_visible_messages
    );

    let index = (current_batch_id - 1) as usize;

    let messages_to_poll = if config.predefined_batch_sizes.len() > index {
        // predefined_batch_sizes are only used in test environments to reproduce specific scenarios
        tracing::info!(
            "Using predefined batch size {} for batch ID {}",
            config.predefined_batch_sizes[index],
            current_batch_id
        );
        std::cmp::min(config.predefined_batch_sizes[index], config.max_batch_size) as u32
    } else {
        // Use the dynamic batch size calculation based on SQS approximate visible messages
        std::cmp::min(approximate_visible_messages, config.max_batch_size as u32)
    };

    let batch_sync_state = BatchSyncState {
        messages_to_poll,
        batch_id: current_batch_id,
    };
    Ok(batch_sync_state)
}

pub async fn get_batch_sync_entries(
    config: &Config,
    own_state: Option<BatchSyncEntries>,
) -> Result<Vec<BatchSyncEntries>> {
    let server_coord_config = config
        .server_coordination
        .as_ref()
        .ok_or(eyre!("Missing server coordination config"))?;

    let all_batch_size_sync_entries_addresses = get_check_addresses(
        &server_coord_config.node_hostnames,
        &server_coord_config.healthcheck_ports,
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
        let mut fetched_state: Option<BatchSyncEntries> = None;

        match timeout(polling_timeout_duration, async {
            loop {
                let res = reqwest::get(host.as_str()).await.with_context(|| {
                    format!("Failed to fetch batch sync entries from party {}", host)
                })?;
                let state: BatchSyncEntries = res.json().await.with_context(|| {
                    format!("Failed to parse batch sync entries from party {}", host)
                })?;

                if !state.batch_sha.eq(&own_sync_state.batch_sha) {
                    tracing::info!(
                        "Party {} (batch_hash {}) differs from own ({}). Retrying in 1 second...",
                        host,
                        hex::encode(&state.batch_sha[0..4]),
                        hex::encode(&own_sync_state.batch_sha[0..4])
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
                    hex::encode(&own_sync_state.batch_sha[0..4])
                );
                return Err(eyre!(
                    "Timeout waiting for party {} to reach batch hash {}",
                    host,
                    hex::encode(&own_sync_state.batch_sha[0..4])
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
    let server_coord_config = &config.server_coordination.clone().unwrap();
    let all_batch_size_sync_addresses = get_check_addresses(
        &server_coord_config.node_hostnames,
        &server_coord_config.healthcheck_ports,
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
                // Add batch_id as query parameter
                let url = format!("{}?batch_id={}", host.as_str(), reference_batch_id);
                let res = reqwest::get(&url).await.with_context(|| {
                    format!("Failed to fetch batch sync state from party {}", host)
                })?;

                tracing::info!("Response Status: {}", res.status());

                // Check if we got a 409 Conflict response (batch_id mismatch)
                if res.status() == reqwest::StatusCode::CONFLICT {
                    let error_body = res
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    tracing::info!(
                        "Party {} returned batch ID mismatch: {}. Retrying in 1 second...",
                        host,
                        error_body
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }

                // Handle other non-OK status codes
                if !res.status().is_success() {
                    let status = res.status();
                    let error_body = res
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    return Err(eyre!(
                        "Party {} returned error status {}: {}",
                        host,
                        status,
                        error_body
                    ));
                }

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
