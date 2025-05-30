use crate::config::Config;
use crate::helpers::sqs::get_approximate_number_of_messages;
use crate::server_coordination::get_check_addresses;
use eyre::{eyre, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::time::{timeout, Duration};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSyncState {
    pub approximate_visible_messages: u32,
    pub batch_id: u64,
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
