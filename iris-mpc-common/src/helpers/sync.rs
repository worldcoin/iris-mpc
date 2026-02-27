use eyre::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt, fmt::Display, str::FromStr};

use crate::config::CommonConfig;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    pub db_len: u64,
    pub modifications: Vec<Modification>,
    pub next_sns_sequence_num: Option<u128>,
    pub common_config: CommonConfig,
    #[serde(default)]
    pub rerand_state: Option<RerandSyncState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerandSyncState {
    pub epoch: i32,
    /// Highest chunk_id where all_confirmed = TRUE. -1 if none confirmed.
    pub max_confirmed_chunk: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncResult {
    pub my_state: SyncState,
    pub all_states: Vec<SyncState>,
}

/// ModificationKey is used to easily look up modifications after a batch to mark them completed.
/// All request types with a pre-determined serial id uses `SerialId` option while uniqueness requests use `RequestId` as they are assigned a serial id after the protocol if they're unique.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModificationKey {
    RequestSerialId(u32),
    RequestId(String),
}

pub const MOD_STATUS_IN_PROGRESS: &str = "IN_PROGRESS";
pub const MOD_STATUS_COMPLETED: &str = "COMPLETED";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModificationStatus {
    InProgress,
    Completed,
}

impl Display for ModificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModificationStatus::InProgress => write!(f, "{MOD_STATUS_IN_PROGRESS}"),
            ModificationStatus::Completed => write!(f, "{MOD_STATUS_COMPLETED}"),
        }
    }
}

impl FromStr for ModificationStatus {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            MOD_STATUS_IN_PROGRESS => Ok(ModificationStatus::InProgress),
            MOD_STATUS_COMPLETED => Ok(ModificationStatus::Completed),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct Modification {
    pub id: i64,
    pub serial_id: Option<i64>,
    pub request_type: String,
    pub s3_url: Option<String>,
    pub status: String,
    pub persisted: bool,
    pub result_message_body: Option<String>,
    pub graph_mutation: Option<Vec<u8>>,
}

impl PartialEq for Modification {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.serial_id == other.serial_id
            && self.request_type == other.request_type
            && self.s3_url == other.s3_url
            && self.status == other.status
            && self.persisted == other.persisted
        // result_message_body graph_mutation are ignored since they are difference across nodes
    }
}

impl Eq for Modification {}

impl fmt::Debug for Modification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let graph_mutation_summary = match &self.graph_mutation {
            Some(bytes) => format!("Some([{} bytes])", bytes.len()),
            None => "None".to_string(),
        };
        let result_message_summary = match &self.result_message_body {
            Some(msg) => format!("Some([{} chars])", msg.chars().count()),
            None => "None".to_string(),
        };

        f.debug_struct("Modification")
            .field("id", &self.id)
            .field("serial_id", &self.serial_id)
            .field("request_type", &self.request_type)
            .field("s3_url", &self.s3_url)
            .field("status", &self.status)
            .field("persisted", &self.persisted)
            .field("result_message_body", &result_message_summary)
            .field("graph_mutation", &graph_mutation_summary)
            .finish()
    }
}

impl Modification {
    /// Marks the modification as completed, setting the status to "COMPLETED", updating the result message body and persisted flag.
    ///
    /// If `updated_serial_id` is provided, it updates the serial_id field as well.
    /// It is used when the modification is a uniqueness request and the serial id is assigned after the protocol.
    pub fn mark_completed(
        &mut self,
        persisted: bool,
        result_message_body: &str,
        updated_serial_id: Option<u32>,
        graph_mutation: Option<Vec<u8>>,
    ) {
        self.status = ModificationStatus::Completed.to_string();
        self.result_message_body = Some(result_message_body.to_string());
        self.persisted = persisted;
        if let Some(serial_id) = updated_serial_id {
            self.serial_id = Some(serial_id as i64);
        }
        self.graph_mutation = graph_mutation;
    }

    /// Updates the node_id field in the SNS message JSON to specified one
    pub fn update_result_message_node_id(&mut self, party_id: usize) -> Result<()> {
        if let Some(message) = &self.result_message_body {
            // Parse the JSON message
            match serde_json::from_str::<serde_json::Value>(message) {
                Ok(mut json_value) => {
                    // Update the node_id field if it exists
                    if let Some(obj) = json_value.as_object_mut() {
                        // Try to update node_id in the main object
                        if obj.contains_key("node_id") {
                            obj.insert(
                                "node_id".to_string(),
                                serde_json::Value::Number(serde_json::Number::from(party_id)),
                            );
                            self.result_message_body = Some(serde_json::to_string(&json_value)?);
                        } else {
                            return Err(eyre::eyre!("Message body does not contain node_id"));
                        }
                    }
                }
                Err(_) => {
                    return Err(eyre::eyre!("Invalid JSON message"));
                }
            }
        } else {
            return Err(eyre::eyre!("Result message body is None"));
        }
        Ok(())
    }
}

impl SyncResult {
    pub fn new(my_state: SyncState, all_states: Vec<SyncState>) -> Self {
        Self {
            my_state,
            all_states,
        }
    }

    /// Check if the common part of the config is the same across all nodes.
    pub fn check_common_config(&self) -> Result<()> {
        let my_config = &self.my_state.common_config;
        for SyncState {
            common_config: other_config,
            ..
        } in self.all_states.iter()
        {
            ensure!(
                my_config == other_config,
                "Inconsistent common config!\nhave: {:?}\ngot: {:?}",
                my_config,
                other_config
            );
        }
        Ok(())
    }

    pub fn max_sns_sequence_num(&self) -> Option<u128> {
        let sequence_nums: Vec<Option<u128>> = self
            .all_states
            .iter()
            .map(|s| s.next_sns_sequence_num)
            .collect();

        // All nodes should either have an empty queue or filled with some items.
        // Otherwise, we can not conclude queue sync and proceed safely.
        // More info: https://linear.app/worldcoin/issue/POP-2577/cover-edge-case-in-sqs-sync
        let any_empty_queues = sequence_nums.iter().any(|seq| seq.is_none());
        let any_non_empty_queues = sequence_nums.iter().any(|seq| seq.is_some());
        if any_empty_queues && any_non_empty_queues {
            panic!(
                "Can not deduce max SNS sequence number safely out of {:?}. Restarting...",
                sequence_nums
            );
        }

        sequence_nums
            .into_iter()
            .max()
            .expect("can get max u128 value")
    }

    /// Compare local `modifications` (my_state) to all other parties'
    /// modifications (all_states), grouping by `id`. Returns: (to_update,
    /// to_delete)
    /// - `to_update`: modifications the local node should add (e.g., mark
    ///   completed/persisted)
    /// - `to_delete`: modifications the local node should remove from the DB
    ///   (in-progress, never completed).
    pub fn compare_modifications(&self) -> (Vec<Modification>, Vec<Modification>) {
        // 1. Group all modifications by id => Vec<Modification> (from different nodes)
        let mut grouped: HashMap<i64, Vec<Modification>> = HashMap::new();
        for m in self.all_states.iter().flat_map(|s| s.modifications.clone()) {
            grouped.entry(m.id).or_default().push(m);
        }

        tracing::info!("Grouped modifications: {:?}", grouped);

        let completed_max_mod_ids: Vec<Option<i64>> = self
            .all_states
            .iter()
            .map(|s| {
                s.modifications
                    .iter()
                    .filter(|m| m.status == MOD_STATUS_COMPLETED)
                    .map(|m| m.id)
                    .max()
            })
            .collect();
        let min_id = completed_max_mod_ids.iter().flatten().copied().min();
        let max_id = completed_max_mod_ids.iter().flatten().copied().max();
        if let (Some(min_id), Some(max_id)) = (min_id, max_id) {
            let mod_id_diff = max_id.saturating_sub(min_id) as usize;
            if mod_id_diff > self.my_state.common_config.get_max_modifications_lookback() {
                panic!(
                    "Modification ID difference across nodes is too large: {:?}. Min: {:?}, Max: {:?}. \
             Can not safely handle this case, consider bumping lookback. Crashing!",
                    completed_max_mod_ids, min_id, max_id
                );
            }
        }

        // Store the results here
        let mut to_update = Vec::new();
        let mut to_delete = Vec::new();

        // 2. Analyze each modification group
        for (&id, group_mods) in &grouped {
            assert_modifications_consistency(group_mods);

            // Find local node's copy, if any
            let local_copy = self.my_state.modifications.iter().find(|m| m.id == id);

            // Evaluate the global state across all nodes:
            let any_completed = group_mods
                .iter()
                .any(|m| m.status == ModificationStatus::Completed.to_string());
            let all_in_progress = group_mods
                .iter()
                .all(|m| m.status == ModificationStatus::InProgress.to_string());
            let any_persisted = group_mods.iter().any(|m| m.persisted);

            if all_in_progress {
                // If they're all in-progress => ignore the modification by deleting it
                if let Some(local_m) = local_copy {
                    to_delete.push(local_m.clone());
                }
            } else if any_completed {
                // If any node completed => unify to COMPLETED
                let first_completed = group_mods
                    .iter()
                    .find(|m| m.status == ModificationStatus::Completed.to_string())
                    .expect("At least one completed modification");
                match local_copy {
                    None => {
                        // If an item is completed for a party, it should at least exist in the
                        // local state because it should have been added during receive_batch.
                        // This can only happen when other party misses an in_progress mod.
                        // Local party will fetch until modification id X while the other party will
                        // fetch until mod id X-1. In this case, local party won't find X-1.
                        // We log and skip updating to avoid rolling back to an older share in local.
                        tracing::info!(
                            "Skip missing completed modification: {:?}",
                            first_completed
                        );
                    }
                    Some(local_m) => {
                        if local_m.status != ModificationStatus::Completed.to_string()
                            || local_m.persisted != any_persisted
                        {
                            // If local is not "completed" or doesn't match the final persisted
                            // We'll roll forward local_m
                            let mut roll_forward = first_completed.clone();
                            roll_forward.status = ModificationStatus::Completed.to_string();
                            roll_forward.persisted = any_persisted;
                            tracing::warn!(
                                "Updating modification row from {:?} to {:?}",
                                local_m,
                                roll_forward
                            );
                            to_update.push(roll_forward);
                        } else {
                            tracing::debug!("Local modification is already in sync: {:?}", local_m);
                        }
                    }
                }
            } else {
                panic!("Unexpected modification state: {:?}", group_mods);
            }
        }

        (to_update, to_delete)
    }
}

/// Assert that all modifications in the group have the same ID, serial ID,
/// request type, and S3 URL. If the assert fails, it means the modifications
/// are inconsistent across nodes. Such a case would need manual intervention.
fn assert_modifications_consistency(modifications: &[Modification]) {
    let first = modifications.first().expect("Empty modifications");
    for m in modifications.iter().skip(1) {
        assert_eq!(first.id, m.id, "Inconsistent modification IDs");
        assert_eq!(
            first.request_type, m.request_type,
            "Inconsistent request types"
        );
        assert_eq!(first.s3_url, m.s3_url, "Inconsistent S3 URLs");

        // Below fields could be missing in the behind party (missing a modification)
        // They should only be compared if both exists
        if first.serial_id.is_some() && m.serial_id.is_some() {
            assert_eq!(first.serial_id, m.serial_id, "Inconsistent serial IDs");
        }
        if first.graph_mutation.is_some() && m.graph_mutation.is_some() {
            assert_eq!(
                first.graph_mutation, m.graph_mutation,
                "Inconsistent graph mutations"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;
    use crate::{
        config::Config,
        helpers::{
            smpc_request::{IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE},
            smpc_response::{IdentityDeletionResult, ReAuthResult},
        },
    };
    use rand::random;

    fn random_graph_mutation() -> Vec<u8> {
        random::<[u8; 16]>().to_vec()
    }

    // Helper function to create a Modification.
    fn create_modification(
        id: i64,
        serial_id: Option<i64>,
        request_type: &str,
        s3_url: Option<&str>,
        status: ModificationStatus,
        persisted: bool,
        graph_mutation: Option<Vec<u8>>,
    ) -> Modification {
        Modification {
            id,
            serial_id,
            request_type: request_type.to_string(),
            s3_url: s3_url.map(|s| s.to_string()),
            status: status.to_string(),
            persisted,
            result_message_body: None,
            graph_mutation,
        }
    }

    // Create a SyncState with a given vector of modifications.
    fn create_sync_state(modifications: Vec<Modification>) -> SyncState {
        let default_lookback = (100 + 64) * 2;
        create_sync_state_with_lookback(modifications, default_lookback)
    }

    // Create a SyncState with a custom lookback value for testing.
    fn create_sync_state_with_lookback(
        modifications: Vec<Modification>,
        lookback: usize,
    ) -> SyncState {
        let mut config = Config::load_config("dummy").unwrap();
        config.max_modifications_lookback = lookback;
        SyncState {
            db_len: modifications.len() as u64,
            modifications,
            next_sns_sequence_num: None,
            common_config: CommonConfig::from(config),
            rerand_state: None,
        }
    }

    #[test]
    fn test_compare_modifications_local_party_outdated() {
        let mod1_graph_mut = random_graph_mutation();
        let mod3_graph_mut = random_graph_mutation();
        let mod5_graph_mut = random_graph_mutation();
        let mod1_local = create_modification(
            1,
            Some(100),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            Some(mod1_graph_mut.clone()),
        );
        let mod2_local = create_modification(
            2,
            Some(200),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let mod3_local = create_modification(
            3,
            Some(300),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
        );
        let mod4_local = create_modification(
            4,
            Some(400),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::InProgress,
            false,
            None,
        );
        let mod5_local = create_modification(
            5,
            None,
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/mod5"),
            ModificationStatus::InProgress,
            false,
            None,
        );
        let mod6_local = create_modification(
            6,
            None,
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/mod6"),
            ModificationStatus::InProgress,
            false,
            None,
        );
        let my_state = create_sync_state(vec![
            mod1_local.clone(),
            mod2_local.clone(),
            mod3_local.clone(),
            mod4_local.clone(),
            mod5_local.clone(),
            mod6_local.clone(),
        ]);

        let mod1_other = create_modification(
            1,
            Some(100),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            Some(mod1_graph_mut.clone()),
        );
        let mod2_other = create_modification(
            2,
            Some(200),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let mod3_other = create_modification(
            3,
            Some(300),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            Some(mod3_graph_mut.clone()),
        );
        let mod4_other = create_modification(
            4,
            Some(400),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let mod5_other = create_modification(
            5,
            Some(500),
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/mod5"),
            ModificationStatus::Completed,
            true,
            Some(mod5_graph_mut.clone()),
        );
        let mod6_other = create_modification(
            6,
            None,
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/mod6"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let other_state = create_sync_state(vec![
            mod1_other,
            mod2_other,
            mod3_other.clone(),
            mod4_other.clone(),
            mod5_other.clone(),
            mod6_other.clone(),
        ]);
        let all_states = vec![my_state.clone(), other_state.clone(), other_state.clone()];

        let sync_result = SyncResult {
            my_state,
            all_states,
        };

        let (to_update, to_delete) = sync_result.compare_modifications();

        // Expectations:
        // For ID=1,2: Already in sync → no action.
        // For ID=2-6: Local is IN_PROGRESS, other nodes are COMPLETED → roll forward to COMPLETED
        assert_eq!(to_update.len(), 4, "Expected four modifications to update");
        assert_eq!(to_delete.len(), 0, "Expected zero modification to delete");

        let update_mod3 = to_update.iter().find(|m| m.id == 3).unwrap();
        assert_eq!(update_mod3.clone(), mod3_other);

        let update_mod4 = to_update.iter().find(|m| m.id == 4).unwrap();
        assert_eq!(update_mod4.clone(), mod4_other);

        let update_mod5 = to_update.iter().find(|m| m.id == 5).unwrap();
        assert_eq!(update_mod5.clone(), mod5_other);

        let update_mod6 = to_update.iter().find(|m| m.id == 6).unwrap();
        assert_eq!(update_mod6.clone(), mod6_other);
    }

    #[test]
    fn test_compare_modifications_local_party_up_to_date() {
        let mod1_graph_mut = random_graph_mutation();
        let mod3_graph_mut = random_graph_mutation();
        let mod5_graph_mut = random_graph_mutation();
        // Create local modifications that are already up-to-date.
        let mod1_local = create_modification(
            1,
            Some(100),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            Some(mod1_graph_mut.clone()),
        );
        let mod2_local = create_modification(
            2,
            Some(200),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let mod3_local = create_modification(
            3,
            Some(300),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            Some(mod3_graph_mut.clone()),
        );
        let mod4_local = create_modification(
            4,
            Some(400),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let mod5_local = create_modification(
            5,
            Some(500),
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/mod5"),
            ModificationStatus::Completed,
            true,
            Some(mod5_graph_mut.clone()),
        );
        let mod6_local = create_modification(
            6,
            None,
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/mod6"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let my_state = create_sync_state(vec![
            mod1_local.clone(),
            mod2_local.clone(),
            mod3_local.clone(),
            mod4_local.clone(),
            mod5_local.clone(),
            mod6_local.clone(),
        ]);

        // Create other states with in-progress modifications.
        let mod1_other = create_modification(
            1,
            Some(100),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            Some(mod1_graph_mut.clone()),
        );
        let mod2_other = create_modification(
            2,
            Some(200),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let mod3_other = create_modification(
            3,
            Some(300),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
        );
        let mod4_other = create_modification(
            4,
            Some(400),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::InProgress,
            false,
            None,
        );
        let mod5_other = create_modification(
            5,
            None,
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/mod5"),
            ModificationStatus::InProgress,
            false,
            None,
        );
        let mod6_other = create_modification(
            6,
            None,
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/mod6"),
            ModificationStatus::InProgress,
            false,
            None,
        );
        let other_state = create_sync_state(vec![
            mod1_other, mod2_other, mod3_other, mod4_other, mod5_other, mod6_other,
        ]);
        let all_states = vec![my_state.clone(), other_state.clone(), other_state.clone()];

        let sync_result = SyncResult {
            my_state,
            all_states,
        };

        let (to_update, to_delete) = sync_result.compare_modifications();

        // Since local is already the most advanced party, nothing should be updated or
        // deleted.
        assert!(to_update.is_empty(), "Expected no modifications to update");
        assert!(to_delete.is_empty(), "Expected no modifications to delete");
    }

    #[test]
    fn test_compare_modifications_update_outside_lookback_modification() {
        // Suppose that we have a lookback window of 2 modifications.
        // If latest modification is IN_PROGRESS in local party and completely missing in other
        // party (not fetched from SQS yet), the other party will return modification outside the
        // lookback window. In this case, local party should skip outside-window modification and
        // delete the latest in progress one.
        let mod2_local = create_modification(
            2,
            Some(200),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let mod3_local = create_modification(
            3,
            Some(300),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
        );
        let my_state = create_sync_state(vec![mod2_local.clone(), mod3_local.clone()]);

        let mut mod1_other = create_modification(
            1,
            Some(100),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            None,
        );
        mod1_other.result_message_body = Some(
            serde_json::to_string(&IdentityDeletionResult {
                node_id: 1,
                serial_id: 100,
                success: true,
            })
            .unwrap(),
        );
        let mod2_other = mod2_local;
        let other_state = create_sync_state(vec![mod1_other.clone(), mod2_other]);
        let all_states = vec![my_state.clone(), other_state.clone(), other_state.clone()];
        let sync_result = SyncResult {
            my_state,
            all_states,
        };

        // Compare modifications across nodes.
        let (to_update, to_delete) = sync_result.compare_modifications();

        assert_eq!(to_update.len(), 0, "Expected no modification to update");
        assert_eq!(to_delete.len(), 1, "Expected one modification to delete");

        // Expectation: Local party should delete mod3.
        assert_eq!(to_delete[0], mod3_local);
    }

    #[test]
    fn test_compare_modifications_remove_in_progress() {
        // Create local modifications with some in-progress.
        let mod1_local = create_modification(
            1,
            Some(100),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            None,
        );
        let mod2_local = create_modification(
            2,
            Some(200),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
            None,
        );
        let mod3_local = create_modification(
            3,
            Some(300),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
        );
        let mod4_local = create_modification(
            4,
            Some(400),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::InProgress,
            false,
            None,
        );
        let my_state = create_sync_state(vec![
            mod1_local.clone(),
            mod2_local.clone(),
            mod3_local.clone(),
            mod4_local.clone(),
        ]);

        let all_states = vec![my_state.clone(), my_state.clone(), my_state.clone()];

        let sync_result = SyncResult {
            my_state,
            all_states,
        };

        // Compare modifications across nodes.
        let (to_update, to_delete) = sync_result.compare_modifications();

        // None of the parties have a final res
        assert!(to_update.is_empty(), "Expected no modifications to update");
        assert_eq!(to_delete.len(), 2, "Expected no modifications to delete");

        let delete_mod3 = to_delete.iter().find(|m| m.id == 3).unwrap();
        assert_eq!(delete_mod3.clone(), mod3_local);

        let delete_mod4 = to_delete.iter().find(|m| m.id == 4).unwrap();
        assert_eq!(delete_mod4.clone(), mod4_local);
    }

    #[test]
    fn test_max_sns_sequence_num() {
        // 1. Test with all Some sequence values
        let states = vec![
            SyncState {
                db_len: 10,
                modifications: vec![],
                next_sns_sequence_num: Some(100),
                common_config: CommonConfig::default(),
                rerand_state: None,
            },
            SyncState {
                db_len: 20,
                modifications: vec![],
                next_sns_sequence_num: Some(200),
                common_config: CommonConfig::default(),
                rerand_state: None,
            },
            SyncState {
                db_len: 30,
                modifications: vec![],
                next_sns_sequence_num: Some(150),
                common_config: CommonConfig::default(),
                rerand_state: None,
            },
        ];

        let sync_result = SyncResult::new(states[0].clone(), states);
        assert_eq!(sync_result.max_sns_sequence_num(), Some(200));

        // 2. Test with all None sequence values
        let state_with_none_sequence_num = SyncState {
            db_len: 10,
            modifications: vec![],
            next_sns_sequence_num: None,
            common_config: CommonConfig::default(),
            rerand_state: None,
        };
        let all_states = vec![
            state_with_none_sequence_num.clone(),
            state_with_none_sequence_num.clone(),
            state_with_none_sequence_num.clone(),
        ];

        let sync_result_none = SyncResult::new(state_with_none_sequence_num, all_states);
        assert_eq!(sync_result_none.max_sns_sequence_num(), None);
    }

    #[test]
    #[should_panic(expected = "Can not deduce max SNS sequence number safely")]
    fn test_max_sns_sequence_num_mixed_panic() {
        // Test the edge case where some nodes have None while others have Some
        // This should panic to prevent the batch mismatch described in the issue
        let states = vec![
            SyncState {
                db_len: 10,
                modifications: vec![],
                next_sns_sequence_num: None, // NodeX - advanced but empty queue
                common_config: CommonConfig::default(),
                rerand_state: None,
            },
            SyncState {
                db_len: 20,
                modifications: vec![],
                next_sns_sequence_num: Some(123), // Other nodes still have messages
                common_config: CommonConfig::default(),
                rerand_state: None,
            },
            SyncState {
                db_len: 30,
                modifications: vec![],
                next_sns_sequence_num: Some(123),
                common_config: CommonConfig::default(),
                rerand_state: None,
            },
        ];

        let sync_result = SyncResult::new(states[0].clone(), states);
        // This should panic due to inconsistent sequence numbers
        sync_result.max_sns_sequence_num();
    }

    #[test]
    fn test_update_sns_message_node_id() {
        // Test 1: ReauthResult
        let original_reauth_result = ReAuthResult {
            reauth_id: "test-reauth-123".to_string(),
            node_id: 1,
            serial_id: 123,
            success: true,
            matched_serial_ids: vec![123],
            or_rule_used: false,
            error: None,
            error_reason: None,
        };

        // Serialize the original result
        let serialized_reauth = serde_json::to_string(&original_reauth_result).unwrap();

        // Create a modification with the serialized result
        let mut modification = Modification {
            id: 1,
            serial_id: Some(123),
            request_type: REAUTH_MESSAGE_TYPE.to_string(),
            s3_url: "http://example.com/123".to_string().into(),
            status: ModificationStatus::Completed.to_string(),
            persisted: true,
            result_message_body: Some(serialized_reauth),
            graph_mutation: None,
        };

        // Update the node_id in the serialized message
        let new_party_id = 2;
        modification
            .update_result_message_node_id(new_party_id)
            .unwrap();

        // Deserialize and check if node_id was updated
        let updated_reauth_result: ReAuthResult =
            serde_json::from_str(&modification.result_message_body.unwrap()).unwrap();
        assert_eq!(updated_reauth_result.node_id, new_party_id);
        assert_eq!(
            updated_reauth_result.reauth_id,
            original_reauth_result.reauth_id
        );
        assert_eq!(
            updated_reauth_result.serial_id,
            original_reauth_result.serial_id
        );
        assert_eq!(
            updated_reauth_result.success,
            original_reauth_result.success
        );

        // Test 2: IdentityDeletionResult
        let original_deletion_result = IdentityDeletionResult {
            node_id: 2,
            serial_id: 456,
            success: true,
        };

        // Serialize the original result
        let serialized_deletion = serde_json::to_string(&original_deletion_result).unwrap();

        // Create a modification with the serialized result
        let mut modification = Modification {
            id: 2,
            serial_id: Some(456),
            request_type: IDENTITY_DELETION_MESSAGE_TYPE.to_string(),
            s3_url: None,
            status: ModificationStatus::Completed.to_string(),
            persisted: true,
            result_message_body: Some(serialized_deletion),
            graph_mutation: None,
        };

        // Update the node_id in the serialized message
        let new_party_id = 0;
        modification
            .update_result_message_node_id(new_party_id)
            .unwrap();

        // Deserialize and check if node_id was updated
        let updated_deletion_result: IdentityDeletionResult =
            serde_json::from_str(&modification.result_message_body.unwrap()).unwrap();
        assert_eq!(updated_deletion_result.node_id, new_party_id);
        assert_eq!(
            updated_deletion_result.serial_id,
            original_deletion_result.serial_id
        );
        assert_eq!(
            updated_deletion_result.success,
            original_deletion_result.success
        );
    }

    #[test]
    fn test_common_config_sync() {
        // load a dummy config with default values
        let config = Config::load_config("dummy").unwrap();
        let mut config1 = config;
        config1.luc_enabled = false;
        let config2 = config1.clone();
        let mut config3 = config1.clone();
        config3.luc_enabled = true;

        // 1. Test with mixed configs
        let states = vec![
            SyncState {
                db_len: 20,
                modifications: vec![],
                next_sns_sequence_num: Some(100),
                common_config: CommonConfig::from(config1),
                rerand_state: None,
            },
            SyncState {
                db_len: 20,
                modifications: vec![],
                next_sns_sequence_num: Some(100),
                common_config: CommonConfig::from(config2),
                rerand_state: None,
            },
            SyncState {
                db_len: 20,
                modifications: vec![],
                next_sns_sequence_num: Some(100),
                common_config: CommonConfig::from(config3),
                rerand_state: None,
            },
        ];

        let sync_result = SyncResult::new(states[0].clone(), states);
        assert!(sync_result.check_common_config().is_err());
    }

    #[test]
    #[should_panic(expected = "Modification ID difference across nodes is too large")]
    fn test_compare_modifications_large_id_difference_panic() {
        // Create a scenario where nodes have completed modifications with IDs
        // that differ by more than the max_modifications_lookback limit.
        // Test lookback is (100 + 64) * 2 = 328, so we'll create a difference of 350.

        // Node 1: has completed modification with ID 1
        let mod1_node1 = create_modification(
            1,
            Some(100),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/100"),
            ModificationStatus::Completed,
            true,
            None,
        );
        let my_state = create_sync_state_with_lookback(vec![mod1_node1], 10);

        // Node 2: has completed modification with ID 15 (difference = 14 > 10)
        let mod15_node2 = create_modification(
            15,
            Some(1500),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/1500"),
            ModificationStatus::Completed,
            true,
            None,
        );
        let other_state1 = create_sync_state_with_lookback(vec![mod15_node2], 10);

        // Node 3: has completed modification with ID 20 (even larger)
        let mod20_node3 = create_modification(
            20,
            Some(2000),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            None,
        );
        let other_state2 = create_sync_state_with_lookback(vec![mod20_node3], 10);

        let all_states = vec![my_state.clone(), other_state1, other_state2];

        let sync_result = SyncResult {
            my_state,
            all_states,
        };

        // This should panic because max_id (20) - min_id (1) = 19 > 10 (test lookback)
        sync_result.compare_modifications();
    }
}
