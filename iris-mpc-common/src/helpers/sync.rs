use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt, fmt::Display, str::FromStr};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    pub db_len: u64,
    pub deleted_request_ids: Vec<String>,
    pub modifications: Vec<Modification>,
    pub next_sns_sequence_num: Option<u128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncResult {
    my_state: SyncState,
    all_states: Vec<SyncState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModificationStatus {
    InProgress,
    Completed,
}

impl Display for ModificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModificationStatus::InProgress => write!(f, "IN_PROGRESS"),
            ModificationStatus::Completed => write!(f, "COMPLETED"),
        }
    }
}

impl FromStr for ModificationStatus {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "IN_PROGRESS" => Ok(ModificationStatus::InProgress),
            "COMPLETED" => Ok(ModificationStatus::Completed),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modification {
    pub id: String,
    pub serial_id: i64,
    pub request_type: String,
    pub s3_url: Option<String>,
    pub status: String,
    pub persisted: bool,
}

impl Modification {
    pub fn mark_completed(&mut self, persisted: bool) {
        self.status = ModificationStatus::Completed.to_string();
        self.persisted = persisted;
    }
}

impl SyncResult {
    pub fn new(my_state: SyncState, all_states: Vec<SyncState>) -> Self {
        Self {
            my_state,
            all_states,
        }
    }

    pub fn must_rollback_storage(&self) -> Option<usize> {
        let smallest_len = self.all_states.iter().map(|s| s.db_len).min()?;
        let all_equal = self.all_states.iter().all(|s| s.db_len == smallest_len);
        if all_equal {
            None
        } else {
            Some(smallest_len as usize)
        }
    }

    pub fn deleted_request_ids(&self) -> Vec<String> {
        // Merge request IDs.
        self.all_states
            .iter()
            .flat_map(|s| s.deleted_request_ids.clone())
            .sorted()
            .dedup()
            .collect()
    }

    pub fn max_sns_sequence_num(&self) -> Option<u128> {
        self.all_states
            .iter()
            .map(|s| s.next_sns_sequence_num)
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
        let mut grouped: HashMap<String, Vec<Modification>> = HashMap::new();
        for m in self.all_states.iter().flat_map(|s| s.modifications.clone()) {
            grouped.entry(m.id.clone()).or_default().push(m);
        }

        tracing::info!("Grouped modifications: {:?}", grouped);

        // Store the results here
        let mut to_update = Vec::new();
        let mut to_delete = Vec::new();

        // 2. Analyze each modification group
        for (id, group_mods) in &grouped {
            assert_modifications_consistency(group_mods);

            // Find local node's copy, if any
            let local_copy = self.my_state.modifications.iter().find(|m| &m.id == id);

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
                match local_copy {
                    None => {
                        // If an item is completed for a party, it should at least exist in the
                        // local state because it should have been added during receive_batch.
                        // So, this case should not happen.
                        panic!(
                            "Completed modification could not be found in local state: {:?}",
                            group_mods
                        );
                    }
                    Some(local_m) => {
                        if local_m.status != ModificationStatus::Completed.to_string()
                            || local_m.persisted != any_persisted
                        {
                            // If local is not "completed" or doesn't match the final persisted
                            // We'll roll forward local_m
                            let mut roll_forward = local_m.clone();
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
        assert_eq!(first.serial_id, m.serial_id, "Inconsistent serial IDs");
        assert_eq!(
            first.request_type, m.request_type,
            "Inconsistent request types"
        );
        assert_eq!(first.s3_url, m.s3_url, "Inconsistent S3 URLs");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::smpc_request::{IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE};

    #[test]
    fn test_compare_states_sync() {
        let sync_res = SyncResult {
            my_state: some_state(),
            all_states: vec![some_state(), some_state(), some_state()],
        };
        assert_eq!(sync_res.must_rollback_storage(), None);
    }

    #[test]
    fn test_compare_states_out_of_sync() {
        let states = vec![
            SyncState {
                db_len: 123,
                deleted_request_ids: vec!["most late".to_string()],
                modifications: vec![],
                next_sns_sequence_num: None,
            },
            SyncState {
                db_len: 456,
                deleted_request_ids: vec!["x".to_string(), "y".to_string()],
                modifications: vec![],
                next_sns_sequence_num: None,
            },
            SyncState {
                db_len: 789,
                deleted_request_ids: vec!["most ahead".to_string()],
                modifications: vec![],
                next_sns_sequence_num: None,
            },
        ];
        let deleted_request_ids = vec![
            "most ahead".to_string(),
            "most late".to_string(),
            "x".to_string(),
            "y".to_string(),
        ];

        let sync_res = SyncResult {
            my_state: states[0].clone(),
            all_states: states.clone(),
        };
        assert_eq!(sync_res.must_rollback_storage(), Some(123)); // most late.
        assert_eq!(sync_res.deleted_request_ids(), deleted_request_ids);
    }

    // Helper function to create a Modification.
    fn create_modification(
        id: &str,
        serial_id: i64,
        request_type: &str,
        s3_url: Option<&str>,
        status: ModificationStatus,
        persisted: bool,
    ) -> Modification {
        Modification {
            id: id.to_string(),
            serial_id,
            request_type: request_type.to_string(),
            s3_url: s3_url.map(|s| s.to_string()),
            status: status.to_string(),
            persisted,
        }
    }

    // Create a SyncState with a given vector of modifications.
    fn create_sync_state(modifications: Vec<Modification>) -> SyncState {
        SyncState {
            db_len: modifications.len() as u64,
            deleted_request_ids: vec![],
            modifications,
            next_sns_sequence_num: None,
        }
    }

    #[test]
    fn test_compare_modifications_local_party_outdated() {
        let mod1_local = create_modification(
            "1",
            100,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
        );
        let mod2_local = create_modification(
            "2",
            200,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
        );
        let mod3_local = create_modification(
            "3",
            300,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
        );
        let mod4_local = create_modification(
            "4",
            400,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::InProgress,
            false,
        );
        let my_state = create_sync_state(vec![
            mod1_local.clone(),
            mod2_local.clone(),
            mod3_local.clone(),
            mod4_local.clone(),
        ]);

        let mod1_other = create_modification(
            "1",
            100,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
        );
        let mod2_other = create_modification(
            "2",
            200,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
        );
        let mod3_other = create_modification(
            "3",
            300,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
        );
        let mod4_other = create_modification(
            "4",
            400,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::Completed,
            false,
        );

        let other_state = create_sync_state(vec![
            mod1_other,
            mod2_other,
            mod3_other.clone(),
            mod4_other.clone(),
        ]);
        let all_states = vec![my_state.clone(), other_state.clone(), other_state.clone()];

        let sync_result = SyncResult {
            my_state,
            all_states,
        };

        let (to_update, to_delete) = sync_result.compare_modifications();

        // Expectations:
        // For ID=1: Already in sync → no action.
        // For ID=2: Already in sync → no action.
        // For ID=3: Local is IN_PROGRESS, other nodes are COMPLETED → roll forward to
        // COMPLETED. For ID=4: Local is IN_PROGRESS, other nodes are COMPLETED
        // → roll forward to COMPLETED.
        assert_eq!(to_update.len(), 2, "Expected two modifications to update");
        assert_eq!(to_delete.len(), 0, "Expected zero modification to delete");

        let update_mod3 = to_update.iter().find(|m| m.id == "3").unwrap();
        assert_eq!(update_mod3.clone(), mod3_other);

        let update_mod4 = to_update.iter().find(|m| m.id == "4").unwrap();
        assert_eq!(update_mod4.clone(), mod4_other);
    }

    #[test]
    fn test_compare_modifications_local_party_up_to_date() {
        // Create local modifications that are already up-to-date.
        let mod1_local = create_modification(
            "1",
            100,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
        );
        let mod2_local = create_modification(
            "2",
            200,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
        );
        let mod3_local = create_modification(
            "3",
            300,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
        );
        let mod4_local = create_modification(
            "4",
            400,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::Completed,
            false,
        );
        let my_state = create_sync_state(vec![
            mod1_local.clone(),
            mod2_local.clone(),
            mod3_local.clone(),
            mod4_local.clone(),
        ]);

        // Create other states with in-progress modifications.
        let mod1_other = create_modification(
            "1",
            100,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
        );
        let mod2_other = create_modification(
            "2",
            200,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
        );
        let mod3_other = create_modification(
            "3",
            300,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
        );
        let mod4_other = create_modification(
            "4",
            400,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::InProgress,
            false,
        );
        let other_state = create_sync_state(vec![mod1_other, mod2_other, mod3_other, mod4_other]);
        let all_states = vec![my_state.clone(), other_state.clone(), other_state.clone()];

        let sync_result = SyncResult {
            my_state,
            all_states,
        };

        // Compare modifications across nodes.
        let (to_update, to_delete) = sync_result.compare_modifications();

        // Since local is already the most advanced party, nothing should be updated or
        // deleted.
        assert!(to_update.is_empty(), "Expected no modifications to update");
        assert!(to_delete.is_empty(), "Expected no modifications to delete");
    }

    #[test]
    fn test_compare_modifications_remove_in_progress() {
        // Create local modifications with some in-progress.
        let mod1_local = create_modification(
            "1",
            100,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
        );
        let mod2_local = create_modification(
            "2",
            200,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/200"),
            ModificationStatus::Completed,
            false,
        );
        let mod3_local = create_modification(
            "3",
            300,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
        );
        let mod4_local = create_modification(
            "4",
            400,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/400"),
            ModificationStatus::InProgress,
            false,
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

        let delete_mod3 = to_delete.iter().find(|m| m.id == "3").unwrap();
        assert_eq!(delete_mod3.clone(), mod3_local);

        let delete_mod4 = to_delete.iter().find(|m| m.id == "4").unwrap();
        assert_eq!(delete_mod4.clone(), mod4_local);
    }

    #[test]
    fn test_max_sns_sequence_num() {
        // 1. Test with mixed sequence values
        let states = vec![
            SyncState {
                db_len: 10,
                deleted_request_ids: vec![],
                modifications: vec![],
                next_sns_sequence_num: Some(100),
            },
            SyncState {
                db_len: 20,
                deleted_request_ids: vec![],
                modifications: vec![],
                next_sns_sequence_num: Some(200),
            },
            SyncState {
                db_len: 30,
                deleted_request_ids: vec![],
                modifications: vec![],
                next_sns_sequence_num: None,
            },
        ];

        let sync_result = SyncResult::new(states[0].clone(), states);
        assert_eq!(sync_result.max_sns_sequence_num(), Some(200));

        // 2. Test with all None sequence values
        let state_with_none_sequence_num = SyncState {
            db_len: 10,
            deleted_request_ids: vec![],
            modifications: vec![],
            next_sns_sequence_num: None,
        };
        let all_states = vec![
            state_with_none_sequence_num.clone(),
            state_with_none_sequence_num.clone(),
            state_with_none_sequence_num.clone(),
        ];

        let sync_result_none = SyncResult::new(state_with_none_sequence_num, all_states);
        assert_eq!(sync_result_none.max_sns_sequence_num(), None);
    }

    fn some_state() -> SyncState {
        SyncState {
            db_len: 123,
            deleted_request_ids: vec!["abc".to_string(), "def".to_string()],
            modifications: vec![],
            next_sns_sequence_num: None,
        }
    }
}
