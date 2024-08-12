use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    pub db_len:              u64,
    pub deleted_request_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncResult {
    my_state:   SyncState,
    all_states: Vec<SyncState>,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_states_sync() {
        let sync_res = SyncResult {
            my_state:   some_state(),
            all_states: vec![some_state(), some_state(), some_state()],
        };
        assert_eq!(sync_res.must_rollback_storage(), None);
    }

    #[test]
    fn test_compare_states_out_of_sync() {
        let states = vec![
            SyncState {
                db_len:              123,
                deleted_request_ids: vec!["most late".to_string()],
            },
            SyncState {
                db_len:              456,
                deleted_request_ids: vec!["x".to_string(), "y".to_string()],
            },
            SyncState {
                db_len:              789,
                deleted_request_ids: vec!["most ahead".to_string()],
            },
        ];
        let deleted_request_ids = vec![
            "most ahead".to_string(),
            "most late".to_string(),
            "x".to_string(),
            "y".to_string(),
        ];

        let sync_res = SyncResult {
            my_state:   states[0].clone(),
            all_states: states.clone(),
        };
        assert_eq!(sync_res.must_rollback_storage(), Some(123)); // most late.
        assert_eq!(sync_res.deleted_request_ids(), deleted_request_ids);
    }

    fn some_state() -> SyncState {
        SyncState {
            db_len:              123,
            deleted_request_ids: vec!["abc".to_string(), "def".to_string()],
        }
    }
}
