use crate::{
    helpers::task_monitor::TaskMonitor,
    threshold_ring::protocol::{Circuits, SendableRcComm},
};
use cudarc::{driver::CudaDevice, nccl::Comm};
use eyre::{eyre, Result};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    pub db_len:              u64,
    pub deleted_request_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncResult {
    InSync,
    OutOfSync(OutOfSync),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfSync {
    #[allow(dead_code)]
    pub my_state:     SyncState,
    pub common_state: SyncState,
}

pub struct Syncer {
    comm:         SendableRcComm,
    task_monitor: TaskMonitor,
}

impl Syncer {
    pub fn new(
        peer_id: usize,
        peer_url: Option<String>,
        server_port: u16,
        device: Arc<CudaDevice>,
    ) -> Self {
        let mut task_monitor = TaskMonitor::new();
        let (mut comms, _server_abort) = Circuits::instantiate_network(
            peer_id,
            peer_url,
            Some(server_port),
            &[device],
            Some(&mut task_monitor),
        );
        task_monitor.check_tasks();

        Self {
            comm: comms.pop().unwrap(),
            task_monitor,
        }
    }

    pub fn sync(&self, state: &SyncState) -> Result<SyncResult> {
        sync(&self.comm, state)
    }

    pub fn stop(&mut self) {
        self.task_monitor.abort_all();
        std::thread::sleep(Duration::from_secs(1));
        self.task_monitor.check_tasks_finished();
    }
}

fn sync(comm: &Comm, state: &SyncState) -> Result<SyncResult> {
    let dev = comm.device();

    let state_dev = comm.device().htod_copy(state.serialize()?).unwrap();
    let mut all_states_dev = comm
        .device()
        .alloc_zeros(state_dev.len * comm.world_size())
        .unwrap();

    comm.all_gather(&state_dev, &mut all_states_dev)
        .map_err(|e| eyre!("{:?}", e.0))?;

    let all_states_ser = dev.dtoh_sync_copy(&all_states_dev).unwrap();
    let all_states = SyncState::deserialize_all(&all_states_ser)?;
    println!("all_states: {:?}", all_states);

    Ok(SyncState::compare_states(state, &all_states))
}

impl SyncState {
    pub const MAX_REQUESTS: usize = 32;
    const SERIAL_SIZE: usize = 4096;

    /// Serialize the state to a fixed-size buffer.
    fn serialize(&self) -> Result<Vec<u8>> {
        let mut state_ser = vec![0; 8];
        serde_json::to_writer(&mut state_ser, self)?;
        // Frame with the buffer length.
        let buf_len = state_ser.len();
        if buf_len > Self::SERIAL_SIZE {
            return Err(eyre!("State too large to serialize"));
        }
        state_ser[..8].copy_from_slice(&(buf_len as u64).to_le_bytes());
        // Pad to fixed size.
        state_ser.resize(Self::SERIAL_SIZE, 0);
        Ok(state_ser)
    }

    /// Deserialize the state from a fixed-size buffer.
    fn deserialize(state_ser: &[u8]) -> Result<Self> {
        // Unframe the buffer.
        let buf_len = u64::from_le_bytes(state_ser[..8].try_into().unwrap()) as usize;
        if buf_len > Self::SERIAL_SIZE {
            return Err(eyre!("State too large to deserialize"));
        }
        let state = serde_json::from_slice(&state_ser[8..buf_len])?;
        Ok(state)
    }

    /// Deserialize all states concatenated in a buffer.
    fn deserialize_all(state_ser: &[u8]) -> Result<Vec<Self>> {
        state_ser
            .chunks(Self::SERIAL_SIZE)
            .map(Self::deserialize)
            .collect()
    }

    fn compare_states(my_state: &Self, states: &[Self]) -> SyncResult {
        let common_state = SyncState {
            db_len:              Self::find_most_late(states).db_len,
            deleted_request_ids: Self::merge_request_ids(states),
        };

        if states.iter().all(|s| s.db_len == common_state.db_len) {
            SyncResult::InSync
        } else {
            SyncResult::OutOfSync(OutOfSync {
                my_state: my_state.clone(),
                common_state,
            })
        }
    }

    fn find_most_late(states: &[Self]) -> Self {
        states.iter().min_by_key(|s| s.db_len).unwrap().clone()
    }

    fn merge_request_ids(states: &[Self]) -> Vec<String> {
        states
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
    use cudarc::{
        driver::CudaDevice,
        nccl::{Comm, Id},
    };
    use eyre::Result;
    use tokio::task::JoinSet;

    #[test]
    fn test_serialize() -> Result<()> {
        // My state.
        let state = SyncState {
            db_len:              123,
            deleted_request_ids: vec!["A".repeat(64); SyncState::MAX_REQUESTS],
        };
        let state_ser = state.serialize()?;
        assert_eq!(state_ser.len(), SyncState::SERIAL_SIZE);
        // Concatenation of states from 3 parties.
        let all_states_ser = vec![state_ser.clone(); 3].concat();
        let all_states = SyncState::deserialize_all(&all_states_ser)?;

        for s in all_states.iter() {
            assert_eq!(s, &state);
        }
        Ok(())
    }

    #[test]
    fn test_compare_states_sync() {
        let my_state = some_state();
        let states = vec![some_state(), some_state(), some_state()];
        assert_eq!(
            SyncState::compare_states(&my_state, &states),
            SyncResult::InSync
        );
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
                deleted_request_ids: vec!["x".to_string()],
            },
            SyncState {
                db_len:              789,
                deleted_request_ids: vec!["most ahead".to_string()],
            },
        ];
        let common_state = SyncState {
            db_len:              123, // most late.
            deleted_request_ids: vec![
                "most ahead".to_string(),
                "most late".to_string(),
                "x".to_string(),
            ],
        };

        let my_state = states[0].clone();
        assert_eq!(
            SyncState::compare_states(&my_state, &states),
            SyncResult::OutOfSync(OutOfSync {
                my_state,
                common_state,
            })
        );
    }

    #[tokio::test]
    async fn test_sync() -> Result<()> {
        let n_parties = 3.min(CudaDevice::count()? as usize);
        let net_id = Id::new().unwrap();
        let expected_state = some_state();

        let sync_task = |i| {
            let my_state = expected_state.clone();
            move || {
                let device = CudaDevice::new(i).unwrap();
                let comm = Comm::from_rank(device, i, n_parties, net_id).unwrap();
                sync(&comm, &my_state).unwrap()
            }
        };

        let mut tasks = JoinSet::new();
        for i in 0..n_parties {
            tasks.spawn_blocking(sync_task(i));
        }

        while let Some(result) = tasks.join_next().await {
            assert_eq!(result?, SyncResult::InSync);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_out_of_sync() -> Result<()> {
        let n_parties = 3.min(CudaDevice::count()? as usize);
        let net_id = Id::new().unwrap();

        let sync_task = |i| {
            let my_state = if i == 0 {
                some_state()
            } else {
                SyncState {
                    db_len:              456,
                    deleted_request_ids: vec![],
                }
            };
            move || {
                let device = CudaDevice::new(i).unwrap();
                let comm = Comm::from_rank(device, i, n_parties, net_id).unwrap();
                sync(&comm, &my_state).unwrap()
            }
        };

        let mut tasks = JoinSet::new();
        for i in 0..n_parties {
            tasks.spawn_blocking(sync_task(i));
        }

        while let Some(result) = tasks.join_next().await {
            match result? {
                SyncResult::InSync => panic!("Expected OutOfSync"),
                SyncResult::OutOfSync(OutOfSync {
                    my_state: _,
                    common_state,
                }) => {
                    assert_eq!(common_state, some_state());
                }
            }
        }
        Ok(())
    }

    fn some_state() -> SyncState {
        SyncState {
            db_len:              123,
            deleted_request_ids: vec![],
        }
    }
}
