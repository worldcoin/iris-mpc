use crate::{helpers::task_monitor::TaskMonitor, threshold_ring::protocol::Circuits};
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
pub struct SyncResult {
    my_state:   SyncState,
    all_states: Vec<SyncState>,
}

impl SyncResult {
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

pub struct Syncer {
    comm:         Arc<Comm>,
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
    Ok(SyncResult {
        my_state: state.clone(),
        all_states,
    })
}

impl SyncState {
    pub const MAX_REQUESTS: usize = 64;
    const SERIAL_SIZE: usize = 8192;

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
        // Make sure we can serialize enough request IDs assuming a maximum length.
        const MAX_REQUEST_ID_LEN: usize = 100;
        // My state.
        let state = SyncState {
            db_len:              123,
            deleted_request_ids: vec!["A".repeat(MAX_REQUEST_ID_LEN); SyncState::MAX_REQUESTS],
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
            assert_eq!(result?.must_rollback_storage(), None);
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
                    db_len:              12, // late
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
            assert_eq!(result?.must_rollback_storage(), Some(12));
        }
        Ok(())
    }

    fn some_state() -> SyncState {
        SyncState {
            db_len:              123,
            deleted_request_ids: vec!["abc".to_string(), "def".to_string()],
        }
    }
}
