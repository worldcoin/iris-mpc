use crate::{helpers::task_monitor::TaskMonitor, threshold_ring::protocol::Circuits};
use cudarc::{driver::CudaDevice, nccl::Comm};
use eyre::{eyre, Result};
use std::{sync::Arc, time::Duration};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncState {
    pub db_len: u64,
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

    pub fn sync(&self, state: &SyncState) -> Result<SyncState> {
        sync(&self.comm, state)
    }

    pub fn stop(&mut self) {
        self.task_monitor.abort_all();
        std::thread::sleep(Duration::from_secs(1));
        self.task_monitor.check_tasks_finished();
    }
}

fn sync(comm: &Comm, state: &SyncState) -> Result<SyncState> {
    let dev = comm.device();

    let my_state = comm.device().htod_copy(vec![state.db_len]).unwrap();

    let mut all_states = comm.device().alloc_zeros(comm.world_size()).unwrap();

    comm.all_gather(&my_state, &mut all_states)
        .map_err(|e| eyre!("{:?}", e.0))?;

    let all_states = dev.dtoh_sync_copy(&all_states).unwrap();
    println!("all_states: {:?}", all_states);

    let common_state = *all_states.iter().min().unwrap();
    Ok(SyncState {
        db_len: common_state,
    })
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

    #[tokio::test]
    async fn test_sync() -> Result<()> {
        let n_parties = 3.min(CudaDevice::count()? as usize);
        let net_id = Id::new().unwrap();
        let expected_state = SyncState { db_len: 123 };

        let sync_task = |i| {
            move || {
                let my_state = SyncState {
                    db_len: expected_state.db_len + i as u64,
                };
                let device = CudaDevice::new(i).unwrap();
                let comm = Comm::from_rank(device, i, n_parties, net_id).unwrap();
                sync(&comm, &my_state).unwrap()
            }
        };

        let mut tasks = JoinSet::new();
        for i in 0..n_parties {
            tasks.spawn_blocking(sync_task(i));
        }

        while let Some(common_state) = tasks.join_next().await {
            assert_eq!(common_state?, expected_state);
        }
        Ok(())
    }
}
