use crate::{
    helpers::task_monitor::TaskMonitor,
    threshold_ring::protocol::{Circuits, SendableRcComm},
};
use cudarc::{driver::CudaDevice, nccl::Comm};
use eyre::{eyre, Result};
use std::sync::Arc;
use tokio::task::AbortHandle;

pub type State = u64;

pub struct Syncer {
    comm:         SendableRcComm,
    task_monitor: TaskMonitor,
    server_abort: Option<AbortHandle>,
}

impl Syncer {
    pub fn new(
        peer_id: usize,
        peer_url: Option<String>,
        server_port: u16,
        device: Arc<CudaDevice>,
    ) -> Self {
        let mut task_monitor = TaskMonitor::new();
        let (mut comms, server_abort) = Circuits::instantiate_network(
            peer_id,
            peer_url,
            Some(server_port),
            &[device],
            Some(&mut task_monitor),
        );

        Self {
            comm: comms.pop().unwrap(),
            task_monitor,
            server_abort,
        }
    }

    pub fn sync(&self, state: State) -> Result<State> {
        sync(&self.comm, state)
    }
}

fn sync(comm: &Comm, state: State) -> Result<u64> {
    let dev = comm.device();

    let my_state = comm.device().htod_copy(vec![state]).unwrap();

    let mut all_states = comm.device().alloc_zeros(comm.world_size()).unwrap();

    comm.all_gather(&my_state, &mut all_states)
        .map_err(|e| eyre!("{:?}", e.0))?;

    let all_states = dev.dtoh_sync_copy(&all_states).unwrap();
    println!("all_states: {:?}", all_states);

    let common_state = *all_states.iter().min().unwrap();
    Ok(common_state)
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
        let expected_state = 123;

        let exchange_task = |i, my_state| {
            move || {
                let device = CudaDevice::new(i).unwrap();
                let comm = Comm::from_rank(device, i, n_parties, net_id).unwrap();
                let common_state = sync(&comm, my_state).unwrap();
                common_state
            }
        };

        let mut tasks = JoinSet::new();
        for i in 0..n_parties {
            tasks.spawn_blocking(exchange_task(i, expected_state + i as u64));
        }

        while let Some(common_state) = tasks.join_next().await {
            assert_eq!(common_state?, expected_state);
        }
        Ok(())
    }
}
