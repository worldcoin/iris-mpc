use cudarc::nccl::Comm;
use eyre::{eyre, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncState {
    pub db_len: u64,
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

pub fn sync(comm: &Comm, state: &SyncState) -> Result<SyncResult> {
    let dev = comm.device();

    let my_state = comm.device().htod_copy(vec![state.db_len]).unwrap();

    let mut all_states = comm.device().alloc_zeros(comm.world_size()).unwrap();

    comm.all_gather(&my_state, &mut all_states)
        .map_err(|e| eyre!("{:?}", e.0))?;

    let all_states = dev.dtoh_sync_copy(&all_states).unwrap();
    println!("all_states: {:?}", all_states);

    let common_state = *all_states.iter().min().unwrap();

    if all_states.iter().all(|s| *s == common_state) {
        Ok(SyncResult::InSync)
    } else {
        Ok(SyncResult::OutOfSync(OutOfSync {
            my_state:     state.clone(),
            common_state: SyncState {
                db_len: common_state,
            },
        }))
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

    #[tokio::test]
    async fn test_sync() -> Result<()> {
        let n_parties = 3.min(CudaDevice::count()? as usize);
        let net_id = Id::new().unwrap();
        let expected_state = SyncState { db_len: 123 };

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
                SyncState { db_len: 123 }
            } else {
                SyncState { db_len: 456 }
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
                    assert_eq!(common_state, SyncState { db_len: 123 });
                }
            }
        }
        Ok(())
    }
}
