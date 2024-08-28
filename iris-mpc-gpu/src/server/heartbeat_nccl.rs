use crate::helpers::device_manager::DeviceManager;
use cudarc::driver::CudaSlice;
use eyre::{eyre, Context};
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::mpsc,
    task::{spawn_blocking, JoinHandle},
    time::timeout,
};
const HEARBEAT_INTERVAL: Duration = Duration::from_secs(5);

pub async fn start_heartbeat(party_id: usize) -> eyre::Result<()> {
    let (tx, mut rx) = mpsc::channel(1);

    let heartbeat_handle: JoinHandle<eyre::Result<()>> = spawn_blocking(move || {
        let device_manager = Arc::new(DeviceManager::init());
        let ids = device_manager.get_ids_from_magic(0xdead);
        let comms = device_manager.instantiate_network_from_ids(party_id, &ids)?;

        let mut pings = vec![];
        let mut pongs = vec![];
        for comm in comms.iter() {
            let ping: CudaSlice<u64> =
                comm.device().alloc_zeros(1).context("Failed to allocate")?;
            let pong: CudaSlice<u64> = comm
                .device()
                .alloc_zeros(comm.world_size())
                .context("Failed to allocate")?;
            pings.push(ping);
            pongs.push(pong);
        }

        let mut counter: u64 = 0;
        loop {
            for i in 0..comms.len() {
                tx.blocking_send(|| -> eyre::Result<()> {
                    device_manager
                        .device(i)
                        .htod_copy_into(vec![counter], &mut pings[i])?;
                    comms[i]
                        .all_gather(&pings[i], &mut pongs[i])
                        .map_err(|e| eyre!(format!("{:?}", e)))?;

                    let pongs_host = device_manager.device(i).dtoh_sync_copy(&pongs[i])?;
                    if !pongs_host.iter().all(|&x| x == counter) {
                        return Err(eyre!("Heartbeat counter mismatch"));
                    }

                    Ok(())
                }())?;
            }
            std::thread::sleep(HEARBEAT_INTERVAL);
            counter += 1;
        }
    });

    loop {
        match timeout(HEARBEAT_INTERVAL * 2, rx.recv()).await {
            Ok(Some(Ok(_))) => continue,
            Ok(None) => {
                tracing::error!("Heartbeat channel closed.");
                break;
            }
            Ok(Some(Err(e))) => {
                tracing::error!("Heartbeat failed: {:?}", e);
                panic!("Heartbeat failed, restarting service");
            }
            Err(_) => {
                tracing::error!("Heartbeat timeout.");
                panic!("Heartbeat timeout, restarting service");
            }
        }
    }

    if let Err(e) = heartbeat_handle.await? {
        tracing::error!("Heartbeat handle exited: {:?}", e);
    }

    Ok(())
}
