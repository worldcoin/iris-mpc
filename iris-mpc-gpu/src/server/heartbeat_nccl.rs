use crate::helpers::device_manager::{DeviceManager, NCCL_START_RETRIES, NCCL_START_WAIT_TIME};
use cudarc::driver::CudaSlice;
use eyre::{eyre, Context};
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::{mpsc, oneshot},
    task::{spawn_blocking, JoinHandle},
    time::timeout,
};

pub async fn start_heartbeat(
    party_id: usize,
    main_tx: oneshot::Sender<eyre::Result<()>>,
    heartbeat_interval_secs: u64,
) -> eyre::Result<()> {
    let (tx, mut rx) = mpsc::channel(1);
    let heartbeat_interval = Duration::from_secs(heartbeat_interval_secs);

    let heartbeat_handle: JoinHandle<eyre::Result<()>> = spawn_blocking(move || {
        let device_manager = Arc::new(DeviceManager::init_with_streams());
        let ids = device_manager.get_ids_from_magic(0xdead);

        let comms = device_manager.instantiate_network_from_ids(party_id, &ids)?;

        tracing::info!("Heartbeat: NCCL connection established");

        // notify the main thread that we are ready
        main_tx
            .send(Ok(()))
            .map_err(|e| eyre!("Failed to send heartbeat ready signal: {:?}", e))?;

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
            tracing::info!("Heartbeat: {}", counter);
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
            std::thread::sleep(heartbeat_interval);
            counter += 1;
        }
    });

    let mut timeout_interval = 10
        * NCCL_START_WAIT_TIME
        * (NCCL_START_RETRIES - 1).try_into()?
        * DeviceManager::init().device_count().try_into()?;
    loop {
        match timeout(timeout_interval, rx.recv()).await {
            // The first heartbeat might take a while due to retries. However, after the connection
            // is established, we switch to the normal heartbeat interval.
            Ok(Some(Ok(_))) => timeout_interval = 2 * heartbeat_interval,
            Ok(None) => {
                tracing::error!("Heartbeat: Channel closed.");
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
