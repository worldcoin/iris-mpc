//! # NCCL DMA bench
//! This script establishes a pairwise connection via NCCL between all devices
//! of two hosts. Each device pair gets its separate NCCL comm channel, with the
//! host device being rank 0. It also starts a HTTP server on the host on port
//! 3000 to exchange the NCCL COMM_IDs. Host: NCCL_DEBUG=INFO cargo run
//! --release -p iris-mpc-bins --bin nccl 0 Node: NCCL_DEBUG=INFO cargo run
//! --release -p iris-mpc-bins --bin nccl
//! {1,2} HOST_IP:3000

use axum::{extract::Path, routing::get, Router};
use cudarc::{
    driver::{CudaDevice, CudaSlice},
    nccl::{Comm, Id},
};
use eyre::Result;
use iris_mpc_gpu::helpers::id_wrapper::IdWrapper;
use std::{env, str::FromStr, sync::LazyLock, time::Instant};

static COMM_ID: LazyLock<Vec<Id>> = LazyLock::new(|| {
    (0..CudaDevice::count().unwrap())
        .map(|_| Id::new().unwrap())
        .collect::<Vec<_>>()
});

const DUMMY_DATA_LEN: usize = 5 * (1 << 30);

async fn root(Path(device_id): Path<String>) -> String {
    let device_id: usize = device_id.parse().unwrap();
    IdWrapper(COMM_ID[device_id]).to_string()
}

#[tokio::main(flavor = "multi_thread", worker_threads = 12)]
async fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    let n_devices = CudaDevice::count().unwrap() as usize;
    let party_id: usize = args[1].parse().unwrap();

    let mut server_join_handle = None;

    if party_id == 0 {
        server_join_handle = Some(tokio::spawn(async move {
            println!("starting server...");
            let app = Router::new().route("/:device_id", get(root));
            let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
            axum::serve(listener, app).await.unwrap();
        }));
    };

    let mut devs = vec![];
    let mut comms = vec![];
    let mut slices = vec![];
    let mut slices1 = vec![];
    let mut slices2 = vec![];
    let mut slices3 = vec![];

    for i in 0..n_devices {
        let id = if party_id == 0 {
            COMM_ID[i]
        } else {
            let res = reqwest::blocking::get(format!("http://{}/{}", args[2], i)).unwrap();
            IdWrapper::from_str(&res.text().unwrap()).unwrap().0
        };

        // This call to CudaDevice::new is only used in context of a benchmark - not
        // used in the server binary
        let dev = CudaDevice::new(i).unwrap();
        let slice: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
        let slice1: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
        let slice2: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
        let slice3: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();

        println!("starting device {i}...");

        let comm = Comm::from_rank(dev.clone(), party_id, 3, id).unwrap();

        devs.push(dev);
        comms.push(comm);
        slices.push(Some(slice));
        slices1.push(slice1);
        slices2.push(slice2);
        slices3.push(slice3);
    }

    for _ in 0..10 {
        let now = Instant::now();

        for i in 0..n_devices {
            devs[i].bind_to_thread().unwrap();

            comms[i]
                .broadcast(slices[i].as_ref(), &mut slices1[i], 0)
                .unwrap();
            comms[i]
                .broadcast(slices[i].as_ref(), &mut slices2[i], 1)
                .unwrap();
            comms[i]
                .broadcast(slices[i].as_ref(), &mut slices3[i], 2)
                .unwrap();
        }

        for dev in devs.iter() {
            dev.synchronize().unwrap();
        }

        if party_id != 0 {
            let elapsed = now.elapsed();
            // Throughput multiplied by 4 because every device sends *and* receives the
            // buffer to/from two peers.
            let throughput = (DUMMY_DATA_LEN as f64 * n_devices as f64 * 4f64)
                / (elapsed.as_millis() as f64)
                / 1_000_000_000f64
                * 1_000f64;
            println!(
                "received in {:?} [{:.2} GB/s] [{:.2} Gbps]",
                elapsed,
                throughput,
                throughput * 8f64
            );
        }
    }

    // Shut down the server, making sure it hasn't panicked or errored.
    if let Some(handle) = server_join_handle {
        handle.abort();
        handle.await?;
    }

    Ok(())
}
