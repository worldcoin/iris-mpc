//! # NCCL DMA bench
//! This script establishes a pairwise connection via NCCL between all devices
//! of two hosts. Each device pair gets its separate NCCL comm channel, with the
//! host device being rank 0. It also starts a HTTP server on the host on port
//! 3000 to exchange the NCCL COMM_IDs. Host: NCCL_DEBUG=INFO cargo run
//! --release --bin nccl 0 Node: NCCL_DEBUG=INFO cargo run --release --bin nccl
//! {1,2} HOST_IP:3000

use axum::{extract::Path, routing::get, Router};
use cudarc::{
    driver::{CudaDevice, CudaSlice},
    nccl::{group_end, group_start, Comm, Id},
};
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
async fn main() -> eyre::Result<()> {
    let args = env::args().collect::<Vec<_>>();
    let n_devices = CudaDevice::count().unwrap() as usize;
    let party_id: i32 = args[1].parse().unwrap();

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
    let mut slices_send = vec![];
    let mut slices_recv = vec![];

    for i in 0..n_devices {
        let id = if party_id == 0 {
            COMM_ID[i]
        } else {
            let res = reqwest::blocking::get(format!("http://{}/{}", args[2], i)).unwrap();
            IdWrapper::from_str(&res.text().unwrap()).unwrap().0
        };

        let dev = CudaDevice::new(i).unwrap();
        let slice_send: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
        let slice_recv: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();

        println!("starting device {i}...");

        let comm = Comm::from_rank(dev.clone(), party_id as usize, 2, id).unwrap();

        devs.push(dev);
        comms.push(comm);
        slices_send.push(slice_send);
        slices_recv.push(slice_recv);
    }

    for _ in 0..10 {
        let now = Instant::now();

        group_start().unwrap();
        for i in 0..n_devices {
            devs[i].bind_to_thread().unwrap();
            comms[i].send(&slices_send[i], (party_id + 1) % 2).unwrap();
            comms[i]
                .recv(&mut slices_recv[i], (party_id + 1) % 2)
                .unwrap();
        }
        group_end().unwrap();

        for dev in devs.iter() {
            dev.synchronize().unwrap();
        }

        if party_id != 0 {
            let elapsed = now.elapsed();
            let throughput = (DUMMY_DATA_LEN as f64 * n_devices as f64 * 2f64)
                / (elapsed.as_millis() as f64)
                / (1 << 20);
            println!(
                "received in {:?} [{:.2} GiB/s] [{:.2} Gbps]",
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
