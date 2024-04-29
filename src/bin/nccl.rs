//! # NCCL DMA bench
//! This script establishes a pairwise connection via NCCL between all devices of two hosts.
//! Each device pair gets its separate NCCL comm channel, with the host device being rank 0.
//! It also starts a HTTP server on the host on port 3000 to exchange the NCCL COMM_IDs.
//! Host: cargo run --release --bin nccl 0
//! Node: cargo run --release --bin nccl 1 HOST_IP:3000

use std::{
    env,
    str::FromStr,
    sync::{Arc, Barrier},
    thread::{self, JoinHandle},
    time::Instant,
};

use atomic_float::AtomicF64;
use axum::{extract::Path, routing::get, Router};
use cudarc::{
    driver::{CudaDevice, CudaSlice},
    nccl::{Comm, Id},
};
use once_cell::sync::Lazy;
use std::sync::atomic::Ordering::{Acquire, SeqCst};

static COMM_ID: Lazy<Vec<Id>> = Lazy::new(|| {
    (0..CudaDevice::count().unwrap())
        .map(|_| Id::new().unwrap())
        .collect::<Vec<_>>()
});

struct IdWrapper(Id);

impl FromStr for IdWrapper {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = hex::decode(s)
            .unwrap()
            .iter()
            .map(|&c| c as i8)
            .collect::<Vec<_>>();

        let mut id = [0i8; 128];
        id.copy_from_slice(&bytes);

        Ok(IdWrapper(Id::uninit(id)))
    }
}

impl ToString for IdWrapper {
    fn to_string(&self) -> String {
        hex::encode(
            self.0
                .internal()
                .iter()
                .map(|&c| c as u8)
                .collect::<Vec<_>>(),
        )
    }
}

const DUMMY_DATA_LEN: usize = 20 * (1 << 30);

async fn root(Path(device_id): Path<String>) -> String {
    let device_id: usize = device_id.parse().unwrap();
    IdWrapper(COMM_ID[device_id]).to_string()
}

#[tokio::main(flavor = "multi_thread", worker_threads = 12)]
async fn main() -> eyre::Result<()> {
    let args = env::args().collect::<Vec<_>>();
    let n_devices = CudaDevice::count().unwrap() as usize;
    let party_id: usize = args[1].parse().unwrap();
    let peer_party: i32 = (party_id as i32 + 1) % 3;

    if party_id == 0 {
        tokio::spawn(async move {
            println!("starting server...");
            let app = Router::new().route("/:device_id", get(root));
            let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    };

    let mut devs = vec![];
    let mut comms = vec![];
    let mut slices = vec![];

    for i in 0..n_devices {
        let id = if party_id == 0 {
            COMM_ID[i]
        } else {
            let res = reqwest::blocking::get(format!("http://{}/{}", args[2], i)).unwrap();
            IdWrapper::from_str(&res.text().unwrap()).unwrap().0
        };

        let dev = CudaDevice::new(i).unwrap();
        let slice: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();

        println!("starting device {i}...");

        let comm = Comm::from_rank(dev.clone(), party_id, 3, id).unwrap();

        devs.push(dev);
        comms.push(comm);
        slices.push(slice);
    }

    for _ in 0..10 {
        let now = Instant::now();

        for i in 0..n_devices {
            devs[i].bind_to_thread().unwrap();
            match party_id {
                0 => {
                    comms[i].send(&slices[i], 1).unwrap();
                    comms[i].recv(&mut slices[i], 1).unwrap();

                    comms[i].send(&slices[i], 2).unwrap();
                    comms[i].recv(&mut slices[i], 2).unwrap();
                }
                1 => {
                    comms[i].recv(&mut slices[i], 0).unwrap();
                    comms[i].send(&slices[i], 0).unwrap();

                    comms[i].send(&slices[i], 2).unwrap();
                    comms[i].recv(&mut slices[i], 2).unwrap();
                }
                2 => {
                    comms[i].recv(&mut slices[i], 0).unwrap();
                    comms[i].send(&slices[i], 0).unwrap();

                    comms[i].recv(&mut slices[i], 2).unwrap();
                    comms[i].send(&slices[i], 2).unwrap();
                }
                _ => unimplemented!()
            }
        }

        for i in 0..n_devices {
            devs[i].synchronize().unwrap();
        }

        if party_id != 0 {
            let elapsed = now.elapsed();
            let throughput =
                (DUMMY_DATA_LEN as f64 * 8f64) / (elapsed.as_millis() as f64) / 1_000_000_000f64
                    * 1_000f64;
            println!(
                "received in {:?} [{:.2} GB/s] [{:.2} Gbps]",
                elapsed,
                throughput,
                throughput * 8f64
            );
        }
    }

    Ok(())
}
