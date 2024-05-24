//! # NCCL DMA bench
//! This script establishes a pairwise connection via NCCL between all devices of two hosts.
//! Each device pair gets its separate NCCL comm channel, with the host device being rank 0.
//! It also starts a HTTP server on the host on port 3000 to exchange the NCCL COMM_IDs.
//! Host: NCCL_DEBUG=INFO cargo run --release --bin nccl 0
//! Node: NCCL_DEBUG=INFO cargo run --release --bin nccl {1,2} HOST_IP:3000

use std::{env, ffi::c_void, str::FromStr, time::Instant};

use axum::{extract::Path, routing::get, Router};
use cudarc::{
    driver::{
        result::event::{self, elapsed},
        sys::CUevent_flags,
        CudaDevice, CudaSlice, DevicePtr, DeviceSlice,
    },
    nccl::{group_end, group_start, result, sys, Comm, Id},
};
use once_cell::sync::Lazy;

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

const DUMMY_DATA_LEN: usize = 558 * (1 << 20);

async fn root(Path(device_id): Path<String>) -> String {
    let device_id: usize = device_id.parse().unwrap();
    IdWrapper(COMM_ID[device_id]).to_string()
}

#[tokio::main]
async fn main() {
    let args = env::args().collect::<Vec<_>>();
    let n_devices = CudaDevice::count().unwrap() as usize;
    let party_id: usize = args[1].parse().unwrap();

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
    let mut slices1 = vec![];
    let mut slices2 = vec![];
    // let mut slices3 = vec![];

    for i in 0..n_devices {
        let id = if party_id == 0 {
            COMM_ID[i]
        } else {
            let res = reqwest::blocking::get(format!("http://{}/{}", args[2], i)).unwrap();
            IdWrapper::from_str(&res.text().unwrap()).unwrap().0
        };

        let dev = CudaDevice::new(i).unwrap();
        let slice: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
        let slice1: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
        let slice2: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
        // let slice3: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();

        println!("starting device {i}...");

        let comm = Comm::from_rank(dev.clone(), party_id, 2, id).unwrap();

        devs.push(dev);
        comms.push(comm);
        slices.push(slice);
        slices1.push(slice1);
        slices2.push(slice2);
        // slices3.push(slice3);
    }

    for _ in 0..10 {
        let now = Instant::now();

        let mut events = vec![];

        for i in 0..n_devices {
            devs[i].bind_to_thread().unwrap();

            for _ in 0..2 {
                let stream = devs[i].fork_default_stream().unwrap();
                let start = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
                unsafe {
                    event::record(start, stream.stream).unwrap();
                }
                // comms[i]
                //     .broadcast(&Some(&slices[i]), &mut slices1[i], 0)
                //     .unwrap();
                // comms[i]
                //     .broadcast(&Some(&slices[i]), &mut slices2[i], 1)
                //     .unwrap();

                unsafe {
                    result::broadcast(
                        *slices[i].device_ptr() as *mut c_void,
                        *slices1[i].device_ptr() as *mut c_void,
                        slices[i].len(),
                        sys::ncclDataType_t::ncclUint8,
                        0,
                        comms[i].comm,
                        stream.stream as *mut _,
                    )
                    .unwrap();

                    result::broadcast(
                        *slices[i].device_ptr() as *mut c_void,
                        *slices2[i].device_ptr() as *mut c_void,
                        slices[i].len(),
                        sys::ncclDataType_t::ncclUint8,
                        1,
                        comms[i].comm,
                        stream.stream as *mut _,
                    )
                    .unwrap();
                }

                // comms[i].broadcast(&Some(&slices[i]), &mut slices3[i], 2).unwrap();
                let end = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
                unsafe {
                    event::record(end, stream.stream).unwrap();
                }
                events.push((start, end));
            }
        }

        for i in 0..n_devices {
            devs[i].synchronize().unwrap();
        }

        for (i, event) in events.iter().enumerate() {
            devs[i].bind_to_thread().unwrap();
            unsafe {
                let time = elapsed(event.0, event.1).unwrap();
                println!("device {} took {:?}", i, time);
            }
        }

        if party_id != 0 {
            let elapsed = now.elapsed();
            // Throughput multiplied by 4 because every device sends *and* receives the buffer to/from two peers.
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
}
