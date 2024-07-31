use super::query_processor::{CudaVec2DSlicerU8, StreamAwareCudaSlice};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{
            self, event, malloc_async, memcpy_htod_async,
            stream::{synchronize, wait_event},
        },
        sys::{CUevent, CUevent_flags},
        CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr,
    },
    nccl::{Comm, Id},
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DeviceManager {
    devices: Vec<Arc<CudaDevice>>,
}

impl DeviceManager {
    pub fn init() -> Self {
        let mut devices = vec![];
        for i in 0..CudaDevice::count().unwrap() {
            devices.push(CudaDevice::new(i as usize).unwrap());
        }
        Self { devices }
    }

    /// Splits the devices into n chunks, returning a device manager for each
    /// chunk.
    /// If too few devices are present, returns the original device manager.
    pub fn split_into_n_chunks(self, n: usize) -> Result<Vec<DeviceManager>, DeviceManager> {
        let n_devices = self.devices.len();
        let chunk_size = n_devices / n;
        if chunk_size == 0 {
            return Err(self);
        }
        let mut ret = vec![];
        for i in 0..n {
            ret.push(DeviceManager {
                devices: self.devices[i * chunk_size..(i + 1) * chunk_size].to_vec(),
            });
        }
        Ok(ret)
    }

    pub fn fork_streams(&self) -> Vec<CudaStream> {
        self.devices
            .iter()
            .map(|dev| dev.fork_default_stream().unwrap())
            .collect::<Vec<_>>()
    }

    pub fn create_cublas(&self, streams: &Vec<CudaStream>) -> Vec<CudaBlas> {
        self.devices
            .iter()
            .zip(streams)
            .map(|(dev, stream)| {
                let blas = CudaBlas::new(dev.clone()).unwrap();
                unsafe {
                    blas.set_stream(Some(stream)).unwrap();
                }
                blas
            })
            .collect::<Vec<_>>()
    }

    pub fn await_streams(&self, streams: &[CudaStream]) {
        for i in 0..self.devices.len() {
            self.devices[i].bind_to_thread().unwrap();
            unsafe { synchronize(streams[i].stream).unwrap() }
        }
    }

    pub fn create_events(&self, blocking_sync: bool) -> Vec<CUevent> {
        let flags = if blocking_sync {
            CUevent_flags::CU_EVENT_BLOCKING_SYNC
        } else {
            CUevent_flags::CU_EVENT_DEFAULT
        };

        let mut events = vec![];
        for idx in 0..self.devices.len() {
            self.devices[idx].bind_to_thread().unwrap();
            events.push(event::create(flags).unwrap());
        }
        events
    }

    pub fn record_event(&self, streams: &[CudaStream], events: &[CUevent]) {
        for idx in 0..self.devices.len() {
            unsafe {
                self.devices[idx].bind_to_thread().unwrap();
                event::record(events[idx], streams[idx].stream).unwrap();
            };
        }
    }

    pub fn await_event(&self, streams: &[CudaStream], events: &[CUevent]) {
        for idx in 0..self.devices.len() {
            unsafe {
                self.devices[idx].bind_to_thread().unwrap();
                wait_event(
                    streams[idx].stream,
                    events[idx],
                    cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                )
                .unwrap();
            };
        }
    }

    pub fn htod_transfer_query(
        &self,
        preprocessed_query: &[Vec<u8>],
        streams: &[CudaStream],
    ) -> eyre::Result<CudaVec2DSlicerU8> {
        let mut slices0 = vec![];
        let mut slices1 = vec![];
        for idx in 0..self.device_count() {
            let device = self.device(idx);
            device.bind_to_thread().unwrap();

            let query0 =
                unsafe { malloc_async(streams[idx].stream, preprocessed_query[0].len()).unwrap() };

            let slice0 = StreamAwareCudaSlice::<u8>::upgrade_ptr_stream(
                query0,
                streams[idx].stream,
                preprocessed_query[0].len(),
            );

            unsafe {
                memcpy_htod_async(query0, &preprocessed_query[0], streams[idx].stream).unwrap();
            }

            let query1 =
                unsafe { malloc_async(streams[idx].stream, preprocessed_query[1].len()).unwrap() };

            let slice1 = StreamAwareCudaSlice::<u8>::upgrade_ptr_stream(
                query1,
                streams[idx].stream,
                preprocessed_query[1].len(),
            );

            unsafe {
                memcpy_htod_async(query1, &preprocessed_query[1], streams[idx].stream).unwrap();
            }

            slices0.push(slice0);
            slices1.push(slice1);
        }
        Ok(CudaVec2DSlicerU8 {
            limb_0: slices0,
            limb_1: slices1,
        })
    }

    pub fn device(&self, index: usize) -> Arc<CudaDevice> {
        self.devices[index].clone()
    }

    pub fn devices(&self) -> &[Arc<CudaDevice>] {
        &self.devices
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn htod_copy_into<T: DeviceRepr + Unpin>(
        &self,
        src: Vec<T>,
        dst: &mut CudaSlice<T>,
        index: usize,
    ) -> Result<(), result::DriverError> {
        self.device(index).bind_to_thread()?;
        unsafe { result::memcpy_htod_sync(*dst.device_ptr(), src.as_ref())? };
        Ok(())
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn instantiate_network(&self, peer_id: usize, magic: u64) -> Vec<Arc<Comm>> {
        let n_devices = self.devices.len();
        let mut comms = Vec::with_capacity(n_devices);
        let mut ids = Vec::with_capacity(n_devices);

        if std::env::var("NCCL_COMM_ID").is_err() {
            panic!("NCCL_COMM_ID must be set to <host0_ip:port>");
        }

        for i in 0..n_devices {
            let id = Id::new().unwrap();
            let mut raw = id.internal().to_owned();
            let magic = magic.wrapping_add(u64::try_from(i).unwrap()).to_be_bytes();
            for i in 0..8 {
                raw[i] = magic[i] as ::core::ffi::c_char;
            }

            let id = Id::uninit(raw);

            ids.push(id);
        }

        for i in 0..n_devices {
            // Bind to thread (important!)
            self.devices[i].bind_to_thread().unwrap();
            comms.push(Arc::new(
                Comm::from_rank(self.devices[i].clone(), peer_id, 3, ids[i]).unwrap(),
            ));
        }
        comms
    }
}
