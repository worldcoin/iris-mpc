use super::{
    comm::NcclComm,
    query_processor::{CudaVec2DSlicerRawPointer, CudaVec2DSlicerU8, StreamAwareCudaSlice},
};
use crate::dot::{share_db::SlicedProcessedDatabase, ROTATIONS};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{
            self, event, malloc_async, memcpy_htod_async,
            stream::{synchronize, wait_event},
        },
        sys::{CUevent, CUevent_flags, CU_MEMHOSTALLOC_PORTABLE},
        CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr,
    },
    nccl::Id,
};
use std::{
    sync::{Arc, Mutex},
    thread::sleep,
    time::{Duration, Instant},
};

pub const NCCL_START_WAIT_TIME: Duration = Duration::from_secs(5);
pub const NCCL_START_RETRIES: usize = 5;

#[derive(Debug)]
pub struct DeviceManager {
    devices: Vec<Arc<CudaDevice>>,
    locked_dbs: Arc<Mutex<Vec<CudaVec2DSlicerRawPointer>>>,
}

impl DeviceManager {
    pub fn init() -> Self {
        let mut devices = vec![];
        for i in 0..CudaDevice::count().unwrap() {
            devices.push(CudaDevice::new(i as usize).unwrap());
        }

        tracing::info!("Found {} devices", devices.len());

        Self {
            devices,
            locked_dbs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn init_with_streams() -> Self {
        let mut devices = vec![];
        for i in 0..CudaDevice::count().unwrap() {
            devices.push(CudaDevice::new_with_stream(i as usize).unwrap());
        }

        tracing::info!("Found {} devices", devices.len());

        Self {
            devices,
            locked_dbs: Arc::new(Mutex::new(Vec::new())),
        }
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
        assert!(
            self.locked_dbs.lock().unwrap().is_empty(),
            "Splitting into chunks only supported before pages are locked."
        );
        let mut ret = vec![];
        for i in 0..n {
            ret.push(DeviceManager {
                devices: self.devices[i * chunk_size..(i + 1) * chunk_size].to_vec(),
                locked_dbs: Arc::new(Mutex::new(Vec::new())),
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
            unsafe { synchronize(streams[i].stream).unwrap() }
        }
    }

    pub fn create_events(&self) -> Vec<CUevent> {
        let mut events = vec![];
        for idx in 0..self.devices.len() {
            self.devices[idx].bind_to_thread().unwrap();
            events.push(event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap());
        }
        events
    }

    pub fn destroy_events(&self, events: Vec<CUevent>) {
        for (device_idx, event) in events.iter().enumerate() {
            self.device(device_idx).bind_to_thread().unwrap();
            unsafe { event::destroy(*event).unwrap() };
        }
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
        batch_size: usize,
        code_size: usize,
    ) -> eyre::Result<CudaVec2DSlicerU8> {
        let mut slices0 = vec![];
        let mut slices1 = vec![];
        let query_size = batch_size * ROTATIONS * code_size;
        for idx in 0..self.device_count() {
            let device = self.device(idx);
            device.bind_to_thread().unwrap();

            let query0 = unsafe { malloc_async(streams[idx].stream, query_size).unwrap() };

            let slice0 = StreamAwareCudaSlice::<u8>::upgrade_ptr_stream(
                query0,
                streams[idx].stream,
                query_size,
            );

            // It might happen that the size of preprocessed_query is smaller than
            // query_size, leading to uninitialized memory here. However, all bit-patterns
            // are valid for u8, so this is not a problem as we truncate the results based
            // on the uninit calculations anyway.
            unsafe {
                memcpy_htod_async(query0, &preprocessed_query[0], streams[idx].stream).unwrap();
            }

            let query1 = unsafe { malloc_async(streams[idx].stream, query_size).unwrap() };

            let slice1 = StreamAwareCudaSlice::<u8>::upgrade_ptr_stream(
                query1,
                streams[idx].stream,
                query_size,
            );

            // It might happen that the size of preprocessed_query is smaller than
            // query_size, leading to uninitialized memory here. However, all bit-patterns
            // are valid for u8, so this is not a problem as we truncate the results based
            // on the uninit calculations anyway.
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

    pub fn register_host_memory(
        &self,
        db: &SlicedProcessedDatabase,
        max_db_length: usize,
        code_length: usize,
    ) {
        let max_size = max_db_length / self.device_count();
        for (device_index, device) in self.devices().iter().enumerate() {
            device.bind_to_thread().unwrap();
            unsafe {
                let _ = cudarc::driver::sys::lib().cuMemHostRegister_v2(
                    db.code_gr.limb_0[device_index] as *mut _,
                    max_size * code_length,
                    CU_MEMHOSTALLOC_PORTABLE,
                );

                let _ = cudarc::driver::sys::lib().cuMemHostRegister_v2(
                    db.code_gr.limb_1[device_index] as *mut _,
                    max_size * code_length,
                    CU_MEMHOSTALLOC_PORTABLE,
                );
            }
            self.track_locked_db(db.code_gr.clone());
        }
    }

    // Method to add a db to tracking
    pub fn track_locked_db(&self, db: CudaVec2DSlicerRawPointer) {
        if let Ok(mut locked) = self.locked_dbs.lock() {
            locked.push(db);
        }
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

    /// Derives a set of `Id`s for all devices from a given magic number, which
    /// is required to be the same on all communication parties to establish a
    /// connection.
    ///
    /// The `Id` is first generated by NCCL-internal functionality from the
    /// `NCCL_COMM_ID` environment variable. However, there is currently a
    /// bug that still produces a random magic number for the `Id` generation
    /// internally, even when this env var is set.
    ///
    /// Therefore we overwrite it manually, however, this process is not ideal
    /// as we rely on internal layout of the `Id`, which is not guaranteed to be
    /// stable.
    ///
    /// # Input Constraints
    /// The magic number is expected to be a 64-bit integer. It internally gets
    /// incremented by the number of devices, so calling this functions multiple
    /// times requires to pass magic numbers that are offset by at least the
    /// number of devices.
    #[allow(clippy::unnecessary_cast)]
    pub fn get_ids_from_magic(&self, magic: u64) -> Vec<Id> {
        let n_devices = self.devices.len();
        let mut ids = Vec::with_capacity(n_devices);

        if std::env::var("NCCL_COMM_ID").is_err() {
            panic!("NCCL_COMM_ID must be set to <host0_ip:port>");
        }

        for i in 0..n_devices {
            let id = Id::new().unwrap();
            let mut raw = id.internal().to_owned();
            // Overwrite the magic number, using a different one for each device by just
            // incrementing the magic number
            let magic = magic.wrapping_add(u64::try_from(i).unwrap()).to_be_bytes();
            for i in 0..8 {
                raw[i] = magic[i] as ::core::ffi::c_char;
            }

            let id = Id::uninit(raw);

            ids.push(id);
        }

        ids
    }

    // TODO: check if we can do this nicer, atm we only use the arc to clone it, so
    // a Rc would do.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn instantiate_network_from_ids(
        &self,
        peer_id: usize,
        ids: &[Id],
    ) -> eyre::Result<Vec<Arc<NcclComm>>> {
        let n_devices = self.devices.len();
        let mut comms = Vec::with_capacity(n_devices);

        for i in 0..n_devices {
            self.devices[i].bind_to_thread().unwrap();

            let mut connected = false;
            for _ in 0..NCCL_START_RETRIES {
                if let Ok(c) = NcclComm::from_rank(self.devices[i].clone(), peer_id, 3, ids[i]) {
                    comms.push(Arc::new(c));
                    connected = true;
                    break;
                }
                sleep(NCCL_START_WAIT_TIME);
            }

            if !connected {
                eyre::bail!("Failed to establish NCCL connection");
            }
        }
        Ok(comms)
    }
}

impl Drop for DeviceManager {
    fn drop(&mut self) {
        let page_unlock_ts = Instant::now();
        // Get all tracked DBs
        if let Ok(locked_dbs) = self.locked_dbs.lock() {
            tracing::info!(
                "DeviceManager is dropping memory for {:?} number of dbs",
                locked_dbs.len()
            );
            for db in locked_dbs.iter() {
                // For each DB, unregister all its memory
                for (device_index, device) in self.devices.iter().enumerate() {
                    device.bind_to_thread().unwrap();
                    unsafe {
                        let _ = cudarc::driver::sys::lib()
                            .cuMemHostUnregister(db.limb_0[device_index] as *mut _);
                        let _ = cudarc::driver::sys::lib()
                            .cuMemHostUnregister(db.limb_1[device_index] as *mut _);
                    }
                }
            }
        }
        tracing::info!(
            "DeviceManager dropped, all memory unregistered in {:?} ",
            page_unlock_ts.elapsed()
        );
    }
}
