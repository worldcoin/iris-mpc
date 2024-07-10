use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{
            self, event, malloc_async, memcpy_htod_async,
            stream::{synchronize, wait_event},
        },
        sys::{CUdeviceptr, CUevent, CUevent_flags, CUevent_wait_flags, CUstream},
        CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr,
    },
};
use std::{marker::PhantomData, sync::Arc};
use tokio::task;

#[derive(Clone)]
pub struct DeviceManager {
    devices: Vec<Arc<CudaDevice>>,
}

#[derive(Clone)]
struct CUeventWrapper {
    event:   CUevent,
    _marker: PhantomData<*mut ()>, // PhantomData to make it `!Send` by default
}

unsafe impl Send for CUeventWrapper {}
unsafe impl Sync for CUeventWrapper {}

#[derive(Clone)]
struct CUstreamWrapper {
    stream:  CUstream,
    _marker: PhantomData<*mut ()>, // PhantomData to make it `!Send` by default
}

unsafe impl Send for CUstreamWrapper {}
unsafe impl Sync for CUstreamWrapper {}

impl DeviceManager {
    pub fn init() -> Self {
        let mut devices = vec![];
        for i in 0..CudaDevice::count().unwrap() {
            devices.push(CudaDevice::new(i as usize).unwrap());
        }
        Self { devices }
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

    pub fn create_events(&self) -> Vec<Arc<CUeventWrapper>> {
        let mut events = vec![];
        for idx in 0..self.devices.len() {
            self.devices[idx].bind_to_thread().unwrap();
            let event = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
            events.push(Arc::new(CUeventWrapper {
                event,
                _marker: PhantomData,
            }));
        }
        events
    }

    pub async fn await_event(&self, streams: &[CUstreamWrapper], events: &[Arc<CUeventWrapper>]) {
        let tasks: Vec<_> = (0..self.devices.len())
            .map(|idx| {
                let device = self.devices[idx].clone();
                let stream = streams[idx].clone();
                let event = events[idx].clone(); // Clone Arc to share the event safely
                tokio::task::spawn_blocking(move || unsafe {
                    device.bind_to_thread().unwrap();
                    cudarc::driver::sys::wait_event(
                        stream.stream,
                        event.event,
                        cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                    )
                    .unwrap();
                })
            })
            .collect();

        // Await all tasks
        futures::future::join_all(tasks).await;
    }

    pub async fn await_streams(&self, streams: &[CUstreamWrapper]) {
        let tasks: Vec<_> = (0..self.devices.len())
            .map(|i| {
                let stream = streams[i].clone();
                tokio::task::spawn_blocking(move || unsafe {
                    cudarc::driver::sys::synchronize(stream.stream).unwrap();
                })
            })
            .collect();

        // Await all tasks
        futures::future::join_all(tasks).await;
    }

    pub fn htod_transfer_query(
        &self,
        preprocessed_query: &[Vec<u8>],
        streams: &[CudaStream],
    ) -> (Vec<CUdeviceptr>, Vec<CUdeviceptr>) {
        let mut query0_ptrs = vec![];
        let mut query1_ptrs = vec![];
        for idx in 0..self.device_count() {
            self.device(idx).bind_to_thread().unwrap();
            let query0 =
                unsafe { malloc_async(streams[idx].stream, preprocessed_query[0].len()).unwrap() };
            unsafe {
                memcpy_htod_async(query0, &preprocessed_query[0], streams[idx].stream).unwrap();
            }

            let query1 =
                unsafe { malloc_async(streams[idx].stream, preprocessed_query[1].len()).unwrap() };
            unsafe {
                memcpy_htod_async(query1, &preprocessed_query[1], streams[idx].stream).unwrap();
            }

            query0_ptrs.push(query0);
            query1_ptrs.push(query1);
        }
        (query0_ptrs, query1_ptrs)
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
}
