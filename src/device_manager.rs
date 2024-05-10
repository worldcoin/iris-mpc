use std::sync::Arc;

use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{
            event,
            stream::{synchronize, wait_event},
        },
        sys::{CUevent, CUevent_flags},
        CudaDevice, CudaStream,
    },
};

#[derive(Clone)]
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

    pub fn await_streams(&self, streams: &Vec<CudaStream>) {
        streams
            .iter()
            .for_each(|s| unsafe { synchronize(s.stream).unwrap() });
    }

    pub fn create_events(&self) -> Vec<CUevent> {
        let mut events = vec![];
        for idx in 0..self.devices.len() {
            self.devices[idx].bind_to_thread().unwrap();
            events.push(event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap());
        }
        events
    }

    pub fn record_event(&self, streams: &Vec<CudaStream>, events: &Vec<CUevent>) {
        for idx in 0..self.devices.len() {
            unsafe {
                self.devices[idx].bind_to_thread().unwrap();
                event::record(events[idx], streams[idx].stream).unwrap();
            };
        }
    }

    pub fn await_event(&self, streams: &Vec<CudaStream>, events: &Vec<CUevent>) {
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

    pub fn device(&self, index: usize) -> Arc<CudaDevice> {
        self.devices[index].clone()
    }

    pub fn devices(&self) -> &Vec<Arc<CudaDevice>> {
        &self.devices
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}
