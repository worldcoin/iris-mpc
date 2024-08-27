use cudarc::{
    driver::{CudaDevice, CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, DeviceSlice},
    nccl::{result, sys, Id, NcclType},
};
use std::{mem::MaybeUninit, sync::Arc};

#[derive(Debug)]
pub struct Comm {
    comm:       sys::ncclComm_t,
    device:     Arc<CudaDevice>,
    rank:       usize,
    world_size: usize,
}

// creation methods
// copied from cudarc, under MIT License, (C) Corey Lowman
// Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option.
impl Comm {
    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Primitive to create new communication link on each process (threads are
    /// possible but not recommended).
    ///
    /// WARNING: If using threads, uou are likely to get limited throughput
    /// using a single core to control multiple GPUs. Cuda drivers
    /// effectively use a global mutex thrashing performance on multi
    /// threaded multi GPU. ```
    /// # use cudarc::driver::safe::{CudaDevice};
    /// # use cudarc::nccl::safe::{Comm, Id, ReduceOp};
    /// let n = 2;
    /// let n_devices = 1; // This is to simplify this example.
    /// // Spawn this only on rank 0
    /// let id = Id::new().unwrap();
    /// // Send id.internal() to other ranks
    /// // let id = Id::uninit(id.internal().clone()); on other ranks
    ///
    /// let rank = 0;
    /// let dev = CudaDevice::new(rank).unwrap();
    /// let comm = Comm::from_rank(dev.clone(), rank, n_devices, id).unwrap();
    /// let slice = dev.htod_copy(vec![(rank + 1) as f32 * 1.0; n]).unwrap();
    /// let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
    /// comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
    ///     .unwrap();

    /// let out = dev.dtoh_sync_copy(&slice_receive).unwrap();

    /// assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
    /// ```
    pub fn from_rank(
        device: Arc<CudaDevice>,
        rank: usize,
        world_size: usize,
        id: Id,
    ) -> Result<Self, result::NcclError> {
        let mut comm = MaybeUninit::uninit();

        let id_low = sys::ncclUniqueId {
            internal: *id.internal(),
        };

        let comm = unsafe {
            result::comm_init_rank(
                comm.as_mut_ptr(),
                world_size
                    .try_into()
                    .expect("World_size cannot be casted to i32"),
                id_low,
                rank.try_into().expect("Rank cannot be cast to i32"),
            )?;
            comm.assume_init()
        };
        Ok(Self {
            comm,
            device,
            rank,
            world_size,
        })
    }
}

// our comm methods
impl Comm {
    pub fn send_u16(
        &mut self,
        send: &CudaSlice<u16>,
        peer_id: usize,
        stream: &CudaStream,
    ) -> Result<result::NcclStatus, result::NcclError> {
        // We have to transmute since u16 is not sendable
        let send_trans: CudaView<u8> = // the transmute_mut is safe because we
        // know that one u16 is 2 u8s, and the buffer is aligned properly for the transmute
         unsafe { send.transmute(send.len() * 2).unwrap() };
        self.send_view(&send_trans, peer_id, stream)
    }

    pub fn receive_u16(
        &mut self,
        receive: &mut CudaSlice<u16>,
        peer_id: usize,
        stream: &CudaStream,
    ) -> Result<result::NcclStatus, result::NcclError> {
        // We have to transmute since u16 is not receivable
        let mut receive_trans: CudaView<u8> = // the transmute_mut is safe
    // because we know that one u16 is 2 u8s, and the buffer is aligned properly for the transmute
    unsafe { receive.transmute(receive.len() * 2).unwrap()
    };
        self.receive_view(&mut receive_trans, peer_id, stream)
    }

    pub fn send_view_u16(
        &mut self,
        send: &CudaView<u16>,
        peer_id: usize,
        stream: &CudaStream,
    ) -> Result<result::NcclStatus, result::NcclError> {
        // We have to transmute since u16 is not sendable
        let send_trans: CudaView<u8> = // the transmute_mut is safe because we
    // know that one u16 is 2 u8s, and the buffer is aligned properly for the transmute
         unsafe { send.transmute(send.len() * 2).unwrap() };
        self.send_view(&send_trans, peer_id, stream)
    }

    pub fn receive_view_u16(
        &mut self,
        receive: &mut CudaView<u16>,
        peer_id: usize,
        stream: &CudaStream,
    ) -> Result<result::NcclStatus, result::NcclError> {
        // We have to transmute since u16 is not receivable
        let mut receive_trans: CudaView<u8> = // the transmute_mut
    // is safe because we know that one u16 is 2 u8s, and the buffer is aligned properly for the transmute
        unsafe { receive.transmute(receive.len() *
    2).unwrap() };
        self.receive_view(&mut receive_trans, peer_id, stream)
    }

    pub fn send_view<T>(
        &mut self,
        send: &CudaView<T>,
        peer_id: usize,
        stream: &CudaStream,
    ) -> Result<result::NcclStatus, result::NcclError>
    where
        T: cudarc::nccl::NcclType,
    {
        unsafe {
            result::send(
                *send.device_ptr() as *mut _,
                send.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comm,
                stream.stream as *mut _,
            )
        }
    }

    pub fn receive_view<T>(
        &mut self,
        receive: &mut CudaView<T>,
        peer_id: usize,
        stream: &CudaStream,
    ) -> Result<result::NcclStatus, result::NcclError>
    where
        T: cudarc::nccl::NcclType,
    {
        unsafe {
            result::recv(
                *receive.device_ptr() as *mut _,
                receive.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comm,
                stream.stream as *mut _,
            )
        }
    }

    pub fn send<T>(
        &mut self,
        send: &CudaSlice<T>,
        peer_id: usize,
        stream: &CudaStream,
    ) -> Result<result::NcclStatus, result::NcclError>
    where
        T: cudarc::nccl::NcclType,
    {
        unsafe {
            result::send(
                *send.device_ptr() as *mut _,
                send.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comm,
                stream.stream as *mut _,
            )
        }
    }

    pub fn receive<T>(
        &mut self,
        receive: &mut CudaSlice<T>,
        peer_id: usize,
        stream: &CudaStream,
    ) -> Result<result::NcclStatus, result::NcclError>
    where
        T: cudarc::nccl::NcclType,
    {
        unsafe {
            result::recv(
                *receive.device_ptr() as *mut _,
                receive.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comm,
                stream.stream as *mut _,
            )
        }
    }
}
