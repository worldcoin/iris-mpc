//! NVMe-file-backed implementation of [`ProcessedDatabase`].
//!
//! Unlike [`SlicedProcessedDatabase`](super::share_db::SlicedProcessedDatabase),
//! which keeps the iris/mask code limbs in host RAM (anonymous mmap), this
//! variant stores the code limbs in per-device files on NVMe storage, for
//! databases that exceed host memory capacity. Per-record sums stay
//! GPU-resident, exactly as in the memory variant.
//!
//! Data reaches the GPU through pinned host staging buffers: `prefetch_*`
//! `pread`s the requested code bytes into a page-locked buffer and then issues
//! an async host-to-device copy into the matmul chunk buffers. The blocking
//! `pread` runs on the CPU while the GPU works on the previous chunk; staging is
//! double-buffered per device so a `pread` cannot clobber an in-flight copy.
//!
//! This type is standalone — wiring it into `ServerActor` (backend selection)
//! is a separate piece of work.

use std::{
    fs::{File, OpenOptions},
    io,
    os::unix::fs::FileExt,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

use cudarc::driver::{
    result::{self, memcpy_dtoh_async, stream::synchronize},
    CudaStream, DevicePtr,
};
use memmap2::MmapMut;

use crate::{
    dot::{
        share_db::{DBChunkBuffers, ProcessedDatabase, ShareDB},
        ROTATIONS,
    },
    helpers::query_processor::{CudaVec2DSlicerU32, CudaVec2DSlicerU8, StreamAwareCudaSlice},
};

/// Size in bytes of a single per-record sum.
const SUM_SIZE: usize = std::mem::size_of::<u32>();

/// Default number of pinned staging slots per device. Two matches the actor's
/// double-buffered prefetch pipeline.
const DEFAULT_STAGING_SLOTS: usize = 2;

// ---------------------------------------------------------------------------
// GPU-free helpers (unit-tested without CUDA)
// ---------------------------------------------------------------------------

/// Map a global record index to its `(device_index, slot)` location, where
/// `slot` is the record's position within that device's shard.
fn shard(index: usize, n_devices: usize) -> (usize, usize) {
    (index % n_devices, index / n_devices)
}

/// Compute the two stored i8 limbs (as bytes) for a u16 record. Mirrors the
/// transform in `ShareDB::load_single_record_from_db`.
fn transform_record(record: &[u16]) -> (Vec<u8>, Vec<u8>) {
    let limb_0: Vec<u8> = record
        .iter()
        .map(|&x| (((x as i8) as i32 - 128) as i8) as u8)
        .collect();
    let limb_1: Vec<u8> = record
        .iter()
        .map(|&x| (((x >> 8) as i32 - 128) as i8) as u8)
        .collect();
    (limb_0, limb_1)
}

/// Sum of the stored bytes, interpreted as i8 and widened (sign-extended) to
/// u32. This matches the memory variant's
/// `slice.iter().map(|&x| x as u32).sum::<u32>()` over an `&[i8]`, but uses
/// wrapping arithmetic so it does not panic on overflow in debug builds.
fn limb_sum(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(0u32, |acc, &b| acc.wrapping_add((b as i8) as u32))
}

// ---------------------------------------------------------------------------
// Pinned host staging
// ---------------------------------------------------------------------------

/// A page-locked host buffer used to stage NVMe reads before an async
/// host-to-device copy. Backed by an anonymous mmap registered with
/// `cuMemHostRegister`; the mapping is kept alive for the lifetime of the
/// buffer and accessed only through the stored raw pointer.
struct PinnedBuf {
    _mmap: MmapMut,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the buffer owns its mapping; concurrent access to the same buffer is
// prevented by the per-device round-robin slot selection together with the
// caller's prefetch pipeline depth.
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    fn new(len: usize) -> io::Result<Self> {
        let mut mmap = MmapMut::map_anon(len)?;
        let ptr = mmap.as_mut_ptr();
        // Page-lock for fast async H2D copies. Best-effort, matching
        // DeviceManager::register_host_memory.
        unsafe {
            let _ = cudarc::driver::sys::lib().cuMemHostRegister_v2(
                ptr as *mut _,
                len,
                cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE,
            );
        }
        Ok(Self {
            _mmap: mmap,
            ptr,
            len,
        })
    }

    /// Host pointer to the start of the buffer (used as the source of an async
    /// host-to-device copy).
    fn host_ptr(&self) -> u64 {
        self.ptr as u64
    }

    /// A mutable view of the first `n` bytes.
    ///
    /// # Safety
    /// The caller must ensure no in-flight transfer is reading this buffer
    /// (guaranteed by the round-robin slot selection and the caller's prefetch
    /// pipeline depth).
    #[allow(clippy::mut_from_ref)]
    unsafe fn as_mut(&self, n: usize) -> &mut [u8] {
        debug_assert!(n <= self.len);
        std::slice::from_raw_parts_mut(self.ptr, n)
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::lib().cuMemHostUnregister(self.ptr as *mut _);
        }
    }
}

/// One staging slot holds a pinned buffer per limb so both limbs of a chunk can
/// be read and copied without aliasing.
struct StagingSlot {
    limb_0: PinnedBuf,
    limb_1: PinnedBuf,
}

// ---------------------------------------------------------------------------
// NvmeProcessedDatabase
// ---------------------------------------------------------------------------

/// An NVMe-file-backed [`ProcessedDatabase`]: code limbs live in per-device
/// files, per-record sums live on the GPU.
pub struct NvmeProcessedDatabase {
    /// `(limb_0, limb_1)` code files, one pair per device.
    limb_files: Vec<(File, File)>,
    /// GPU-resident per-record sums (identical layout to the memory variant).
    code_sums_gr: CudaVec2DSlicerU32,
    /// Per-device `(limb_0, limb_1)` sums accumulated during loading, uploaded
    /// to `code_sums_gr` in `preprocess`.
    host_sums: Vec<(Vec<u32>, Vec<u32>)>,
    /// Per-device pinned staging slots: `staging[device][slot]`.
    staging: Vec<Vec<StagingSlot>>,
    /// Per-device round-robin selector over staging slots.
    staging_next: Vec<AtomicUsize>,
    code_length: usize,
    max_records_per_device: usize,
}

impl NvmeProcessedDatabase {
    /// Create a store backed by files under `dir`, sized for up to
    /// `max_db_length` records total (split across the engine's devices).
    /// `max_chunk_size` sizes the pinned staging buffers (the largest chunk
    /// that will be prefetched at once).
    pub fn new(
        engine: &ShareDB,
        max_db_length: usize,
        dir: &Path,
        max_chunk_size: usize,
    ) -> io::Result<Self> {
        Self::with_staging_slots(
            engine,
            max_db_length,
            dir,
            max_chunk_size,
            DEFAULT_STAGING_SLOTS,
        )
    }

    /// As [`new`](Self::new), with an explicit number of pinned staging slots
    /// per device (the prefetch pipeline depth).
    pub fn with_staging_slots(
        engine: &ShareDB,
        max_db_length: usize,
        dir: &Path,
        max_chunk_size: usize,
        staging_slots: usize,
    ) -> io::Result<Self> {
        assert!(staging_slots >= 1, "need at least one staging slot");
        let n_devices = engine.device_manager().device_count();
        let code_length = engine.code_length();
        let max_records_per_device = max_db_length / n_devices;
        let file_len = (max_records_per_device * code_length) as u64;

        std::fs::create_dir_all(dir)?;

        let mut limb_files = Vec::with_capacity(n_devices);
        let mut host_sums = Vec::with_capacity(n_devices);
        let mut staging = Vec::with_capacity(n_devices);
        let mut staging_next = Vec::with_capacity(n_devices);
        let mut sums_0 = Vec::with_capacity(n_devices);
        let mut sums_1 = Vec::with_capacity(n_devices);

        for idx in 0..n_devices {
            limb_files.push((
                open_limb_file(dir, idx, 0, file_len)?,
                open_limb_file(dir, idx, 1, file_len)?,
            ));
            host_sums.push((
                vec![0u32; max_records_per_device],
                vec![0u32; max_records_per_device],
            ));

            let device = engine.device_manager().device(idx);
            device.bind_to_thread().unwrap();
            // SAFETY: freshly allocated device memory, sized to hold all sums.
            let (s0, s1) = unsafe {
                (
                    device.alloc::<u32>(max_records_per_device).unwrap(),
                    device.alloc::<u32>(max_records_per_device).unwrap(),
                )
            };
            sums_0.push(StreamAwareCudaSlice::from(s0));
            sums_1.push(StreamAwareCudaSlice::from(s1));

            let mut slots = Vec::with_capacity(staging_slots);
            for _ in 0..staging_slots {
                slots.push(StagingSlot {
                    limb_0: PinnedBuf::new(max_chunk_size * code_length)?,
                    limb_1: PinnedBuf::new(max_chunk_size * code_length)?,
                });
            }
            staging.push(slots);
            staging_next.push(AtomicUsize::new(0));
        }

        for device in engine.device_manager().devices() {
            device.synchronize().unwrap();
        }

        Ok(Self {
            limb_files,
            code_sums_gr: CudaVec2DSlicerU32 {
                limb_0: sums_0,
                limb_1: sums_1,
            },
            host_sums,
            staging,
            staging_next,
            code_length,
            max_records_per_device,
        })
    }

    /// Pick the next staging slot for a device, round-robin.
    fn next_slot(&self, device_index: usize) -> usize {
        let n = self.staging[device_index].len();
        self.staging_next[device_index].fetch_add(1, Ordering::Relaxed) % n
    }

    /// Write both limbs of a single record to a device's files at `slot`, and
    /// record their sums. Shared by the DB and S3 load paths.
    fn store_record(&mut self, device: usize, slot: usize, limb_0: &[u8], limb_1: &[u8]) {
        assert!(
            slot < self.max_records_per_device,
            "record slot {slot} exceeds per-device capacity {}",
            self.max_records_per_device
        );
        let offset = (slot * self.code_length) as u64;
        self.limb_files[device].0.write_all_at(limb_0, offset).unwrap();
        self.limb_files[device].1.write_all_at(limb_1, offset).unwrap();
        self.host_sums[device].0[slot] = limb_sum(limb_0);
        self.host_sums[device].1[slot] = limb_sum(limb_1);
    }
}

impl ProcessedDatabase for NvmeProcessedDatabase {
    fn load_single_record_from_db(&mut self, engine: &ShareDB, index: usize, record: &[u16]) {
        assert_eq!(record.len(), engine.code_length());
        let (device, slot) = shard(index, engine.device_manager().device_count());
        let (limb_0, limb_1) = transform_record(record);
        self.store_record(device, slot, &limb_0, &limb_1);
    }

    fn load_single_record_from_s3(
        &mut self,
        engine: &ShareDB,
        index: usize,
        a0_host: &[u8],
        a1_host: &[u8],
    ) {
        assert_eq!(a0_host.len(), engine.code_length());
        assert_eq!(a1_host.len(), engine.code_length());
        let (device, slot) = shard(index, engine.device_manager().device_count());
        self.store_record(device, slot, a0_host, a1_host);
    }

    fn preprocess(&mut self, engine: &ShareDB, db_lens: &[usize]) {
        for device_index in 0..engine.device_manager().device_count() {
            engine
                .device_manager()
                .device(device_index)
                .bind_to_thread()
                .unwrap();
            let n = db_lens[device_index];
            // SAFETY: device pointers and host slices are valid and sized to n.
            unsafe {
                result::memcpy_htod_sync(
                    self.code_sums_gr.limb_0[device_index].cu_device_ptr,
                    &self.host_sums[device_index].0[..n],
                )
                .unwrap();
                result::memcpy_htod_sync(
                    self.code_sums_gr.limb_1[device_index].cu_device_ptr,
                    &self.host_sums[device_index].1[..n],
                )
                .unwrap();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_at_index(
        &self,
        engine: &ShareDB,
        query: &CudaVec2DSlicerU8,
        sums: &CudaVec2DSlicerU32,
        src_index: usize,
        dst_index: usize,
        device_index: usize,
        streams: &[CudaStream],
    ) {
        let code_length = engine.code_length();
        engine
            .device_manager()
            .device(device_index)
            .bind_to_thread()
            .unwrap();

        let src_off = (code_length * 15 + src_index * code_length * ROTATIONS) as u64;
        let dst_off = (dst_index * code_length) as u64;
        let stream = streams[device_index].stream;
        let slot = self.next_slot(device_index);
        let stage = &self.staging[device_index][slot];

        for (buf, query_limb, file) in [
            (&stage.limb_0, &query.limb_0, &self.limb_files[device_index].0),
            (&stage.limb_1, &query.limb_1, &self.limb_files[device_index].1),
        ] {
            // SAFETY: this slot is not in flight (round-robin + caller pipeline).
            let host = unsafe { buf.as_mut(code_length) };
            unsafe {
                memcpy_dtoh_async(host, *query_limb[device_index].device_ptr() + src_off, stream)
                    .unwrap();
                synchronize(stream).unwrap();
            }
            file.write_all_at(host, dst_off).unwrap();
        }

        // Sums stay on the GPU: copy the new record's sums device-to-device,
        // exactly as the memory variant does.
        let sum_src_off = SUM_SIZE * 15 + src_index * SUM_SIZE * ROTATIONS;
        unsafe {
            crate::helpers::dtod_at_offset(
                *self.code_sums_gr.limb_0[device_index].device_ptr(),
                dst_index * SUM_SIZE,
                *sums.limb_0[device_index].device_ptr(),
                sum_src_off,
                SUM_SIZE,
                stream,
            );
            crate::helpers::dtod_at_offset(
                *self.code_sums_gr.limb_1[device_index].device_ptr(),
                dst_index * SUM_SIZE,
                *sums.limb_1[device_index].device_ptr(),
                sum_src_off,
                SUM_SIZE,
                stream,
            );
        }
    }

    fn prefetch_chunk(
        &self,
        engine: &ShareDB,
        buffers: &DBChunkBuffers,
        chunk_sizes: &[usize],
        offset: &[usize],
        db_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        let code_length = engine.code_length();
        for idx in 0..engine.device_manager().device_count() {
            engine.device_manager().device(idx).bind_to_thread().unwrap();

            if offset[idx] >= db_sizes[idx]
                || offset[idx] + chunk_sizes[idx] > db_sizes[idx]
                || chunk_sizes[idx] == 0
            {
                continue;
            }

            let nbytes = chunk_sizes[idx] * code_length;
            let file_off = (offset[idx] * code_length) as u64;
            let stream = streams[idx].stream;
            let slot = self.next_slot(idx);
            let stage = &self.staging[idx][slot];

            for (buf, file, dst) in [
                (&stage.limb_0, &self.limb_files[idx].0, &buffers.limb_0[idx]),
                (&stage.limb_1, &self.limb_files[idx].1, &buffers.limb_1[idx]),
            ] {
                let host_ptr = buf.host_ptr();
                // SAFETY: this slot is not in flight (round-robin + caller pipeline).
                let host = unsafe { buf.as_mut(nbytes) };
                file.read_exact_at(host, file_off).unwrap();
                unsafe {
                    cudarc::driver::sys::lib()
                        .cuMemcpyHtoDAsync_v2(
                            *dst.device_ptr(),
                            host_ptr as *mut _,
                            nbytes,
                            stream,
                        )
                        .result()
                        .unwrap();
                }
            }
        }
    }

    fn prefetch_subset(
        &self,
        engine: &ShareDB,
        buffers: &DBChunkBuffers,
        indices: &[Vec<u32>],
        streams: &[CudaStream],
    ) {
        let code_length = engine.code_length();
        for idx in 0..engine.device_manager().device_count() {
            engine.device_manager().device(idx).bind_to_thread().unwrap();

            let wanted = &indices[idx];
            if wanted.is_empty() {
                continue;
            }
            let nbytes = wanted.len() * code_length;
            let stream = streams[idx].stream;
            let slot = self.next_slot(idx);
            let stage = &self.staging[idx][slot];

            // Gather the requested records (scattered on disk) into contiguous
            // staging, then a single async H2D per limb.
            for (buf, file, dst) in [
                (&stage.limb_0, &self.limb_files[idx].0, &buffers.limb_0[idx]),
                (&stage.limb_1, &self.limb_files[idx].1, &buffers.limb_1[idx]),
            ] {
                let host_ptr = buf.host_ptr();
                // SAFETY: this slot is not in flight (round-robin + caller pipeline).
                let host = unsafe { buf.as_mut(nbytes) };
                for (slot_offset, &wanted_idx) in wanted.iter().enumerate() {
                    let file_off = (wanted_idx as usize * code_length) as u64;
                    let dst_byte = slot_offset * code_length;
                    file.read_exact_at(&mut host[dst_byte..dst_byte + code_length], file_off)
                        .unwrap();
                }
                unsafe {
                    cudarc::driver::sys::lib()
                        .cuMemcpyHtoDAsync_v2(
                            *dst.device_ptr(),
                            host_ptr as *mut _,
                            nbytes,
                            stream,
                        )
                        .result()
                        .unwrap();
                }
            }

            // Gather the matching sums GPU-to-GPU (unchanged from the memory
            // variant).
            unsafe {
                for (slot_offset, &wanted_idx) in wanted.iter().enumerate() {
                    cudarc::driver::sys::lib()
                        .cuMemcpyDtoDAsync_v2(
                            *buffers.sums.limb_0[idx].device_ptr() + (slot_offset * SUM_SIZE) as u64,
                            *self.code_sums_gr.limb_0[idx].device_ptr()
                                + (wanted_idx as usize * SUM_SIZE) as u64,
                            SUM_SIZE,
                            stream,
                        )
                        .result()
                        .unwrap();
                    cudarc::driver::sys::lib()
                        .cuMemcpyDtoDAsync_v2(
                            *buffers.sums.limb_1[idx].device_ptr() + (slot_offset * SUM_SIZE) as u64,
                            *self.code_sums_gr.limb_1[idx].device_ptr()
                                + (wanted_idx as usize * SUM_SIZE) as u64,
                            SUM_SIZE,
                            stream,
                        )
                        .result()
                        .unwrap();
                }
            }
        }
    }
}

/// Open (creating if needed) and size a per-device limb file.
fn open_limb_file(dir: &Path, device: usize, limb: usize, len: u64) -> io::Result<File> {
    let path = dir.join(format!("device_{device}_limb_{limb}.bin"));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    file.set_len(len)?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn shard_maps_round_robin() {
        // 3 devices: index -> (device, slot)
        assert_eq!(shard(0, 3), (0, 0));
        assert_eq!(shard(1, 3), (1, 0));
        assert_eq!(shard(2, 3), (2, 0));
        assert_eq!(shard(3, 3), (0, 1));
        assert_eq!(shard(7, 3), (1, 2));
    }

    #[test]
    fn transform_matches_reference() {
        // Reference transform from ShareDB::load_single_record_from_db.
        let record: Vec<u16> = vec![0, 1, 128, 255, 256, 65535, 32768, 511];
        let (l0, l1) = transform_record(&record);

        let ref0: Vec<u8> = record
            .iter()
            .map(|&x| (((x as i8) as i32 - 128) as i8) as u8)
            .collect();
        let ref1: Vec<u8> = record
            .iter()
            .map(|&x| (((x >> 8) as i32 - 128) as i8) as u8)
            .collect();
        assert_eq!(l0, ref0);
        assert_eq!(l1, ref1);
    }

    #[test]
    fn limb_sum_matches_signed_widening_sum() {
        // limb_sum must equal the memory variant's `iter().map(|&x| x as u32).sum()`
        // over the bytes read as i8 (computed here with wrapping to avoid debug
        // overflow panics).
        let bytes: Vec<u8> = (0u16..600).map(|x| x as u8).collect();
        let expected = bytes
            .iter()
            .fold(0u32, |acc, &b| acc.wrapping_add((b as i8) as u32));
        assert_eq!(limb_sum(&bytes), expected);

        // A couple of explicit values: 0xFF as i8 = -1 -> 0xFFFF_FFFF.
        assert_eq!(limb_sum(&[0xFF]), 0xFFFF_FFFFu32);
        assert_eq!(limb_sum(&[0x01, 0x02]), 3);
        assert_eq!(limb_sum(&[0x80]), (-128i32) as u32); // i8::MIN sign-extended
    }

    #[test]
    fn file_round_trip_at_sharded_offsets() {
        // Write two records to a device file at their sharded offsets via the
        // same offset math as the store, then read them back.
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = format!(
            "nvme_db_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();

        let code_length = 4usize;
        let max_records = 8usize;
        let file = open_limb_file(&dir, 0, 0, (max_records * code_length) as u64).unwrap();

        let rec_a = [1u8, 2, 3, 4];
        let rec_b = [9u8, 8, 7, 6];
        // slots 0 and 2
        file.write_all_at(&rec_a, 0).unwrap();
        file.write_all_at(&rec_b, (2 * code_length) as u64).unwrap();

        let mut back_a = [0u8; 4];
        let mut back_b = [0u8; 4];
        file.read_exact_at(&mut back_a, 0).unwrap();
        file.read_exact_at(&mut back_b, (2 * code_length) as u64).unwrap();
        assert_eq!(back_a, rec_a);
        assert_eq!(back_b, rec_b);

        // slot 1 (between them) is untouched / zero from set_len.
        let mut gap = [0xAAu8; 4];
        file.read_exact_at(&mut gap, code_length as u64).unwrap();
        assert_eq!(gap, [0u8; 4]);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
