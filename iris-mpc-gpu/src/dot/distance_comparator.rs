use super::ROTATIONS;
use crate::helpers::device_manager::DeviceManager;
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, CudaView, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};
use std::sync::Arc;

const PTX_SRC: &str = include_str!("kernel.cu");
const OPEN_RESULTS_FUNCTION: &str = "openResults";
const MERGE_DB_RESULTS_FUNCTION: &str = "mergeDbResults";
const MERGE_BATCH_RESULTS_FUNCTION: &str = "mergeBatchResults";

pub struct DistanceComparator {
    pub device_manager:          Arc<DeviceManager>,
    pub open_kernels:            Vec<CudaFunction>,
    pub merge_db_kernels:        Vec<CudaFunction>,
    pub merge_batch_kernels:     Vec<CudaFunction>,
    pub query_length:            usize,
    pub opened_results:          Vec<CudaSlice<u32>>,
    pub final_results:           Vec<CudaSlice<u32>>,
    pub results_init_host:       Vec<u32>,
    pub final_results_init_host: Vec<u32>,
}

impl DistanceComparator {
    pub fn init(query_length: usize, device_manager: Arc<DeviceManager>) -> Self {
        let ptx = compile_ptx(PTX_SRC).unwrap();
        let mut open_kernels = Vec::new();
        let mut merge_db_kernels = Vec::new();
        let mut merge_batch_kernels = Vec::new();
        let mut opened_results = vec![];
        let mut final_results = vec![];

        let devices_count = device_manager.device_count();

        let results_init_host = vec![u32::MAX; query_length];
        let final_results_init_host = vec![u32::MAX; query_length / ROTATIONS];

        for i in 0..devices_count {
            let device = device_manager.device(i);
            device
                .load_ptx(ptx.clone(), "", &[
                    OPEN_RESULTS_FUNCTION,
                    MERGE_DB_RESULTS_FUNCTION,
                    MERGE_BATCH_RESULTS_FUNCTION,
                ])
                .unwrap();

            let open_results_function = device.get_func("", OPEN_RESULTS_FUNCTION).unwrap();
            let merge_db_results_function = device.get_func("", MERGE_DB_RESULTS_FUNCTION).unwrap();
            let merge_batch_results_function =
                device.get_func("", MERGE_BATCH_RESULTS_FUNCTION).unwrap();

            opened_results.push(device.htod_copy(results_init_host.clone()).unwrap());
            final_results.push(device.htod_copy(final_results_init_host.clone()).unwrap());

            open_kernels.push(open_results_function);
            merge_db_kernels.push(merge_db_results_function);
            merge_batch_kernels.push(merge_batch_results_function);
        }

        Self {
            device_manager,
            open_kernels,
            merge_db_kernels,
            merge_batch_kernels,
            query_length,
            opened_results,
            final_results,
            results_init_host,
            final_results_init_host,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_results(
        &self,
        results1: &[CudaView<u64>],
        results2: &[CudaView<u64>],
        results3: &[CudaView<u64>],
        matches_bitmap: &[CudaSlice<u64>],
        db_sizes: &[usize],
        real_db_sizes: &[usize],
        offset: usize,
        total_db_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        for i in 0..self.device_manager.device_count() {
            let num_elements = (db_sizes[i] * self.query_length).div_ceil(64);
            let threads_per_block = 256;
            let blocks_per_grid = num_elements.div_ceil(threads_per_block);
            let cfg = LaunchConfig {
                block_dim:        (threads_per_block as u32, 1, 1),
                grid_dim:         (blocks_per_grid as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            self.device_manager.device(i).bind_to_thread().unwrap();

            unsafe {
                self.open_kernels[i]
                    .clone()
                    .launch_on_stream(
                        &streams[i],
                        cfg,
                        (
                            &results1[i],
                            &results2[i],
                            &results3[i],
                            &matches_bitmap[i],
                            db_sizes[i],
                            self.query_length,
                            offset,
                            num_elements,
                            real_db_sizes[i],
                            total_db_sizes[i],
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn join_db_matches(
        &self,
        matches_bitmap_left: &[CudaSlice<u64>],
        matches_bitmap_right: &[CudaSlice<u64>],
        final_results: &[CudaSlice<u32>],
        db_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        self.join_matches(
            matches_bitmap_left,
            matches_bitmap_right,
            final_results,
            db_sizes,
            streams,
            &self.merge_db_kernels,
        );
    }

    pub fn join_batch_matches(
        &self,
        matches_bitmap_left: &[CudaSlice<u64>],
        matches_bitmap_right: &[CudaSlice<u64>],
        final_results: &[CudaSlice<u32>],
        streams: &[CudaStream],
    ) {
        self.join_matches(
            matches_bitmap_left,
            matches_bitmap_right,
            final_results,
            &vec![self.query_length; self.device_manager.device_count()],
            streams,
            &self.merge_batch_kernels,
        );
    }

    fn join_matches(
        &self,
        matches_bitmap_left: &[CudaSlice<u64>],
        matches_bitmap_right: &[CudaSlice<u64>],
        final_results: &[CudaSlice<u32>],
        db_sizes: &[usize],
        streams: &[CudaStream],
        kernels: &[CudaFunction],
    ) {
        for i in 0..self.device_manager.device_count() {
            let num_elements = (db_sizes[i] * self.query_length / ROTATIONS).div_ceil(64);
            let threads_per_block = 256;
            let blocks_per_grid = num_elements.div_ceil(threads_per_block);
            let cfg = LaunchConfig {
                block_dim:        (threads_per_block as u32, 1, 1),
                grid_dim:         (blocks_per_grid as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                kernels[i]
                    .clone()
                    .launch_on_stream(
                        &streams[i],
                        cfg,
                        (
                            &matches_bitmap_left[i],
                            &matches_bitmap_right[i],
                            &final_results[i],
                            (self.query_length / ROTATIONS) as u64,
                            db_sizes[i] as u64,
                            num_elements as u64,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn fetch_final_results(&self, final_results_ptrs: &[CudaSlice<u32>]) -> Vec<Vec<u32>> {
        let mut results = vec![];
        for i in 0..self.device_manager.device_count() {
            results.push(
                self.device_manager
                    .device(i)
                    .dtoh_sync_copy(&final_results_ptrs[i])
                    .unwrap(),
            );
        }
        results
    }

    pub fn prepare_results(&self) -> Vec<CudaSlice<u32>> {
        (0..self.device_manager.device_count())
            .map(|i| {
                self.device_manager
                    .device(i)
                    .htod_copy(self.results_init_host.clone())
                    .unwrap()
            })
            .collect::<Vec<_>>()
    }

    pub fn prepare_final_results(&self) -> Vec<CudaSlice<u32>> {
        (0..self.device_manager.device_count())
            .map(|i| {
                self.device_manager
                    .device(i)
                    .htod_copy(self.final_results_init_host.clone())
                    .unwrap()
            })
            .collect::<Vec<_>>()
    }

    pub fn prepare_db_match_list(&self, db_size: usize) -> Vec<CudaSlice<u64>> {
        (0..self.device_manager.device_count())
            .map(|i| {
                self.device_manager
                    .device(i)
                    .alloc_zeros(db_size * self.query_length / ROTATIONS / 64)
                    .unwrap()
            })
            .collect::<Vec<_>>()
    }
}
