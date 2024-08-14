use super::ROTATIONS;
use crate::helpers::device_manager::DeviceManager;
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, CudaView, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};
use std::sync::Arc;

const PTX_SRC: &str = include_str!("kernel.cu");
const RECOMBINE_RESULTS_FUNCTION: &str = "recombineResults";
const MAP_RESULTS_FUNCTION: &str = "mapResults";
const MERGE_RESULTS_FUNCTION: &str = "mergeResults";

pub struct DistanceComparator {
    pub device_manager:          Arc<DeviceManager>,
    pub map_kernels:             Vec<CudaFunction>,
    pub recombine_kernels:       Vec<CudaFunction>,
    pub merge_kernels:           Vec<CudaFunction>,
    pub query_length:            usize,
    pub results_init_host:       Vec<u32>,
    pub final_results_init_host: Vec<u32>,
}

impl DistanceComparator {
    pub fn init(query_length: usize, device_manager: Arc<DeviceManager>) -> Self {
        let ptx = compile_ptx(PTX_SRC).unwrap();
        let mut recombine_kernels = Vec::new();
        let mut map_kernels = Vec::new();
        let mut merge_kernels = Vec::new();

        let devices_count = device_manager.device_count();

        let results_init_host = vec![u32::MAX; query_length];
        let final_results_init_host = vec![u32::MAX; query_length / ROTATIONS];

        for i in 0..devices_count {
            let device = device_manager.device(i);
            device
                .load_ptx(ptx.clone(), "", &[
                    RECOMBINE_RESULTS_FUNCTION,
                    MAP_RESULTS_FUNCTION,
                    MERGE_RESULTS_FUNCTION,
                ])
                .unwrap();

            let map_results_function = device.get_func("", MAP_RESULTS_FUNCTION).unwrap();
            let recombine_results_function =
                device.get_func("", RECOMBINE_RESULTS_FUNCTION).unwrap();
            let merge_results_function = device.get_func("", MERGE_RESULTS_FUNCTION).unwrap();

            recombine_kernels.push(recombine_results_function);
            map_kernels.push(map_results_function);
            merge_kernels.push(merge_results_function);
        }

        Self {
            device_manager,
            map_kernels,
            recombine_kernels,
            merge_kernels,
            query_length,
            results_init_host,
            final_results_init_host,
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn recombine_results(
        &self,
        results1: &[CudaView<u64>],
        results2: &[CudaView<u64>],
        results3: &[CudaView<u64>],
        results: &[CudaSlice<u64>],
        db_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        for i in 0..self.device_manager.device_count() {
            let num_queries = db_sizes[i] * self.query_length / ROTATIONS;
            assert!(
                num_queries % 64 == 0,
                "Number of queries must be divisible by 64"
            );
            let threads_per_block = 256;
            let blocks_per_grid = num_queries.div_ceil(threads_per_block);
            let cfg = LaunchConfig {
                block_dim:        (threads_per_block as u32, 1, 1),
                grid_dim:         (blocks_per_grid as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            self.device_manager.device(i).bind_to_thread().unwrap();

            unsafe {
                self.recombine_kernels[i]
                    .clone()
                    .launch_on_stream(
                        &streams[i],
                        cfg,
                        (
                            &results1[i],
                            &results2[i],
                            &results3[i],
                            &results[i],
                            num_queries,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn map_results(
        &self,
        results_left: &[CudaView<u64>],
        results_right: &[CudaView<u64>],
        results_ptrs: &[CudaSlice<u32>],
        db_sizes: &[usize],
        real_db_sizes: &[usize],
        offset: usize,
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
                self.map_kernels[i]
                    .clone()
                    .launch_on_stream(
                        &streams[i],
                        cfg,
                        (
                            &results_left[i],
                            &results_right[i],
                            &results_ptrs[i],
                            db_sizes[i],
                            self.query_length,
                            offset,
                            num_elements,
                            real_db_sizes[i],
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn merge_results(
        &self,
        match_results_self: &[CudaSlice<u32>],
        match_results: &[CudaSlice<u32>],
        final_results: &[CudaSlice<u32>],
        streams: &[CudaStream],
    ) {
        let num_elements = self.query_length / ROTATIONS;
        let threads_per_block = 256;
        let blocks_per_grid = num_elements.div_ceil(threads_per_block);
        let cfg = LaunchConfig {
            block_dim:        (threads_per_block as u32, 1, 1),
            grid_dim:         (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        for i in 0..self.device_manager.device_count() {
            unsafe {
                self.merge_kernels[i]
                    .clone()
                    .launch_on_stream(
                        &streams[i],
                        cfg,
                        (
                            &match_results_self[i],
                            &match_results[i],
                            &final_results[i],
                            (self.query_length / ROTATIONS) as u64,
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
}
