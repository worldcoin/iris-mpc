use super::ROTATIONS;
use crate::helpers::{
    device_manager::DeviceManager, launch_config_from_elements_and_threads,
    DEFAULT_LAUNCH_CONFIG_THREADS,
};
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, CudaView, LaunchAsync},
    nvrtc::compile_ptx,
};
use std::{cmp::min, sync::Arc};

const PTX_SRC: &str = include_str!("kernel.cu");
const OPEN_RESULTS_FUNCTION: &str = "openResults";
const MERGE_DB_RESULTS_FUNCTION: &str = "mergeDbResults";
const MERGE_BATCH_RESULTS_FUNCTION: &str = "mergeBatchResults";
const ALL_MATCHES_LEN: usize = 256;

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
    pub match_counters:          Vec<CudaSlice<u32>>,
    pub all_matches:             Vec<CudaSlice<u32>>,
    pub match_counters_left:     Vec<CudaSlice<u32>>,
    pub match_counters_right:    Vec<CudaSlice<u32>>,
    pub partial_results_left:    Vec<CudaSlice<u32>>,
    pub partial_results_right:   Vec<CudaSlice<u32>>,
}

impl DistanceComparator {
    pub fn init(query_length: usize, device_manager: Arc<DeviceManager>) -> Self {
        let ptx = compile_ptx(PTX_SRC).unwrap();
        let mut open_kernels: Vec<CudaFunction> = Vec::new();
        let mut merge_db_kernels = Vec::new();
        let mut merge_batch_kernels = Vec::new();
        let mut opened_results = vec![];
        let mut final_results = vec![];
        let mut match_counters = vec![];
        let mut match_counters_left = vec![];
        let mut match_counters_right = vec![];
        let mut all_matches = vec![];
        let mut partial_results_left = vec![];
        let mut partial_results_right = vec![];

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
            match_counters.push(device.alloc_zeros(query_length / ROTATIONS).unwrap());
            match_counters_left.push(device.alloc_zeros(query_length / ROTATIONS).unwrap());
            match_counters_right.push(device.alloc_zeros(query_length / ROTATIONS).unwrap());
            all_matches.push(
                device
                    .alloc_zeros(ALL_MATCHES_LEN * query_length / ROTATIONS)
                    .unwrap(),
            );
            partial_results_left.push(
                device
                    .alloc_zeros(ALL_MATCHES_LEN * query_length / ROTATIONS)
                    .unwrap(),
            );
            partial_results_right.push(
                device
                    .alloc_zeros(ALL_MATCHES_LEN * query_length / ROTATIONS)
                    .unwrap(),
            );

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
            match_counters,
            match_counters_left,
            match_counters_right,
            all_matches,
            partial_results_left,
            partial_results_right,
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
        ignore_db_results: &[bool],
        streams: &[CudaStream],
    ) {
        for i in 0..self.device_manager.device_count() {
            // Those correspond to 0 length dbs, which were just artificially increased to
            // length 1 to avoid division by zero in the kernel
            if ignore_db_results[i] {
                continue;
            }
            let num_elements = (db_sizes[i] * self.query_length).div_ceil(64);
            let threads_per_block = DEFAULT_LAUNCH_CONFIG_THREADS; // ON CHANGE: sync with kernel
            let cfg = launch_config_from_elements_and_threads(
                num_elements as u32,
                threads_per_block,
                &self.device_manager.devices()[i],
            );
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
            if db_sizes[i] == 0 {
                continue;
            }
            let num_elements = (db_sizes[i] * self.query_length / ROTATIONS).div_ceil(64);
            let threads_per_block = DEFAULT_LAUNCH_CONFIG_THREADS; // ON CHANGE: sync with kernel
            let cfg = launch_config_from_elements_and_threads(
                num_elements as u32,
                threads_per_block,
                &self.device_manager.devices()[i],
            );

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
                            &self.match_counters[i],
                            &self.all_matches[i],
                            &self.match_counters_left[i],
                            &self.match_counters_right[i],
                            &self.partial_results_left[i],
                            &self.partial_results_right[i],
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

    pub fn fetch_match_counters(&self, counters: &[CudaSlice<u32>]) -> Vec<Vec<u32>> {
        let mut results = vec![];
        for i in 0..self.device_manager.device_count() {
            results.push(
                self.device_manager
                    .device(i)
                    .dtoh_sync_copy(&counters[i])
                    .unwrap(),
            );
        }
        results
    }

    pub fn fetch_all_match_ids(
        &self,
        match_counters: Vec<Vec<u32>>,
        matches: &[CudaSlice<u32>],
    ) -> Vec<Vec<u32>> {
        let mut results = vec![];
        for i in 0..self.device_manager.device_count() {
            results.push(
                self.device_manager
                    .device(i)
                    .dtoh_sync_copy(&matches[i])
                    .unwrap(),
            );
        }

        let mut matches_per_query = vec![vec![]; match_counters[0].len()];
        let n_devices = self.device_manager.device_count();
        for i in 0..self.device_manager.device_count() {
            let mut offset = 0;
            for j in 0..match_counters[0].len() {
                let len = match_counters[i][j] as usize;
                let ids = results[i][offset..offset + min(len, ALL_MATCHES_LEN)]
                    .iter()
                    .map(|idx| idx * n_devices as u32 + i as u32)
                    .collect::<Vec<_>>();
                matches_per_query[j].extend_from_slice(&ids);
                offset += ALL_MATCHES_LEN;
            }
        }

        matches_per_query
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
