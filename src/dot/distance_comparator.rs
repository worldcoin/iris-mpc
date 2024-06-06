use std::sync::Arc;

use cudarc::{
    driver::{
        result::{
            launch_kernel, memcpy_dtoh_async,
            stream::synchronize,
        },
        CudaDevice, CudaFunction, CudaSlice, CudaStream, DeviceRepr,
        LaunchAsync, LaunchConfig,
    },
    nvrtc::compile_ptx,
};

use super::ROTATIONS;

const PTX_SRC: &str = include_str!("kernel.cu");
const MATCH_RATIO: f64 = 0.375;
const DIST_FUNCTION_NAME: &str = "reconstructAndCompare";
const DEDUP_FUNCTION_NAME: &str = "dedupResults";

pub struct DistanceComparator {
    pub devs: Vec<Arc<CudaDevice>>,
    pub dist_kernels: Vec<CudaFunction>,
    pub dedup_kernels: Vec<CudaFunction>,
    pub query_length: usize,
    pub n_devices: usize,
}

impl DistanceComparator {
    pub fn init(query_length: usize) -> Self {
        let ptx = compile_ptx(PTX_SRC).unwrap();
        let mut dist_kernels = Vec::new();
        let mut dedup_kernels = Vec::new();
        let mut devs = Vec::new();
        let n_devices = CudaDevice::count().unwrap() as usize;

        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            dev.load_ptx(ptx.clone(), DIST_FUNCTION_NAME, &[DIST_FUNCTION_NAME])
                .unwrap();
            let dist_function = dev
                .get_func(DIST_FUNCTION_NAME, DIST_FUNCTION_NAME)
                .unwrap();

            dev.load_ptx(
                ptx.clone(),
                DEDUP_FUNCTION_NAME,
                &[DEDUP_FUNCTION_NAME],
            )
            .unwrap();
            let dedup_function = dev
                .get_func(DEDUP_FUNCTION_NAME, DEDUP_FUNCTION_NAME)
                .unwrap();

            dist_kernels.push(dist_function);
            dedup_kernels.push(dedup_function);
            devs.push(dev);
        }

        Self {
            devs,
            dist_kernels,
            dedup_kernels,
            query_length,
            n_devices,
        }
    }

    pub fn prepare_results(&self) -> Vec<CudaSlice<u32>> {
        let results_uninit = vec![u32::MAX; self.query_length];
        (0..self.n_devices)
            .map(|i| self.devs[i].htod_copy(results_uninit.clone()).unwrap())
            .collect::<Vec<_>>()
    }

    pub fn prepare_final_results(&self) -> Vec<CudaSlice<u32>> {
        let results_uninit = vec![u32::MAX; self.query_length / ROTATIONS];
        (0..self.n_devices)
            .map(|i| self.devs[i].htod_copy(results_uninit.clone()).unwrap())
            .collect::<Vec<_>>()
    }

    pub fn dedup_results(
        &self,
        match_results_self: &Vec<u64>,
        match_results: &Vec<u64>,
        final_results: &Vec<u64>,
        db_size_ptrs: &Vec<u64>,
        streams: &Vec<CudaStream>,
    ) {
        let num_elements = self.query_length / ROTATIONS;
        let threads_per_block = 256;
        let blocks_per_grid = num_elements.div_ceil(threads_per_block);
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        for i in 0..self.n_devices {
            self.devs[i].bind_to_thread().unwrap();

            let params = vec![
                match_results_self[i],
                match_results[i],
                final_results[i],
                db_size_ptrs[i],
                (self.query_length / ROTATIONS) as u64,
            ];

            unsafe {
                let mut params = params
                    .iter()
                    .map(|x| x.as_kernel_param())
                    .collect::<Vec<_>>();
                launch_kernel(
                    self.dedup_kernels[i].cu_function,
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    streams[i].stream,
                    &mut params,
                )
                .unwrap();
            }
        }
    }

    pub fn fetch_results(&self, dev_results: &Vec<u64>, streams: &Vec<u64>) -> Vec<Vec<u32>> {
        let mut results = vec![];
        for i in 0..self.n_devices {
            self.devs[i].bind_to_thread().unwrap();
            let mut tmp = vec![0u32; self.query_length];
            unsafe {
                memcpy_dtoh_async(&mut tmp, dev_results[i], streams[i] as *mut _).unwrap();
                synchronize(streams[i] as *mut _).unwrap();
            }
            results.push(tmp);
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::{sys::lib, DevicePtr, DeviceSlice};

    use crate::{dot::{device_manager::DeviceManager, distance_comparator::DistanceComparator}, helpers::device_ptrs};

    #[test]
    fn test_dedup_query() {
        const ROTATIONS: usize = 31;
        const QUERIES: usize = 248;

        // set all to matches
        let mut match_results = [u32::MAX; QUERIES].to_vec();

        let mut match_results_self = [u32::MAX; QUERIES].to_vec();
        // set 1 to match
        // match_results_self[15] = 0;

        let final_results = [u32::MAX; QUERIES / ROTATIONS].to_vec();

        let device_manager = DeviceManager::init();
        let distance_comparator = DistanceComparator::init(QUERIES);

        let streams = device_manager.fork_streams();

        let mut result_ptrs = vec![];
        let mut result_ptrs_self = vec![];
        let mut final_results_ptrs = vec![];
        let mut db_sizes = vec![];

        for i in 0..device_manager.device_count() {
            // set ith to match
            if i > 0 {
                match_results[(i-1) * 32] = u32::MAX;
            }
            match_results[i * 32] = 0;
            
            result_ptrs.push(
                device_manager
                    .device(i)
                    .htod_copy(match_results.clone())
                    .unwrap(),
            );

            result_ptrs_self.push(
                device_manager
                    .device(i)
                    .htod_copy(match_results_self.clone())
                    .unwrap(),
            );

            final_results_ptrs.push(
                device_manager
                    .device(i)
                    .htod_copy(final_results.clone())
                    .unwrap(),
            );

            db_sizes.push(device_manager.device(i).htod_copy(vec![0u32; 1]).unwrap());
        }

        device_manager.await_streams(&streams);

        distance_comparator.dedup_results(
            &device_ptrs(&result_ptrs_self),
            &device_ptrs(&result_ptrs),
            &device_ptrs(&final_results_ptrs),
            &device_ptrs(&db_sizes),
            &streams,
        );

        device_manager.await_streams(&streams);

        for i in 0..8 {
            let mut result = [3u32; QUERIES / ROTATIONS].to_vec();
            unsafe {
                device_manager.device(i).bind_to_thread().unwrap();
                lib().cuMemcpyDtoH_v2(
                    result.as_mut_ptr() as *mut _,
                    *final_results_ptrs[i].device_ptr(),
                    final_results_ptrs[i].len() * 4,
                );
            }
            
            for j in 0..result.len() {
                if j == i {
                    assert_eq!(result[j], 0);
                } else {
                    assert_eq!(result[j], u32::MAX);
                }
            }

        }
    }
}