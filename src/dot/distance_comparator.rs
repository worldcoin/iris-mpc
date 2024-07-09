use super::{error::CudaError, ROTATIONS};
use cudarc::{
    driver::{
        result::{launch_kernel, memcpy_dtoh_sync},
        CudaDevice, CudaFunction, CudaSlice, CudaView, DeviceRepr, LaunchAsync, LaunchConfig,
    },
    nvrtc::{compile_ptx, CompileError},
};
use std::sync::Arc;

const PTX_SRC: &str = include_str!("kernel.cu");
const OPEN_RESULTS_FUNCTION: &str = "openResults";
const MERGE_RESULTS_FUNCTION: &str = "mergeResults";

pub struct DistanceComparator {
    pub devs: Vec<Arc<CudaDevice>>,
    pub open_kernels: Vec<CudaFunction>,
    pub merge_kernels: Vec<CudaFunction>,
    pub query_length: usize,
    pub n_devices: usize,
    pub opened_results: Vec<CudaSlice<u32>>,
    pub final_results: Vec<CudaSlice<u32>>,
    pub results_init_host: Vec<u32>,
    pub final_results_init_host: Vec<u32>,
}

impl DistanceComparator {
    pub fn init(query_length: usize) -> Result<Self, CudaError> {
        let ptx = compile_ptx(PTX_SRC)?;
        let mut open_kernels = Vec::new();
        let mut merge_kernels = Vec::new();
        let mut devs = Vec::new();
        let n_devices = CudaDevice::count()? as usize;
        let mut opened_results = vec![];
        let mut final_results = vec![];

        let results_init_host = vec![u32::MAX; query_length];
        let final_results_init_host = vec![u32::MAX; query_length / ROTATIONS];

        for i in 0..n_devices {
            let dev = CudaDevice::new(i)?;
            dev.load_ptx(
                ptx.clone(),
                "",
                &[OPEN_RESULTS_FUNCTION, MERGE_RESULTS_FUNCTION],
            )?;

            let open_results_function = dev
                .get_func("", OPEN_RESULTS_FUNCTION)
                .ok_or_else(|| CudaError::CudaFunctionError(OPEN_RESULTS_FUNCTION.to_string()))?;

            let merge_results_function = dev
                .get_func("", MERGE_RESULTS_FUNCTION)
                .ok_or_else(|| CudaError::CudaFunctionError(MERGE_RESULTS_FUNCTION.to_string()))?;

            opened_results.push(dev.htod_copy(results_init_host.clone())?);
            final_results.push(dev.htod_copy(final_results_init_host.clone())?);

            open_kernels.push(open_results_function);
            merge_kernels.push(merge_results_function);
            devs.push(dev);
        }

        Ok(Self {
            devs,
            open_kernels,
            merge_kernels,
            query_length,
            n_devices,
            opened_results,
            final_results,
            results_init_host,
            final_results_init_host,
        })
    }

    pub fn open_results(
        &self,
        results1: &[CudaView<u64>],
        results2: &[CudaView<u64>],
        results3: &[CudaView<u64>],
        results_ptrs: &[CudaSlice<u32>],
        db_sizes: &[usize],
    ) -> Result<(), CudaError> {
        for i in 0..self.n_devices {
            let num_elements = (db_sizes[i] * self.query_length).div_ceil(64);
            let threads_per_block = 256;
            let blocks_per_grid = num_elements.div_ceil(threads_per_block);
            let cfg = LaunchConfig {
                block_dim: (threads_per_block as u32, 1, 1),
                grid_dim: (blocks_per_grid as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            self.devs[i].bind_to_thread()?;

            unsafe {
                self.open_kernels[i].clone().launch(
                    cfg,
                    (
                        &results1[i],
                        &results2[i],
                        &results3[i],
                        &results_ptrs[i],
                        db_sizes[i],
                        self.query_length,
                    ),
                )?;
            }
        }

        Ok(())
    }

    pub fn merge_results(
        &self,
        match_results_self: &[u64],
        match_results: &[u64],
        final_results: &[u64],
        streams: &[u64],
    ) -> Result<Vec<Vec<u32>>, CudaError> {
        let num_elements = self.query_length / ROTATIONS;
        let threads_per_block = 256;
        let blocks_per_grid = num_elements.div_ceil(threads_per_block);
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        for i in 0..self.n_devices {
            self.devs[i].bind_to_thread()?;

            let params = [
                match_results_self[i],
                match_results[i],
                final_results[i],
                (self.query_length / ROTATIONS) as u64,
            ];

            unsafe {
                let mut params = params
                    .iter()
                    .map(|x| x.as_kernel_param())
                    .collect::<Vec<_>>();
                launch_kernel(
                    self.merge_kernels[i].cu_function,
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    streams[i] as *mut _,
                    &mut params,
                )?;
            }
        }

        self.fetch_final_results(final_results)
    }

    pub fn fetch_final_results(
        &self,
        final_results_ptrs: &[u64],
    ) -> Result<Vec<Vec<u32>>, CudaError> {
        let mut results = vec![];
        for i in 0..self.n_devices {
            self.devs[i].bind_to_thread()?;
            let mut tmp = vec![0u32; self.query_length / ROTATIONS];
            unsafe {
                memcpy_dtoh_sync(&mut tmp, final_results_ptrs[i])?;
            }
            results.push(tmp);
        }
        Ok(results)
    }

    pub fn prepare_results(&self) -> Result<Vec<CudaSlice<u32>>, CudaError> {
        let results = (0..self.n_devices)
            .map(|i| self.devs[i].htod_copy(self.results_init_host.clone()))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    pub fn prepare_final_results(&self) -> Result<Vec<CudaSlice<u32>>, CudaError> {
        let results = (0..self.n_devices)
            .map(|i| self.devs[i].htod_copy(self.final_results_init_host.clone()))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }
}
