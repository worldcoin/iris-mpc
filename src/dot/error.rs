use cudarc::{driver::DriverError, nvrtc::CompileError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    DriverError(#[from] DriverError),
    #[error(transparent)]
    CompileError(#[from] CompileError),
    #[error("Failed to get CudaFunction")]
    CudaFunctionError(String),
}
