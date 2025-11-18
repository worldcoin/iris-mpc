pub(crate) mod binary;
// re-export just the constant we want public
// load as: `use iris_mpc_cpu::protocol::USE_PARALLEL_THRESH;`
pub use self::binary::USE_PARALLEL_THRESH;

pub mod ops;
pub(crate) mod prf;
pub mod shared_iris;

// stats for FSS
pub mod perf_stats;

// part of the FSS MSB extraction logic
pub(crate) mod binary_fss;
