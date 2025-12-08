pub(crate) mod binary;
// re-export the knobs/metrics we want public
// load as: `use iris_mpc_cpu::protocol::USE_PARALLEL_THRESH;`
pub use self::binary::{msb_fss_total_inputs, USE_PARALLEL_THRESH};

pub mod ops;
pub(crate) mod prf;
pub mod shared_iris;

// stats for FSS
pub mod perf_stats;

// part of the FSS MSB extraction logic
pub(crate) mod binary_fss;
pub use self::binary_fss::{
    dealer_traffic_breakdown, eval_traffic_breakdown, format_traffic_bytes, fss_key_package_bytes,
    traffic_totals_per_party, DealerTrafficSnapshot, EvalTrafficSnapshot,
};
