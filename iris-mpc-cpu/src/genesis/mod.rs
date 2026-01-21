mod batch_generator;
mod hawk_handle;
mod hawk_job;
pub mod plaintext;
pub mod state_accessor;
pub mod state_sync;
pub mod utils;

use eyre::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

pub use batch_generator::{Batch, BatchGenerator, BatchIterator, BatchSize};
pub use hawk_handle::Handle;
pub use hawk_job::{Job, JobRequest, JobResult};
pub use state_accessor::{
    get_iris_deletions, get_last_indexed_iris_id, get_last_indexed_modification_id,
    set_last_indexed_iris_id, set_last_indexed_modification_id,
};
pub use utils::logger;
pub use utils::{
    errors::IndexationError,
    logger::{log_error, log_info, log_warn},
};

/// Configuration for batch sizing during genesis indexation.
///
/// Supports two modes:
/// - `Static`: Fixed batch size for all batches
/// - `Dynamic`: Batch size grows with graph size, capped at a maximum
///
/// # CLI Format
/// - Static: `"static:<size>"` (e.g., `"static:100"`)
/// - Dynamic: `"dynamic:cap=<cap>,error_rate=<rate>"` (e.g., `"dynamic:cap=500,error_rate=128"`)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchSizeConfig {
    /// Fixed batch size.
    Static { size: usize },
    /// Dynamic batch size with formula: `min(N/(M*r-1)+1, cap)`
    /// where N = graph size, M = HNSW M parameter, r = error_rate.
    Dynamic { cap: usize, error_rate: usize },
}

impl BatchSizeConfig {
    /// Parse from CLI string format.
    ///
    /// # Formats
    /// - `"static:<size>"` - e.g., `"static:100"`
    /// - `"dynamic:cap=<cap>,error_rate=<rate>"` - e.g., `"dynamic:cap=500,error_rate=128"`
    pub fn parse(s: &str) -> Result<Self> {
        if let Some(size_str) = s.strip_prefix("static:") {
            let size: usize = size_str.parse().map_err(|_| {
                eyre::eyre!(
                    "Invalid static batch size '{}'. Expected format: static:<size> (e.g., static:100)",
                    size_str
                )
            })?;
            if size == 0 {
                bail!("Static batch size must be greater than 0");
            }
            return Ok(BatchSizeConfig::Static { size });
        }

        if let Some(params_str) = s.strip_prefix("dynamic:") {
            let mut cap: Option<usize> = None;
            let mut error_rate: Option<usize> = None;

            for part in params_str.split(',') {
                let part = part.trim();
                if let Some(val) = part.strip_prefix("cap=") {
                    cap = Some(val.parse().map_err(|_| {
                        eyre::eyre!("Invalid cap value '{}'. Expected a positive integer.", val)
                    })?);
                } else if let Some(val) = part.strip_prefix("error_rate=") {
                    error_rate = Some(val.parse().map_err(|_| {
                        eyre::eyre!(
                            "Invalid error_rate value '{}'. Expected a positive integer.",
                            val
                        )
                    })?);
                } else {
                    bail!(
                        "Unknown parameter '{}'. Expected 'cap=<value>' or 'error_rate=<value>'.",
                        part
                    );
                }
            }

            let cap = cap.ok_or_else(|| {
                eyre::eyre!("Missing 'cap' parameter. Expected format: dynamic:cap=<cap>,error_rate=<rate>")
            })?;
            let error_rate = error_rate.ok_or_else(|| {
                eyre::eyre!("Missing 'error_rate' parameter. Expected format: dynamic:cap=<cap>,error_rate=<rate>")
            })?;

            if cap == 0 {
                bail!("Dynamic batch cap must be greater than 0");
            }
            if error_rate == 0 {
                bail!("Dynamic error_rate must be greater than 0");
            }

            return Ok(BatchSizeConfig::Dynamic { cap, error_rate });
        }

        bail!(
            "Invalid batch size format '{}'. Expected:\n  \
             - static:<size> (e.g., static:100)\n  \
             - dynamic:cap=<cap>,error_rate=<rate> (e.g., dynamic:cap=500,error_rate=128)",
            s
        )
    }
}

impl fmt::Display for BatchSizeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchSizeConfig::Static { size } => write!(f, "static:{}", size),
            BatchSizeConfig::Dynamic { cap, error_rate } => {
                write!(f, "dynamic:cap={},error_rate={}", cap, error_rate)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_static() {
        let config = BatchSizeConfig::parse("static:100").unwrap();
        assert_eq!(config, BatchSizeConfig::Static { size: 100 });
    }

    #[test]
    fn test_parse_static_zero_fails() {
        let result = BatchSizeConfig::parse("static:0");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_dynamic() {
        let config = BatchSizeConfig::parse("dynamic:cap=500,error_rate=128").unwrap();
        assert_eq!(
            config,
            BatchSizeConfig::Dynamic {
                cap: 500,
                error_rate: 128
            }
        );
    }

    #[test]
    fn test_parse_dynamic_reversed_order() {
        let config = BatchSizeConfig::parse("dynamic:error_rate=128,cap=500").unwrap();
        assert_eq!(
            config,
            BatchSizeConfig::Dynamic {
                cap: 500,
                error_rate: 128
            }
        );
    }

    #[test]
    fn test_parse_dynamic_missing_cap() {
        let result = BatchSizeConfig::parse("dynamic:error_rate=128");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cap"));
    }

    #[test]
    fn test_parse_dynamic_missing_error_rate() {
        let result = BatchSizeConfig::parse("dynamic:cap=500");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("error_rate"));
    }

    #[test]
    fn test_parse_invalid_format() {
        let result = BatchSizeConfig::parse("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_display_static() {
        let config = BatchSizeConfig::Static { size: 100 };
        assert_eq!(config.to_string(), "static:100");
    }

    #[test]
    fn test_display_dynamic() {
        let config = BatchSizeConfig::Dynamic {
            cap: 500,
            error_rate: 128,
        };
        assert_eq!(config.to_string(), "dynamic:cap=500,error_rate=128");
    }

    #[test]
    fn test_roundtrip() {
        for input in ["static:42", "dynamic:cap=1000,error_rate=64"] {
            let config = BatchSizeConfig::parse(input).unwrap();
            let output = config.to_string();
            let reparsed = BatchSizeConfig::parse(&output).unwrap();
            assert_eq!(config, reparsed);
        }
    }
}
