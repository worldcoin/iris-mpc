use super::super::CliOptions;
use super::types::Batch;

/// Encapsulates logic for generating SMPC service requests.
#[derive(Debug)]
pub struct Generator {
    // Count of generated batches.
    batch_count: usize,

    // Associated generator options.
    options: Options,
}

impl Generator {
    pub fn options(&self) -> &Options {
        &self.options
    }
}

impl From<&CliOptions> for Generator {
    fn from(options: &CliOptions) -> Self {
        Self {
            batch_count: 0,
            options: Options::from(options),
        }
    }
}

/// Encapsulates options for generating SMPC service requests.
#[derive(Debug, Clone)]
struct Options {
    /// Number of request batches to dispatch.
    batch_count: usize,

    /// Maximum size of each batch.
    batch_size_max: usize,
}

impl From<&CliOptions> for Options {
    fn from(options: &CliOptions) -> Self {
        Self {
            batch_count: *options.batch_count(),
            batch_size_max: *options.batch_size_max(),
        }
    }
}
