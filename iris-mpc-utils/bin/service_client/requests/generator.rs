use super::super::CliOptions;

/// Encapsulates logic for generating SMPC service requests.
pub struct Generator {
    // Associated generator options.
    options: Options,
}

impl Generator {
    pub fn options(&self) -> &Options {
        &self.options
    }

    pub fn new(options: &CliOptions) -> Self {
        Self {
            options: Options::from(options),
        }
    }
}

/// Encapsulates options for generating SMPC service requests.
struct Options {}

impl From<&CliOptions> for Options {
    fn from(_options: &CliOptions) -> Self {
        Self {}
    }
}
