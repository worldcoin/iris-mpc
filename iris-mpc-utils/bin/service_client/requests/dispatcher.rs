use super::super::CliOptions;

/// Encapsulates logic for dispatching SMPC service requests.
pub struct Dispatcher {
    // Associated dispatch options.
    options: Options,
}

impl Dispatcher {
    pub fn options(&self) -> &Options {
        &self.options
    }

    pub fn new(options: &CliOptions) -> Self {
        Self {
            options: Options::from(options),
        }
    }
}

/// Encapsulates options for dispatching SMPC service requests.
struct Options {}

impl From<&CliOptions> for Options {
    fn from(_options: &CliOptions) -> Self {
        Self {}
    }
}
