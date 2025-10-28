use super::super::CliOptions;

/// Encapsulates logic for dispatching SMPC service requests.
#[derive(Debug)]
pub struct Dispatcher {
    // Associated dispatch options.
    options: Options,
}

impl Dispatcher {
    pub fn options(&self) -> &Options {
        &self.options
    }
}

impl From<&CliOptions> for Dispatcher {
    fn from(options: &CliOptions) -> Self {
        Self {
            options: Options::from(options),
        }
    }
}

/// Encapsulates options for dispatching SMPC service requests.
#[derive(Debug, Clone)]
struct Options {}

impl From<&CliOptions> for Options {
    fn from(_options: &CliOptions) -> Self {
        Self {}
    }
}
