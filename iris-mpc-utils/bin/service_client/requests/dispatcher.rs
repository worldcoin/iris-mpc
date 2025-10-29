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

    pub fn new(options: Options) -> Self {
        Self { options }
    }
}

/// Encapsulates options for dispatching SMPC service requests.
#[derive(Debug, Clone)]
pub struct Options {}

impl Options {
    pub fn new() -> Self {
        Self {}
    }
}
