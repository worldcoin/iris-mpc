use super::types::Batch;

/// Encapsulates logic for dispatching SMPC service requests.
#[derive(Debug)]
pub struct Dispatcher {
    // Associated options.
    options: DispatcherOptions,
}

impl Dispatcher {
    pub fn new(options: DispatcherOptions) -> Self {
        Self { options }
    }

    pub async fn dispatch_batch(&self, batch: Batch) {
        println!("TODO: dispatch request batch: {:?}", batch);
    }
}

/// Options for dispatching SMPC service requests.
#[derive(Debug, Clone)]
pub struct DispatcherOptions {}

impl DispatcherOptions {
    pub fn new() -> Self {
        Self {}
    }
}
