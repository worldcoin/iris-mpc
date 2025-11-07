mod correlator;
mod dispatcher;
mod generator;
mod types;

pub use dispatcher::RequestDispatcher;
pub use generator::RequestGenerator;
pub use types::{RequestBatchKind, RequestBatchSize};

#[derive(Debug)]
pub struct Client {
    request_dispatcher: RequestDispatcher,
    request_iterator: RequestGenerator,
}

impl Client {
    pub fn new(request_dispatcher: RequestDispatcher, request_iterator: RequestGenerator) -> Self {
        Self {
            request_dispatcher,
            request_iterator,
        }
    }

    pub async fn run(&mut self) {
        while let Some(batch) = self.request_iterator.next().await {
            self.request_dispatcher.dispatch(batch).await;
        }
    }
}
