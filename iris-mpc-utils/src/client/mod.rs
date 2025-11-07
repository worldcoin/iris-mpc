mod request_dispatcher;
mod request_generator;
mod response_correlator;
mod types;

pub use request_dispatcher::AwsRequestDispatcher;
pub use request_generator::RequestGenerator;
pub use types::{RequestBatchKind, RequestBatchSize, RequestDispatcher, RequestIterator};

#[derive(Debug)]
pub struct Client<RD, RI>
where
    RD: RequestDispatcher,
    RI: RequestIterator,
{
    request_dispatcher: RD,
    request_iterator: RI,
}

impl<RD, RI> Client<RD, RI>
where
    RD: RequestDispatcher,
    RI: RequestIterator,
{
    pub fn new(request_dispatcher: RD, request_iterator: RI) -> Self {
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
