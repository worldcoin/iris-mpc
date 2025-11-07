use super::super::types::RequestBatch;

/// A component responsible for correlating system requests with system responses.
#[derive(Debug)]
pub struct ResponseCorrelator {}

impl ResponseCorrelator {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn correlate(&self, batch: &RequestBatch) {
        println!(
            "TODO: correlate enqueued requests with dequeued responses: {}",
            batch
        );
    }
}

impl Default for ResponseCorrelator {
    fn default() -> Self {
        Self::new()
    }
}
