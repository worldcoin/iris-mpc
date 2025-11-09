use super::super::types::RequestBatch;

/// A component responsible for dequeuing system responses from network egress queues.
#[derive(Debug)]
pub struct ResponseDequeuer {}

impl ResponseDequeuer {
    pub fn new() -> Self {
        Self {}
    }

    /// Dequeues system responses from network egress queues.
    #[allow(dead_code)]
    pub async fn dequeue(&self, batch: &RequestBatch) {
        println!("TODO: dequeue responses: {}", batch);
    }
}

impl Default for ResponseDequeuer {
    fn default() -> Self {
        Self::new()
    }
}
