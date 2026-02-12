mod data;
mod errors;
mod traits;

pub(crate) use data::{
    IrisDescriptor, IrisPairDescriptor, Request, RequestBatch, RequestBatchKind, RequestBatchSet,
    RequestBatchSize, RequestPayload, RequestStatus, ResponsePayload, UniquenessRequestDescriptor,
};
pub use errors::ServiceClientError;
pub(crate) use traits::{Initialize, ProcessRequestBatch};
