mod data;
mod errors;
mod traits;

pub(crate) use data::{
    BatchKind, IrisDescriptor, IrisPairDescriptor, Request, RequestBatch, RequestBatchSet,
    RequestPayload, RequestStatus, ResponsePayload, UniquenessRequestDescriptor,
};
pub use errors::ServiceClientError;
pub(crate) use traits::{Initialize, ProcessRequestBatch};
