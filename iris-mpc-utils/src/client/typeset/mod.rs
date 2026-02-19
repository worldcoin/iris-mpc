mod data;
mod errors;

pub use data::{
    BatchKind, IrisDescriptor, IrisPairDescriptor, Request, RequestBatch, RequestInfo,
    RequestPayload, RequestStatus, ResponsePayload,
};
pub use errors::ServiceClientError;
