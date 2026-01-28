use serde::{Deserialize, Serialize};

/// Options over an associated Iris share pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrisDescriptor {
    // Ordinal identifer typically pointing to a row within an NDJSON file.
    index: usize,

    // TODO: Optionally apply noise, rotations, mirroring, etc.
    mutation: Option<()>,
}

/// Options over an individual request within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOptions {
    // Optional label for cross referencing within batch series.
    label: Option<String>,

    // Inner request payload options.
    payload: RequestPayloadOptions,
}

// /// Options over a batch of requests.
// pub struct RequestBatchOptions(Vec<RequestOptions>);

// /// Options over a series of request batches.
// pub struct RequestBatchSeriesOptions {
//     // Contextual reference to remote system state prior to execution of this batch series.
//     // E.G. a hex encoded hash value or a label.
//     prestate: Option<()>,

//     // Set of batches to be dispatched to target system.
//     batches: Vec<RequestBatchOptions>,
// }

/// Options over an associated request descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestDescriptor {
    // Label to identify request within batch/file scope.
    Label(String),

    // Iris serial identifer as assigned by remote system.
    SerialId(usize),
}

/// Options over a request's payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestPayloadOptions {
    // Options over a uniqueness request payload.
    Uniqueness {
        iris_pair: (IrisDescriptor, IrisDescriptor),
        insertion_layers: Option<(usize, usize)>,
    },
    // Options over a reset check request payload.
    ResetCheck {
        iris_pair: (IrisDescriptor, IrisDescriptor),
    },
    // Options over a reset update request payload.
    ResetUpdate {
        iris_pair: (IrisDescriptor, IrisDescriptor),
        parent: RequestDescriptor,
    },
    // Options over a reauthorisation request payload.
    Reauthorisation {
        iris_pair: (IrisDescriptor, IrisDescriptor),
        parent: RequestDescriptor,
    },
    // Options over a deletion request payload.
    Deletion {
        parent: RequestDescriptor,
    },
}
