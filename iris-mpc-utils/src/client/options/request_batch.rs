/// Options over an associated Iris share pair.
pub struct IrisDescriptor {
    // Ordinal identifer typically pointing to
    index: usize,

    // Optionally apply noise, rotations, mirroring, etc.
    // Type left as TODO
    mutation: Option<()>,
}

/// Options over an individual request within a batch.
pub struct Request {
    // Optional label for cross referencing within batch series.
    nickname: Option<String>,

    // Inner request payload options.
    payload: RequestPayload,
}

/// Options over a batch of requests.
pub struct RequestBatch(Vec<Request>);

/// Options over a series of request batches.
pub struct RequestBatchSeries {
    // Contextual reference to remote system state prior to execution of this batch series.
    // E.G. a hex encoded hash value or a label.
    prestate: Option<()>,

    // Set of batches to be dispatched to target system.
    batches: Vec<Vec<RequestBatch>>,
}

/// Options over an associated request descriptor.
pub enum RequestDescriptor {
    // Nickname within batch/file scope.
    Label(String),

    // Iris serial identifer as assigned by remote system.
    SerialId(usize),
}

/// Options over a request's payload.
pub enum RequestPayload {
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
