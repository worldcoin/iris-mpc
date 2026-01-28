use serde::{Deserialize, Serialize};

use iris_mpc_common::IrisSerialId;

use super::descriptors::{IrisPairDescriptor, RequestDescriptor};

/// Set of variants over inputs to request batch creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestBatchOptions {
    // Options for instanitating a set of interleaved request batches.
    Series {
        // Contextual reference to remote system state prior to execution of this batch series.
        // E.G. a hex encoded hash value or a label.
        prestate: Option<String>,

        // Batches of request batch options.
        batches: Vec<Vec<RequestOptions>>,
    },
    // Options for generating a set of simple request batches.
    Simple {
        /// Number of request batches to generate.
        batch_count: usize,

        /// Determines type of requests to be included in each batch.
        batch_kind: String,

        /// Size of each batch.
        batch_size: usize,

        // A known serial identifier that allows response correlation to be bypassed.
        known_iris_serial_id: Option<IrisSerialId>,
    },
}

/// Options over an individual request within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOptions {
    // Optional label for cross referencing within batch series.
    label: Option<String>,

    // Inner request payload options.
    payload: RequestPayloadOptions,
}

impl RequestOptions {
    pub fn new(label: Option<&str>, payload: RequestPayloadOptions) -> Self {
        Self {
            label: match label {
                Option::Some(inner) => Some(inner.to_string()),
                Option::None => None,
            },
            payload,
        }
    }
}

/// Options over a request's payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestPayloadOptions {
    // Options over a deletion request payload.
    IdentityDeletion {
        parent: RequestDescriptor,
    },
    // Options over a reauthorisation request payload.
    Reauthorisation {
        iris_pair: IrisPairDescriptor,
        parent: RequestDescriptor,
    },
    // Options over a reset check request payload.
    ResetCheck {
        iris_pair: IrisPairDescriptor,
    },
    // Options over a reset update request payload.
    ResetUpdate {
        iris_pair: IrisPairDescriptor,
        parent: RequestDescriptor,
    },
    // Options over a uniqueness request payload.
    Uniqueness {
        iris_pair: IrisPairDescriptor,
        insertion_layers: Option<(usize, usize)>,
    },
}

#[cfg(test)]
mod tests {
    use super::super::descriptors::{
        tests::{
            REQUEST_DESCRIPTOR_0, REQUEST_DESCRIPTOR_1, REQUEST_DESCRIPTOR_2, REQUEST_DESCRIPTOR_3,
            REQUEST_DESCRIPTOR_4_0, REQUEST_DESCRIPTOR_4_1, REQUEST_DESCRIPTOR_4_2,
        },
        IrisPairDescriptor, RequestDescriptor,
    };
    use super::{RequestOptions, RequestPayloadOptions};

    impl RequestOptions {
        /// Identity deletion.
        fn new_0() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_0),
                RequestPayloadOptions::IdentityDeletion {
                    parent: RequestDescriptor::new_4_0(),
                },
            )
        }

        /// Reauthorisation.
        fn new_1() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_1),
                RequestPayloadOptions::Reauthorisation {
                    iris_pair: IrisPairDescriptor::new_0(11),
                    parent: RequestDescriptor::new_4_1(),
                },
            )
        }

        /// ResetCheck.
        fn new_2() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_2),
                RequestPayloadOptions::ResetCheck {
                    iris_pair: IrisPairDescriptor::new_0(13),
                },
            )
        }

        /// ResetUpdate.
        fn new_3() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_3),
                RequestPayloadOptions::ResetUpdate {
                    iris_pair: IrisPairDescriptor::new_0(15),
                    parent: RequestDescriptor::new_4_2(),
                },
            )
        }

        /// Uniqueness 1.
        fn new_4_0() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_4_0),
                RequestPayloadOptions::Uniqueness {
                    iris_pair: IrisPairDescriptor::new_0(1),
                    insertion_layers: None,
                },
            )
        }

        /// Uniqueness 2.
        fn new_4_1() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_4_1),
                RequestPayloadOptions::Uniqueness {
                    iris_pair: IrisPairDescriptor::new_0(3),
                    insertion_layers: None,
                },
            )
        }

        /// Uniqueness 3.
        fn new_4_2() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_4_2),
                RequestPayloadOptions::Uniqueness {
                    iris_pair: IrisPairDescriptor::new_0(5),
                    insertion_layers: None,
                },
            )
        }

        fn new_batch_0() -> Vec<Self> {
            // Uniqueness only.
            vec![Self::new_4_0(), Self::new_4_1(), Self::new_4_2()]
        }

        fn new_batch_1() -> Vec<Self> {
            vec![
                Self::new_0(),
                Self::new_1(),
                Self::new_2(),
                Self::new_3(),
                Self::new_4_0(),
                Self::new_4_1(),
                Self::new_4_2(),
            ]
        }
    }

    #[test]
    fn test_new_request_opts() {
        for opts_factory in [
            RequestOptions::new_0,
            RequestOptions::new_1,
            RequestOptions::new_2,
            RequestOptions::new_3,
            RequestOptions::new_4_0,
            RequestOptions::new_4_1,
            RequestOptions::new_4_2,
        ] {
            let _ = opts_factory();
        }
    }

    #[test]
    fn test_new_request_opts_batch() {
        for opts_batch_factory in [RequestOptions::new_batch_0, RequestOptions::new_batch_1] {
            let _ = opts_batch_factory();
        }
    }
}
