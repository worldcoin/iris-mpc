use serde::{Deserialize, Serialize};

use iris_mpc_common::IrisSerialId;

use super::descriptors::{IrisPairDescriptor, UniquenessRequestDescriptor};

/// Set of variants over inputs to request batch creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestBatchOptions {
    // Options for instanitating a set of interleaved request batches.
    Series {
        // Batches of batches of request options.
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

impl RequestBatchOptions {
    pub fn new_series(batches: Vec<Vec<RequestOptions>>) -> Self {
        Self::Series { batches }
    }

    pub fn new_simple(
        batch_kind: &str,
        batch_count: usize,
        batch_size: usize,
        known_iris_serial_id: Option<IrisSerialId>,
    ) -> Self {
        Self::Simple {
            batch_count,
            batch_kind: batch_kind.to_string(),
            batch_size,
            known_iris_serial_id,
        }
    }

    pub fn iris_indexes(&self) -> Vec<usize> {
        match self {
            Self::Series { batches } => {
                unimplemented!()
            }
            Self::Simple { .. } => {
                vec![]
            }
        }
    }
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

    pub fn label(&self) -> &Option<String> {
        &self.label
    }

    pub fn payload(&self) -> &RequestPayloadOptions {
        &self.payload
    }

    pub fn iris_indexes(&self) -> Option<(usize, usize)> {
        self.payload().iris_indexes()
    }
}

/// Options over a request's payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestPayloadOptions {
    // Options over a deletion request payload.
    IdentityDeletion {
        parent: UniquenessRequestDescriptor,
    },
    // Options over a reauthorisation request payload.
    Reauthorisation {
        iris_pair: IrisPairDescriptor,
        parent: UniquenessRequestDescriptor,
    },
    // Options over a reset check request payload.
    ResetCheck {
        iris_pair: IrisPairDescriptor,
    },
    // Options over a reset update request payload.
    ResetUpdate {
        iris_pair: IrisPairDescriptor,
        parent: UniquenessRequestDescriptor,
    },
    // Options over a uniqueness request payload.
    Uniqueness {
        iris_pair: IrisPairDescriptor,
        insertion_layers: Option<(usize, usize)>,
    },
}

impl RequestPayloadOptions {
    pub fn iris_indexes(&self) -> Option<(usize, usize)> {
        match &self {
            Self::IdentityDeletion { .. } | Self::ResetCheck { .. } => None,
            Self::Reauthorisation { iris_pair, .. }
            | Self::ResetUpdate { iris_pair, .. }
            | Self::Uniqueness { iris_pair, .. } => Some(iris_pair.indexes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json;
    use toml;

    use iris_mpc_common::helpers::smpc_request;

    use super::super::descriptors::{
        tests::{
            REQUEST_DESCRIPTOR_0, REQUEST_DESCRIPTOR_1, REQUEST_DESCRIPTOR_2, REQUEST_DESCRIPTOR_3,
            REQUEST_DESCRIPTOR_4_00, REQUEST_DESCRIPTOR_4_01, REQUEST_DESCRIPTOR_4_02,
            REQUEST_DESCRIPTOR_4_10, REQUEST_DESCRIPTOR_4_11, REQUEST_DESCRIPTOR_4_12,
        },
        IrisPairDescriptor, UniquenessRequestDescriptor,
    };
    use super::{RequestBatchOptions, RequestOptions, RequestPayloadOptions};

    impl RequestBatchOptions {
        fn new_simple_1() -> Self {
            Self::new_simple(smpc_request::UNIQUENESS_MESSAGE_TYPE, 10, 10, None)
        }

        fn new_series_1() -> Self {
            Self::new_series(vec![
                RequestOptions::new_batch_0(),
                RequestOptions::new_batch_1(),
            ])
        }
    }

    impl RequestOptions {
        /// Identity deletion.
        fn new_0() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_0),
                RequestPayloadOptions::IdentityDeletion {
                    parent: UniquenessRequestDescriptor::new_4_10(),
                },
            )
        }

        /// Reauthorisation.
        fn new_1() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_1),
                RequestPayloadOptions::Reauthorisation {
                    iris_pair: IrisPairDescriptor::new_0(20),
                    parent: UniquenessRequestDescriptor::new_4_11(),
                },
            )
        }

        /// ResetCheck.
        fn new_2() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_2),
                RequestPayloadOptions::ResetCheck {
                    iris_pair: IrisPairDescriptor::new_0(22),
                },
            )
        }

        /// ResetUpdate.
        fn new_3() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_3),
                RequestPayloadOptions::ResetUpdate {
                    iris_pair: IrisPairDescriptor::new_0(24),
                    parent: UniquenessRequestDescriptor::new_4_12(),
                },
            )
        }

        /// Uniqueness 00.
        fn new_4(descriptor_label: &str, iris_pair_offset: usize) -> Self {
            Self::new(
                Some(descriptor_label),
                RequestPayloadOptions::Uniqueness {
                    iris_pair: IrisPairDescriptor::new_0(iris_pair_offset),
                    insertion_layers: None,
                },
            )
        }

        /// Uniqueness 00.
        fn new_4_00() -> Self {
            Self::new_4(REQUEST_DESCRIPTOR_4_00, 0)
        }

        /// Uniqueness 01.
        fn new_4_01() -> Self {
            Self::new_4(REQUEST_DESCRIPTOR_4_01, 2)
        }

        /// Uniqueness 02.
        fn new_4_02() -> Self {
            Self::new_4(REQUEST_DESCRIPTOR_4_02, 4)
        }

        /// Uniqueness 10.
        fn new_4_10() -> Self {
            Self::new_4(REQUEST_DESCRIPTOR_4_10, 10)
        }

        /// Uniqueness 11.
        fn new_4_11() -> Self {
            Self::new_4(REQUEST_DESCRIPTOR_4_11, 12)
        }

        /// Uniqueness 12.
        fn new_4_12() -> Self {
            Self::new_4(REQUEST_DESCRIPTOR_4_12, 14)
        }

        fn new_batch_0() -> Vec<Self> {
            // Uniqueness only.
            vec![Self::new_4_00(), Self::new_4_01(), Self::new_4_02()]
        }

        fn new_batch_1() -> Vec<Self> {
            vec![
                Self::new_0(),
                Self::new_1(),
                Self::new_2(),
                Self::new_3(),
                Self::new_4_10(),
                Self::new_4_11(),
                Self::new_4_12(),
            ]
        }
    }

    #[test]
    fn test_new_request_options() {
        for entity_factory in [
            RequestOptions::new_0,
            RequestOptions::new_1,
            RequestOptions::new_2,
            RequestOptions::new_3,
            RequestOptions::new_4_00,
            RequestOptions::new_4_01,
            RequestOptions::new_4_02,
        ] {
            let _ = entity_factory();
        }
    }

    #[test]
    fn test_new_request_options_batch() {
        for entity_factory in [RequestOptions::new_batch_0, RequestOptions::new_batch_1] {
            let _ = entity_factory();
        }
    }

    #[test]
    fn test_new_request_batch_options_1() {
        for entity_factory in [
            RequestBatchOptions::new_series_1,
            RequestBatchOptions::new_simple_1,
        ] {
            let _ = entity_factory();
        }
    }

    #[test]
    fn test_serialise_request_batch_options_to_json() {
        for entity_factory in [
            RequestBatchOptions::new_series_1,
            RequestBatchOptions::new_simple_1,
        ] {
            let entity = entity_factory();
            let _ = serde_json::to_string(&entity).unwrap();
        }
    }

    #[test]
    fn test_serialise_request_batch_options_to_toml() {
        for entity_factory in [
            RequestBatchOptions::new_series_1,
            RequestBatchOptions::new_simple_1,
        ] {
            let entity = entity_factory();
            let _ = toml::to_string(&entity).unwrap();
        }
    }
}
