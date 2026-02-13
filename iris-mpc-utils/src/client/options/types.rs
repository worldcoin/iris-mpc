use serde::{Deserialize, Serialize};

use iris_mpc_common::IrisSerialId;

/// AWS specific configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwsOptions {
    /// Execution environment.
    environment: String,

    /// Base URL for downloading node encryption public keys.
    public_key_base_url: String,

    /// S3: request ingress queue URL.
    s3_request_bucket_name: String,

    /// SNS: system request ingress queue topic.
    sns_request_topic_arn: String,

    /// SQS: long polling interval (seconds).
    sqs_long_poll_wait_time: usize,

    /// SQS: system response eqgress queue URL.
    sqs_response_queue_url: String,

    /// SQS: wait time (seconds) between receive message polling.
    sqs_wait_time_seconds: usize,
}

impl AwsOptions {
    pub fn environment(&self) -> &String {
        &self.environment
    }

    pub fn public_key_base_url(&self) -> &String {
        &self.public_key_base_url
    }

    pub fn s3_request_bucket_name(&self) -> &String {
        &self.s3_request_bucket_name
    }

    pub fn sns_request_topic_arn(&self) -> &String {
        &self.sns_request_topic_arn
    }

    pub fn sqs_long_poll_wait_time(&self) -> &usize {
        &self.sqs_long_poll_wait_time
    }

    pub fn sqs_response_queue_url(&self) -> &String {
        &self.sqs_response_queue_url
    }

    pub fn sqs_wait_time_seconds(&self) -> &usize {
        &self.sqs_wait_time_seconds
    }
}

/// Enumeration over types of strategy to apply when selecting
/// Iris codes from an NDJSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrisCodeSelectionStrategyOptions {
    // All Iris codes are selected.
    All,
    // Every other Iris code is selected beginning at an even offset.
    Even,
    // Every other Iris code is selected beginning at an odd offset.
    Odd,
}

/// A descriptor over an Iris code cached within a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrisDescriptorOptions {
    // Ordinal identifer typically pointing to a row within an NDJSON file.
    index: usize,

    // TODO: Optionally apply noise, rotations, mirroring, etc.
    mutation: Option<()>,
}

impl IrisDescriptorOptions {
    pub fn new(index: usize, mutation: Option<()>) -> Self {
        Self { index, mutation }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn mutation(&self) -> Option<()> {
        self.mutation
    }
}

/// A descriptor over a pair of Iris codes cached within a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrisPairDescriptorOptions((IrisDescriptorOptions, IrisDescriptorOptions));

impl IrisPairDescriptorOptions {
    pub fn new(left: IrisDescriptorOptions, right: IrisDescriptorOptions) -> Self {
        Self((left, right))
    }

    pub fn new_from_indexes(left: usize, right: usize) -> Self {
        Self::new(
            IrisDescriptorOptions::new(left, None),
            IrisDescriptorOptions::new(right, None),
        )
    }

    pub fn left(&self) -> &IrisDescriptorOptions {
        &self.0 .0
    }

    pub fn right(&self) -> &IrisDescriptorOptions {
        &self.0 .1
    }

    pub fn indexes(&self) -> (usize, usize) {
        (self.left().index, self.right().index)
    }
}

/// Set of variants over inputs to request batch creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestBatchOptions {
    // Options for instanitating a set of interleaved request batches.
    Complex {
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
    pub fn new_complex(batches: Vec<Vec<RequestOptions>>) -> Self {
        Self::Complex { batches }
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

    /// Returns set of Iris code indexes to be read from NDJSON file.
    pub(crate) fn iris_code_indexes(&self) -> Vec<usize> {
        match self {
            Self::Complex { batches } => batches
                .iter()
                .flat_map(|batch| batch.iter())
                .filter_map(|item| item.iris_pair())
                .flat_map(|iris_pair| [iris_pair.left().index(), iris_pair.right().index()])
                .collect(),
            _ => vec![],
        }
    }

    /// Returns set of declared request labels.
    pub(crate) fn labels(&self) -> Vec<String> {
        match self {
            Self::Complex { batches } => batches
                .iter()
                .flat_map(|batch| batch.iter())
                .filter_map(|item| item.label())
                .collect(),
            _ => vec![],
        }
    }

    /// Returns set of declared parent request labels.
    pub(crate) fn labels_of_parents(&self) -> Vec<String> {
        match self {
            Self::Complex { batches } => batches
                .iter()
                .flat_map(|batch| batch.iter())
                .filter_map(|item| item.label_of_parent())
                .collect(),
            _ => vec![],
        }
    }
}

/// Options over an individual request within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOptions {
    // Optional label for cross referencing within batch.
    label: Option<String>,

    // Inner request payload options.
    payload: RequestPayloadOptions,
}

impl RequestOptions {
    pub fn new(label: Option<&str>, payload: RequestPayloadOptions) -> Self {
        Self {
            label: label.map(|s| s.to_string()),
            payload,
        }
    }

    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    pub fn payload(&self) -> &RequestPayloadOptions {
        &self.payload
    }

    pub fn iris_pair(&self) -> Option<&IrisPairDescriptorOptions> {
        self.payload().iris_pair()
    }

    pub fn label_of_parent(&self) -> Option<String> {
        self.payload().label_of_parent()
    }
}

/// Options over a request's payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestPayloadOptions {
    // Options over a deletion request payload.
    IdentityDeletion {
        parent: UniquenessRequestDescriptorOptions,
    },
    // Options over a reauthorisation request payload.
    Reauthorisation {
        iris_pair: IrisPairDescriptorOptions,
        parent: UniquenessRequestDescriptorOptions,
    },
    // Options over a reset check request payload.
    ResetCheck {
        iris_pair: IrisPairDescriptorOptions,
    },
    // Options over a reset update request payload.
    ResetUpdate {
        iris_pair: IrisPairDescriptorOptions,
        parent: UniquenessRequestDescriptorOptions,
    },
    // Options over a uniqueness request payload.
    Uniqueness {
        iris_pair: IrisPairDescriptorOptions,
        insertion_layers: Option<(usize, usize)>,
    },
}

impl RequestPayloadOptions {
    pub fn iris_pair(&self) -> Option<&IrisPairDescriptorOptions> {
        match &self {
            Self::IdentityDeletion { .. } | Self::ResetCheck { .. } => None,
            Self::Reauthorisation { iris_pair, .. }
            | Self::ResetUpdate { iris_pair, .. }
            | Self::Uniqueness { iris_pair, .. } => Some(iris_pair),
        }
    }

    pub fn label_of_parent(&self) -> Option<String> {
        match &self {
            Self::IdentityDeletion { parent } => parent.label(),
            Self::Reauthorisation { parent, .. } => parent.label(),
            Self::ResetUpdate { parent, .. } => parent.label(),
            _ => None,
        }
    }
}

/// Set of variants over inputs to iris shares generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharesGeneratorOptions {
    /// Shares are generated via a random number generator.
    FromCompute {
        // An optional RNG seed.
        rng_seed: Option<u64>,
    },
    /// Shares are generated from a pre-built file.
    FromFile {
        // Path to an NDJSON file.
        path_to_ndjson_file: String,

        // An optional RNG seed.
        rng_seed: Option<u64>,

        // Instruction in respect of Iris code selection.
        selection_strategy: Option<IrisCodeSelectionStrategyOptions>,
    },
}

/// A descriptor over a system Request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UniquenessRequestDescriptorOptions {
    // Label to identify request within batch/file scope.
    Label(String),

    // Iris serial identifer as assigned by remote system.
    SerialId(IrisSerialId),
}

impl UniquenessRequestDescriptorOptions {
    pub fn label(&self) -> Option<String> {
        match self {
            Self::Label(label) => Some(label.clone()),
            _ => None,
        }
    }

    pub fn new_label(label: &str) -> Self {
        Self::Label(label.to_string())
    }

    pub fn new_serial_id(serial_id: IrisSerialId) -> Self {
        Self::SerialId(serial_id)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json;
    use toml;

    use iris_mpc_common::helpers::smpc_request;

    use super::{
        IrisDescriptorOptions, IrisPairDescriptorOptions, RequestBatchOptions, RequestOptions,
        RequestPayloadOptions, UniquenessRequestDescriptorOptions,
    };

    pub(crate) const REQUEST_DESCRIPTOR_0: &str = "IdentityDeletion-0";
    pub(crate) const REQUEST_DESCRIPTOR_1: &str = "ResetCheck-0";
    pub(crate) const REQUEST_DESCRIPTOR_2: &str = "ResetUpdate-0";
    pub(crate) const REQUEST_DESCRIPTOR_3: &str = "Reauthorisation-0";
    pub(crate) const REQUEST_DESCRIPTOR_4_00: &str = "Uniqueness-00";
    pub(crate) const REQUEST_DESCRIPTOR_4_01: &str = "Uniqueness-01";
    pub(crate) const REQUEST_DESCRIPTOR_4_02: &str = "Uniqueness-02";
    pub(crate) const REQUEST_DESCRIPTOR_4_10: &str = "Uniqueness-10";
    pub(crate) const REQUEST_DESCRIPTOR_4_11: &str = "Uniqueness-11";
    pub(crate) const REQUEST_DESCRIPTOR_4_12: &str = "Uniqueness-12";

    impl IrisPairDescriptorOptions {
        pub(crate) fn new_0(offset: usize) -> Self {
            Self::new(
                IrisDescriptorOptions::new(offset + 1, None),
                IrisDescriptorOptions::new(offset + 2, None),
            )
        }
    }

    impl UniquenessRequestDescriptorOptions {
        #[allow(dead_code)]
        pub(crate) fn new_4_00() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_00)
        }

        #[allow(dead_code)]
        pub(crate) fn new_4_01() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_01)
        }

        #[allow(dead_code)]
        pub(crate) fn new_4_02() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_02)
        }

        pub(crate) fn new_4_10() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_10)
        }

        pub(crate) fn new_4_11() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_11)
        }

        pub(crate) fn new_4_12() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_12)
        }
    }

    impl RequestBatchOptions {
        fn new_complex_1() -> Self {
            Self::new_complex(vec![
                RequestOptions::new_batch_0(),
                RequestOptions::new_batch_1(),
            ])
        }

        fn new_simple_1() -> Self {
            Self::new_simple(smpc_request::UNIQUENESS_MESSAGE_TYPE, 10, 10, None)
        }
    }

    impl RequestOptions {
        /// Identity deletion.
        fn new_0() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_0),
                RequestPayloadOptions::IdentityDeletion {
                    parent: UniquenessRequestDescriptorOptions::new_4_10(),
                },
            )
        }

        /// Reauthorisation.
        fn new_1() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_1),
                RequestPayloadOptions::Reauthorisation {
                    iris_pair: IrisPairDescriptorOptions::new_0(20),
                    parent: UniquenessRequestDescriptorOptions::new_4_11(),
                },
            )
        }

        /// ResetCheck.
        fn new_2() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_2),
                RequestPayloadOptions::ResetCheck {
                    iris_pair: IrisPairDescriptorOptions::new_0(22),
                },
            )
        }

        /// ResetUpdate.
        fn new_3() -> Self {
            Self::new(
                Some(REQUEST_DESCRIPTOR_3),
                RequestPayloadOptions::ResetUpdate {
                    iris_pair: IrisPairDescriptorOptions::new_0(24),
                    parent: UniquenessRequestDescriptorOptions::new_4_12(),
                },
            )
        }

        /// Uniqueness 00.
        fn new_4(descriptor_label: &str, iris_pair_offset: usize) -> Self {
            Self::new(
                Some(descriptor_label),
                RequestPayloadOptions::Uniqueness {
                    iris_pair: IrisPairDescriptorOptions::new_0(iris_pair_offset),
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
            RequestBatchOptions::new_complex_1,
            RequestBatchOptions::new_simple_1,
        ] {
            let _ = entity_factory();
        }
    }

    #[test]
    fn test_serialise_request_batch_options_to_json() {
        for entity_factory in [
            RequestBatchOptions::new_complex_1,
            RequestBatchOptions::new_simple_1,
        ] {
            let entity = entity_factory();
            let _ = serde_json::to_string(&entity).unwrap();
        }
    }

    #[test]
    fn test_serialise_request_batch_options_to_toml() {
        for entity_factory in [
            RequestBatchOptions::new_complex_1,
            RequestBatchOptions::new_simple_1,
        ] {
            let entity = entity_factory();
            let _ = toml::to_string(&entity).unwrap();
        }
    }
}
