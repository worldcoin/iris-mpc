use std::fmt;

use serde::{
    de::{self, Deserializer, Visitor},
    ser::Serializer,
    Deserialize, Serialize,
};

use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::utils::serialization::iris_ndjson::IrisSelection;
use uuid::Uuid;

use crate::client::{typeset::IrisPairDescriptor, Request, RequestInfo};

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

    /// SQS: system response egress queue URLs.
    sqs_response_queue_urls: Vec<String>,

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

    pub fn sqs_response_queue_urls(&self) -> &Vec<String> {
        &self.sqs_response_queue_urls
    }

    pub fn sqs_wait_time_seconds(&self) -> &usize {
        &self.sqs_wait_time_seconds
    }
}

/// A parent reference: either a label (resolved later) or a known serial ID.
///
/// In TOML:
///   `parent = "some-label"` → `Parent::Label` (child waits for parent)
///   `parent = 42`           → `Parent::Id`    (serial_id already known)
#[derive(Debug, Clone)]
pub enum Parent {
    /// A label referring to a Uniqueness request whose serial ID is not yet known.
    Label(String),
    /// A known Iris serial ID; no dependency resolution needed.
    Id(IrisSerialId),
}

impl Serialize for Parent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Parent::Label(s) => serializer.serialize_str(s),
            Parent::Id(id) => serializer.serialize_u32(*id),
        }
    }
}

impl<'de> Deserialize<'de> for Parent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ParentVisitor;

        impl<'de> Visitor<'de> for ParentVisitor {
            type Value = Parent;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a string label or an integer serial ID")
            }

            fn visit_u32<E: de::Error>(self, v: u32) -> Result<Self::Value, E> {
                Ok(Parent::Id(v))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                u32::try_from(v)
                    .map(Parent::Id)
                    .map_err(|_| E::custom("serial ID out of range for u32"))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                u32::try_from(v)
                    .map(Parent::Id)
                    .map_err(|_| E::custom("serial ID must be non-negative"))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(Parent::Label(v.to_string()))
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(Parent::Label(v))
            }
        }

        deserializer.deserialize_any(ParentVisitor)
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
    // Options for generating a set of homogeneous request batches.
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
    /// Used to make large heterogenerous batches with minimal configuration
    Random {
        batch_count: usize,

        batch_size: usize,

        percent_uniqueness: usize,

        percent_reauth: usize,

        percent_other: usize,
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

    /// Returns set of Iris code pairs as (left_index, right_index) tuples.
    pub(crate) fn iris_code_pairs(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Complex { batches } => batches
                .iter()
                .flat_map(|batch| batch.iter())
                .filter_map(|item| item.iris_pair())
                .map(|iris_pair| (iris_pair.left().index(), iris_pair.right().index()))
                .collect(),
            _ => unreachable!("not valid for Simple variant"),
        }
    }

    /// Returns the first duplicate label found, if any.
    ///
    /// # Panics
    /// Panics if called on `RequestBatchOptions::Simple`.
    pub(crate) fn find_duplicate_label(&self) -> Option<String> {
        let labels = self.labels();
        let mut seen = std::collections::HashSet::with_capacity(labels.len());
        labels.into_iter().find(|l| !seen.insert(l.clone()))
    }

    /// Validates that every `Parent::Label` references a declared Uniqueness request.
    /// Returns `Ok(())` if valid, or `Err(message)` describing the first violation.
    pub(crate) fn validate_parents(&self) -> Result<(), String> {
        match self {
            Self::Complex { batches } => {
                let all_items: Vec<_> = batches.iter().flat_map(|batch| batch.iter()).collect();
                let all_labels: std::collections::HashSet<_> =
                    all_items.iter().filter_map(|item| item.label()).collect();
                let uniqueness_labels: std::collections::HashSet<_> = all_items
                    .iter()
                    .filter(|item| {
                        matches!(
                            item.payload(),
                            RequestPayloadOptions::Uniqueness { .. }
                                | RequestPayloadOptions::Mirrored { .. }
                        )
                    })
                    .filter_map(|item| item.label())
                    .collect();
                for item in &all_items {
                    if let Some(parent_label) = item.label_of_parent() {
                        if !all_labels.contains(&parent_label) {
                            return Err(format!(
                                "contains a parent label '{}' that is not found in labels",
                                parent_label
                            ));
                        }
                        if !uniqueness_labels.contains(&parent_label) {
                            return Err(format!(
                                "parent '{}' must be a Uniqueness or Mirrored request",
                                parent_label
                            ));
                        }
                    }
                }
                Ok(())
            }
            _ => unreachable!("not valid for Simple variant"),
        }
    }

    /// Returns an error if any iris index appears in multiple different pairs.
    /// Duplicate pairs (same or swapped eyes) are allowed.
    ///
    /// # Panics
    /// Panics if called on `RequestBatchOptions::Simple`.
    pub(crate) fn validate_iris_pairs(&self) -> Result<(), String> {
        let mut index_to_pair: std::collections::HashMap<usize, (usize, usize)> =
            std::collections::HashMap::new();
        for pair in self.iris_code_pairs() {
            let normalized = if pair.0 <= pair.1 {
                pair
            } else {
                (pair.1, pair.0)
            };
            for idx in [normalized.0, normalized.1] {
                if let Some(existing) = index_to_pair.get(&idx) {
                    if *existing != normalized {
                        return Err(format!(
                            "iris index {} appears in multiple different pairs",
                            idx
                        ));
                    }
                } else {
                    index_to_pair.insert(idx, normalized);
                }
            }
        }
        Ok(())
    }

    /// Returns an error if a `Parent::Label` child references a parent in the same or later batch.
    pub(crate) fn validate_batch_ordering(&self) -> Result<(), String> {
        match self {
            Self::Complex { batches } => {
                let mut labels_seen = std::collections::HashSet::new();
                for batch in batches {
                    for item in batch {
                        if let Some(parent_label) = item.label_of_parent() {
                            if !labels_seen.contains(&parent_label) {
                                return Err(format!(
                                    "parent '{}' must be in an earlier batch",
                                    parent_label
                                ));
                            }
                        }
                    }
                    let batch_labels: std::collections::HashSet<_> =
                        batch.iter().filter_map(|item| item.label()).collect();
                    labels_seen.extend(batch_labels);
                }
                Ok(())
            }
            _ => unreachable!("not valid for Simple variant"),
        }
    }

    /// Returns set of declared request labels.
    pub fn labels(&self) -> Vec<String> {
        match self {
            Self::Complex { batches } => batches
                .iter()
                .flat_map(|batch| batch.iter())
                .filter_map(|item| item.label())
                .collect(),
            _ => unreachable!("not valid for Simple variant"),
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

    // Optional expected response fields for validation.
    #[serde(default)]
    expected: Option<serde_json::Value>,
}

impl RequestOptions {
    pub fn new(label: Option<&str>, payload: RequestPayloadOptions) -> Self {
        Self {
            label: label.map(|s| s.to_string()),
            expected: None,
            payload,
        }
    }

    pub fn with_expected(mut self, expected: serde_json::Value) -> Self {
        self.expected = Some(expected);
        self
    }

    pub fn expected(&self) -> Option<&serde_json::Value> {
        self.expected.as_ref()
    }

    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    pub fn payload(&self) -> &RequestPayloadOptions {
        &self.payload
    }

    pub fn iris_pair(&self) -> Option<&IrisPairDescriptor> {
        self.payload().iris_pair()
    }

    /// Returns the parent label if this request has a `Parent::Label` parent.
    pub fn label_of_parent(&self) -> Option<String> {
        self.payload().label_of_parent()
    }

    pub fn make_request(
        &self,
        info: RequestInfo,
        parent_serial_id: Option<IrisSerialId>,
    ) -> Request {
        let corr_uuid = Uuid::new_v4();

        match self.payload() {
            RequestPayloadOptions::Uniqueness { iris_pair, .. }
            | RequestPayloadOptions::Mirrored { iris_pair, .. } => Request::Uniqueness {
                info,
                iris_pair: *iris_pair,
                signup_id: corr_uuid,
            },
            RequestPayloadOptions::Reauthorisation { iris_pair, .. } => Request::Reauthorization {
                info,
                iris_pair: *iris_pair,
                parent: parent_serial_id.unwrap(),
                reauth_id: corr_uuid,
            },
            RequestPayloadOptions::ResetCheck { iris_pair } => Request::ResetCheck {
                info,
                iris_pair: *iris_pair,
                reset_id: corr_uuid,
            },
            RequestPayloadOptions::ResetUpdate { iris_pair, .. } => Request::ResetUpdate {
                info,
                iris_pair: *iris_pair,
                parent: parent_serial_id.unwrap(),
                reset_id: corr_uuid,
            },
            RequestPayloadOptions::IdentityDeletion { .. } => Request::IdentityDeletion {
                info,
                parent: parent_serial_id.unwrap(),
            },
        }
    }

    pub fn is_mirrored(&self) -> bool {
        matches!(self.payload(), RequestPayloadOptions::Mirrored { .. })
    }

    pub fn get_parent(&self) -> Option<Parent> {
        match self.payload() {
            RequestPayloadOptions::IdentityDeletion { parent }
            | RequestPayloadOptions::Reauthorisation { parent, .. }
            | RequestPayloadOptions::ResetUpdate { parent, .. } => Some(parent.clone()),
            _ => None,
        }
    }
}

/// Options over a request's payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestPayloadOptions {
    // Options over a deletion request payload.
    IdentityDeletion {
        parent: Parent,
    },
    // Options over a reauthorisation request payload.
    Reauthorisation {
        #[serde(default)]
        iris_pair: Option<IrisPairDescriptor>,
        parent: Parent,
    },
    // Options over a reset check request payload.
    ResetCheck {
        #[serde(default)]
        iris_pair: Option<IrisPairDescriptor>,
    },
    // Options over a reset update request payload.
    ResetUpdate {
        #[serde(default)]
        iris_pair: Option<IrisPairDescriptor>,
        parent: Parent,
    },
    // Options over a uniqueness request payload.
    Uniqueness {
        #[serde(default)]
        iris_pair: Option<IrisPairDescriptor>,
        insertion_layers: Option<(usize, usize)>,
    },
    // Options over a mirrored uniqueness request payload.
    // Generates a Uniqueness request whose iris shares are mirror-transformed.
    Mirrored {
        #[serde(default)]
        iris_pair: Option<IrisPairDescriptor>,
        insertion_layers: Option<(usize, usize)>,
    },
}

impl RequestPayloadOptions {
    pub fn iris_pair(&self) -> Option<&IrisPairDescriptor> {
        match &self {
            Self::IdentityDeletion { .. } => None,
            Self::Reauthorisation { iris_pair, .. }
            | Self::ResetCheck { iris_pair, .. }
            | Self::ResetUpdate { iris_pair, .. }
            | Self::Uniqueness { iris_pair, .. }
            | Self::Mirrored { iris_pair, .. } => iris_pair.as_ref(),
        }
    }

    /// Returns the parent label only for `Parent::Label` variants.
    pub fn label_of_parent(&self) -> Option<String> {
        match &self {
            Self::IdentityDeletion {
                parent: Parent::Label(l),
            }
            | Self::Reauthorisation {
                parent: Parent::Label(l),
                ..
            }
            | Self::ResetUpdate {
                parent: Parent::Label(l),
                ..
            } => Some(l.clone()),
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
        // Path to an NDJSON file (optional in TOML; can be supplied via CLI).
        #[serde(default)]
        path_to_ndjson_file: Option<String>,

        // An optional RNG seed.
        rng_seed: Option<u64>,

        // Instruction in respect of Iris code selection.
        selection_strategy: Option<IrisSelection>,
    },
}

impl IntoIterator for RequestBatchOptions {
    type Item = Vec<RequestOptions>;
    type IntoIter = std::vec::IntoIter<Vec<RequestOptions>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            RequestBatchOptions::Complex { batches } => batches.into_iter(),
            RequestBatchOptions::Random {
                batch_count,
                batch_size,
                percent_uniqueness,
                percent_reauth,
                percent_other,
            } => random_into_iter(
                batch_count,
                batch_size,
                percent_uniqueness,
                percent_reauth,
                percent_other,
            ),
            RequestBatchOptions::Simple {
                batch_count,
                batch_kind,
                batch_size,
                ..
            } => simple_into_iter(batch_count, batch_kind, batch_size),
        }
    }
}

fn random_into_iter(
    batch_count: usize,
    batch_size: usize,
    percent_uniqueness: usize,
    percent_reauth: usize,
    percent_other: usize,
) -> std::vec::IntoIter<Vec<RequestOptions>> {
    use rand::seq::SliceRandom;
    use rand::Rng;
    use std::collections::BTreeSet;

    let mut batches: Vec<Vec<RequestOptions>> = Vec::new();
    let mut prev_labels: BTreeSet<String> = BTreeSet::new();
    let mut uniqueness_counter = 0;
    let mut rng = rand::thread_rng();

    // Calculate number of each type of request in this batch
    let num_uniqueness = (batch_size * percent_uniqueness) / 100;
    let num_reauth = (batch_size * percent_reauth) / 100;
    let num_other = (batch_size * percent_other) / 100;

    // Start with an initial batch of 50 uniqueness requests to seed the pool
    let mut initial_batch = Vec::new();
    for _ in 0..50 {
        let label = format!("uniqueness-{}", uniqueness_counter);
        uniqueness_counter += 1;

        initial_batch.push(RequestOptions::new(
            Some(&label),
            RequestPayloadOptions::Uniqueness {
                iris_pair: None,
                insertion_layers: None,
            },
        ));

        prev_labels.insert(label);
    }
    batches.push(initial_batch);

    // Generate the remaining batches
    for _batch_idx in 0..batch_count {
        let mut batch = Vec::new();

        let mut new_labels = BTreeSet::new();

        // Generate uniqueness requests for this batch
        for _ in 0..num_uniqueness {
            let label = format!("uniqueness-{}", uniqueness_counter);
            uniqueness_counter += 1;

            batch.push(RequestOptions::new(
                Some(&label),
                RequestPayloadOptions::Uniqueness {
                    iris_pair: None,
                    insertion_layers: None,
                },
            ));

            new_labels.insert(label);
        }

        // Generate reauth requests - only reference labels from previous batches
        for _ in 0..num_reauth {
            let payload = if !prev_labels.is_empty() {
                let random_index = rng.gen_range(0..prev_labels.len());
                let parent_label = prev_labels.iter().nth(random_index).unwrap().clone();
                RequestPayloadOptions::Reauthorisation {
                    iris_pair: None,
                    parent: Parent::Label(parent_label),
                }
            } else {
                // No labels available yet, use ResetCheck as fallback
                RequestPayloadOptions::ResetCheck { iris_pair: None }
            };

            batch.push(RequestOptions::new(None, payload));
        }

        // Generate other requests - only reference labels from previous batches
        for i in 0..num_other {
            let payload = if !prev_labels.is_empty() {
                match i % 3 {
                    0 => {
                        // IdentityDeletion - remove the label after using it
                        let current_count = prev_labels.len();
                        let random_index = rng.gen_range(0..current_count);
                        let parent_label = prev_labels.iter().nth(random_index).unwrap().clone();
                        prev_labels.remove(&parent_label);
                        RequestPayloadOptions::IdentityDeletion {
                            parent: Parent::Label(parent_label),
                        }
                    }
                    1 => RequestPayloadOptions::ResetCheck { iris_pair: None },
                    2 => {
                        let current_count = prev_labels.len();
                        let random_index = rng.gen_range(0..current_count);
                        let parent_label = prev_labels.iter().nth(random_index).unwrap().clone();
                        RequestPayloadOptions::ResetUpdate {
                            iris_pair: None,
                            parent: Parent::Label(parent_label),
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                // No labels available, use ResetCheck as fallback
                RequestPayloadOptions::ResetCheck { iris_pair: None }
            };

            batch.push(RequestOptions::new(None, payload));
        }

        for x in new_labels.into_iter() {
            prev_labels.insert(x);
        }

        batch.shuffle(&mut rng);
        batches.push(batch);
    }

    batches.into_iter()
}

fn simple_into_iter(
    batch_count: usize,
    batch_kind: String,
    batch_size: usize,
) -> std::vec::IntoIter<Vec<RequestOptions>> {
    use iris_mpc_common::helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    };

    let requires_parent = matches!(
        batch_kind.as_str(),
        IDENTITY_DELETION_MESSAGE_TYPE | REAUTH_MESSAGE_TYPE | RESET_UPDATE_MESSAGE_TYPE
    );

    let mut v: Vec<Vec<RequestOptions>> = vec![];
    for _ in 0..batch_count {
        if !requires_parent {
            let batch = (0..batch_size)
                .map(|_| match batch_kind.as_str() {
                    UNIQUENESS_MESSAGE_TYPE => RequestOptions::new(
                        None,
                        RequestPayloadOptions::Uniqueness {
                            iris_pair: None,
                            insertion_layers: None,
                        },
                    ),
                    RESET_CHECK_MESSAGE_TYPE => RequestOptions::new(
                        None,
                        RequestPayloadOptions::ResetCheck { iris_pair: None },
                    ),
                    _ => unreachable!(
                        "Simple batch_kind '{}' should have been rejected by validation",
                        batch_kind
                    ),
                })
                .collect();
            v.push(batch);
        } else {
            // Two batches: uniqueness preamble (with UUID labels) + desired type
            // (referencing those labels via Parent::Label).
            let mut uniqueness_batch = vec![];
            let mut child_batch = vec![];
            for _ in 0..batch_size {
                let label = uuid::Uuid::new_v4().to_string();
                uniqueness_batch.push(RequestOptions::new(
                    Some(label.as_str()),
                    RequestPayloadOptions::Uniqueness {
                        iris_pair: None,
                        insertion_layers: None,
                    },
                ));
                let payload = match batch_kind.as_str() {
                    IDENTITY_DELETION_MESSAGE_TYPE => RequestPayloadOptions::IdentityDeletion {
                        parent: Parent::Label(label),
                    },
                    REAUTH_MESSAGE_TYPE => RequestPayloadOptions::Reauthorisation {
                        iris_pair: None,
                        parent: Parent::Label(label),
                    },
                    RESET_UPDATE_MESSAGE_TYPE => RequestPayloadOptions::ResetUpdate {
                        iris_pair: None,
                        parent: Parent::Label(label),
                    },
                    _ => unreachable!("already checked requires_parent"),
                };
                child_batch.push(RequestOptions::new(None, payload));
            }
            v.push(uniqueness_batch);
            v.push(child_batch);
        }
    }
    v.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_batch_options_simple_roundtrip() {
        let toml_str = r#"
            [Simple]
            batch_count = 2
            batch_size = 10
            batch_kind = "uniqueness"
        "#;
        let opts: RequestBatchOptions = toml::from_str(toml_str).unwrap();
        let _ = toml::to_string(&opts).unwrap();
    }

    #[test]
    fn test_request_batch_options_simple_with_serial_id_roundtrip() {
        let toml_str = r#"
            [Simple]
            batch_count = 2
            batch_size = 10
            batch_kind = "reauth"
            known_iris_serial_id = 1
        "#;
        let opts: RequestBatchOptions = toml::from_str(toml_str).unwrap();
        let _ = toml::to_string(&opts).unwrap();
    }

    #[test]
    fn test_request_batch_options_complex_roundtrip() {
        let toml_str = r#"
            [Complex]
            batches = [
                [
                    { label = "Uniqueness-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                    { label = "Uniqueness-1", payload = { Uniqueness = { iris_pair = [{ index = 3 }, { index = 4 }] } } },
                ],
                [
                    { label = "Deletion-0", payload = { IdentityDeletion = { parent = "Uniqueness-0" } } },
                    { label = "Reauth-0", payload = { Reauthorisation = { iris_pair = [{ index = 5 }, { index = 6 }], parent = "Uniqueness-1" } } },
                    { label = "Check-0", payload = { ResetCheck = { iris_pair = [{ index = 7 }, { index = 8 }] } } },
                    { label = "Update-0", payload = { ResetUpdate = { iris_pair = [{ index = 9 }, { index = 10 }], parent = "Uniqueness-0" } } },
                ],
            ]
        "#;
        let opts: RequestBatchOptions = toml::from_str(toml_str).unwrap();
        let _ = toml::to_string(&opts).unwrap();
    }

    #[test]
    fn test_parent_label_deserialization() {
        let toml_str = r#"
            [Complex]
            batches = [[
                { label = "U-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                { label = "D-0", payload = { IdentityDeletion = { parent = "U-0" } } },
            ]]
        "#;
        let opts: RequestBatchOptions = toml::from_str(toml_str).unwrap();
        if let RequestBatchOptions::Complex { batches } = &opts {
            let deletion = &batches[0][1];
            if let RequestPayloadOptions::IdentityDeletion { parent } = deletion.payload() {
                assert!(matches!(parent, Parent::Label(l) if l == "U-0"));
            } else {
                panic!("Expected IdentityDeletion");
            }
        }
    }

    #[test]
    fn test_parent_id_deserialization() {
        let toml_str = r#"
            [Complex]
            batches = [[
                { label = "D-0", payload = { IdentityDeletion = { parent = 42 } } },
            ]]
        "#;
        let opts: RequestBatchOptions = toml::from_str(toml_str).unwrap();
        if let RequestBatchOptions::Complex { batches } = &opts {
            let deletion = &batches[0][0];
            if let RequestPayloadOptions::IdentityDeletion { parent } = deletion.payload() {
                assert!(matches!(parent, Parent::Id(42)));
            } else {
                panic!("Expected IdentityDeletion");
            }
        }
    }

    #[test]
    fn test_shares_generator_options_from_compute_roundtrip() {
        let toml_str = r#"
            [FromCompute]
            rng_seed = 42
        "#;
        let opts: SharesGeneratorOptions = toml::from_str(toml_str).unwrap();
        let _ = toml::to_string(&opts).unwrap();
    }

    #[test]
    fn test_shares_generator_options_from_file_roundtrip() {
        let toml_str = r#"
            [FromFile]
            path_to_ndjson_file = "/tmp/irises.ndjson"
            rng_seed = 42
            selection_strategy = "All"
        "#;
        let opts: SharesGeneratorOptions = toml::from_str(toml_str).unwrap();
        let _ = toml::to_string(&opts).unwrap();
    }

    #[test]
    fn test_shares_generator_options_from_file_without_path_roundtrip() {
        let toml_str = r#"
            [FromFile]
            rng_seed = 42
            selection_strategy = "All"
        "#;
        let opts: SharesGeneratorOptions = toml::from_str(toml_str).unwrap();
        match &opts {
            SharesGeneratorOptions::FromFile {
                path_to_ndjson_file,
                ..
            } => assert!(path_to_ndjson_file.is_none()),
            _ => panic!("Expected FromFile variant"),
        }
    }
}
