use serde::{Deserialize, Serialize};

use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::utils::serialization::iris_ndjson::IrisSelection;

use crate::client::typeset::IrisPairDescriptor;

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

    /// Validates that every parent label references a declared Uniqueness request.
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
                        matches!(item.payload(), RequestPayloadOptions::Uniqueness { .. })
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
                                "parent '{}' must be a Uniqueness request",
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

    /// Returns an error if a child references a parent in the same or later batch.
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

    pub fn iris_pair(&self) -> Option<&IrisPairDescriptor> {
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
        parent: String,
    },
    // Options over a reauthorisation request payload.
    Reauthorisation {
        iris_pair: IrisPairDescriptor,
        parent: String,
    },
    // Options over a reset check request payload.
    ResetCheck {
        iris_pair: IrisPairDescriptor,
    },
    // Options over a reset update request payload.
    ResetUpdate {
        iris_pair: IrisPairDescriptor,
        parent: String,
    },
    // Options over a uniqueness request payload.
    Uniqueness {
        iris_pair: IrisPairDescriptor,
        insertion_layers: Option<(usize, usize)>,
    },
}

impl RequestPayloadOptions {
    pub fn iris_pair(&self) -> Option<&IrisPairDescriptor> {
        match &self {
            Self::IdentityDeletion { .. } | Self::ResetCheck { .. } => None,
            Self::Reauthorisation { iris_pair, .. }
            | Self::ResetUpdate { iris_pair, .. }
            | Self::Uniqueness { iris_pair, .. } => Some(iris_pair),
        }
    }

    pub fn label_of_parent(&self) -> Option<String> {
        match &self {
            Self::IdentityDeletion { parent }
            | Self::Reauthorisation { parent, .. }
            | Self::ResetUpdate { parent, .. } => Some(parent.clone()),
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
