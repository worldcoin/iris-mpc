use serde::{Deserialize, Serialize};

mod types;

pub use types::AwsOptions;
pub use types::{Parent, RequestBatchOptions, RequestPayloadOptions, SharesGeneratorOptions};

use crate::client::{BatchKind, ServiceClientError};

/// Service client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceClientOptions {
    // A representation of remote system state prior to execution. E.G. a hex encoded hash value.
    prestate: Option<String>,

    // Associated request batch generation configuration.
    pub request_batch: RequestBatchOptions,

    // Associated Iris shares generator configuration.
    pub shares_generator: SharesGeneratorOptions,
}

impl ServiceClientOptions {
    pub fn request_batch(&self) -> &RequestBatchOptions {
        &self.request_batch
    }

    pub fn shares_generator(&self) -> &SharesGeneratorOptions {
        &self.shares_generator
    }

    /// Overrides the NDJSON file path when shares generator is `FromFile`.
    pub fn set_iris_shares_path(&mut self, path: &str) {
        if let SharesGeneratorOptions::FromFile {
            path_to_ndjson_file,
            ..
        } = &mut self.shares_generator
        {
            *path_to_ndjson_file = Some(path.to_string());
        }
    }
}

impl ServiceClientOptions {
    pub fn validate(&self) -> Result<(), ServiceClientError> {
        // Error if FromFile is used without a path to the NDJSON file.
        if let SharesGeneratorOptions::FromFile {
            path_to_ndjson_file,
            ..
        } = self.shares_generator()
        {
            if path_to_ndjson_file.as_deref().unwrap_or("").is_empty() {
                return Err(ServiceClientError::InvalidOptions(
                    "SharesGeneratorOptions::FromFile requires a path to the NDJSON file \
                     (provide via CLI --path-to-iris-shares)"
                        .to_string(),
                ));
            }
        }

        match self.request_batch() {
            RequestBatchOptions::Complex { .. } => {
                // Error if used alongside compute shares generation.
                if matches!(
                    self.shares_generator(),
                    SharesGeneratorOptions::FromCompute { .. }
                ) {
                    return Err(ServiceClientError::InvalidOptions("RequestBatchOptions::Complex can only be used with SharesGeneratorOptions::FromFile".to_string()));
                }

                let rb = self.request_batch();
                for result in [
                    rb.validate_iris_pairs(),
                    rb.find_duplicate_label().map_or(Ok(()), |dup| {
                        Err(format!("contains duplicate label '{}'", dup))
                    }),
                    rb.validate_parents(),
                    rb.validate_batch_ordering(),
                ] {
                    if let Err(msg) = result {
                        return Err(ServiceClientError::InvalidOptions(format!(
                            "RequestBatchOptions::Complex {}",
                            msg
                        )));
                    }
                }
            }
            RequestBatchOptions::Simple {
                batch_count,
                batch_kind,
                batch_size,
                known_iris_serial_id: maybe_known_iris_serial_id,
                ..
            } => {
                // Error if total requests exceed arbitrary limit.
                if batch_count * batch_size > 1_000_000 {
                    return Err(ServiceClientError::InvalidOptions(
                        "RequestBatchOptions::Simple total requests will exceed limit of 1_000_000"
                            .to_string(),
                    ));
                }

                // Error if batch kind cannot be mapped to a supported SMPC request type.
                if BatchKind::parse(batch_kind).is_none() {
                    return Err(ServiceClientError::InvalidOptions(format!(
                        "RequestBatchOptions::Simple batch_kind ({}) is unsupported",
                        batch_kind
                    )));
                }

                // Error if known serial id exceeds reasonable upper bound.
                if let Some(known_iris_serial_id) = maybe_known_iris_serial_id {
                    if *known_iris_serial_id > 20_000_000_u32 {
                        return Err(ServiceClientError::InvalidOptions(format!(
                            "RequestBatchOptions::Simple known_iris_serial_id ({}) exceeds reasonable upper bound",
                            known_iris_serial_id
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ServiceClientOptions;
    use crate::fsys::{local::get_path_to_service_client_simple_opts, reader::read_toml};

    fn opts(toml_str: &str) -> ServiceClientOptions {
        toml::from_str(toml_str).expect("Failed to parse TOML")
    }

    fn assert_invalid_options(opts: &ServiceClientOptions, expected_substr: &str) {
        let err = opts.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(expected_substr),
            "Expected error containing '{}', got: {}",
            expected_substr,
            msg
        );
    }

    #[test]
    fn test_exec_opts_deserialization() {
        (1..=5).for_each(move |opts_idx| {
            let path_to_opts = get_path_to_service_client_simple_opts(opts_idx);
            let _ = read_toml::<ServiceClientOptions>(path_to_opts.as_path())
                .expect("Failed to deserialize service client exec options file");
        });
    }

    #[test]
    fn complex_rejects_from_compute_shares_generator() {
        let o = opts(
            r#"
            [shares_generator.FromCompute]
            rng_seed = 42
            [request_batch.Complex]
            batches = [[
                { payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
            ]]
        "#,
        );
        assert_invalid_options(&o, "can only be used with SharesGeneratorOptions::FromFile");
    }

    #[test]
    fn complex_rejects_iris_index_in_multiple_pairs() {
        let o = opts(
            r#"
            [shares_generator.FromFile]
            path_to_ndjson_file = "/tmp/test.ndjson"
            [request_batch.Complex]
            batches = [[
                { label = "U-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                { label = "U-1", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 3 }] } } },
            ]]
        "#,
        );
        assert_invalid_options(&o, "iris index 1 appears in multiple different pairs");
    }

    #[test]
    fn complex_rejects_duplicate_labels() {
        let o = opts(
            r#"
            [shares_generator.FromFile]
            path_to_ndjson_file = "/tmp/test.ndjson"
            [request_batch.Complex]
            batches = [[
                { label = "dup", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                { label = "dup", payload = { Uniqueness = { iris_pair = [{ index = 3 }, { index = 4 }] } } },
            ]]
        "#,
        );
        assert_invalid_options(&o, "duplicate label");
    }

    #[test]
    fn complex_rejects_invalid_parent_label() {
        let o = opts(
            r#"
            [shares_generator.FromFile]
            path_to_ndjson_file = "/tmp/test.ndjson"
            [request_batch.Complex]
            batches = [[
                { label = "U-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                { label = "D-0", payload = { IdentityDeletion = { parent = "deadbeef" } } },
            ]]
        "#,
        );
        assert_invalid_options(&o, "parent label 'deadbeef' that is not found in labels");
    }

    #[test]
    fn complex_rejects_parent_in_later_batch() {
        let o = opts(
            r#"
            [shares_generator.FromFile]
            path_to_ndjson_file = "/tmp/test.ndjson"
            [request_batch.Complex]
            batches = [
                [
                    { label = "D-0", payload = { IdentityDeletion = { parent = "U-0" } } },
                ],
                [
                    { label = "U-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                ],
            ]
        "#,
        );
        assert_invalid_options(&o, "parent 'U-0' must be in an earlier batch");
    }

    #[test]
    fn complex_rejects_parent_in_same_batch() {
        let o = opts(
            r#"
            [shares_generator.FromFile]
            path_to_ndjson_file = "/tmp/test.ndjson"
            [request_batch.Complex]
            batches = [[
                { label = "U-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                { label = "D-0", payload = { IdentityDeletion = { parent = "U-0" } } },
            ]]
        "#,
        );
        assert_invalid_options(&o, "parent 'U-0' must be in an earlier batch");
    }

    #[test]
    fn complex_allows_duplicate_iris_pair() {
        let o = opts(
            r#"
            [shares_generator.FromFile]
            path_to_ndjson_file = "/tmp/test.ndjson"
            [request_batch.Complex]
            batches = [
                [
                    { label = "U-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                    { label = "U-1", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                ],
            ]
        "#,
        );
        o.validate().expect("duplicate pair should be allowed");
    }

    #[test]
    fn complex_allows_swapped_iris_pair() {
        let o = opts(
            r#"
            [shares_generator.FromFile]
            path_to_ndjson_file = "/tmp/test.ndjson"
            [request_batch.Complex]
            batches = [
                [
                    { label = "U-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                    { label = "U-1", payload = { Uniqueness = { iris_pair = [{ index = 2 }, { index = 1 }] } } },
                ],
            ]
        "#,
        );
        o.validate().expect("swapped pair should be allowed");
    }

    #[test]
    fn complex_valid_multi_batch_passes() {
        let o = opts(
            r#"
            [shares_generator.FromFile]
            path_to_ndjson_file = "/tmp/test.ndjson"
            [request_batch.Complex]
            batches = [
                [
                    { label = "U-0", payload = { Uniqueness = { iris_pair = [{ index = 1 }, { index = 2 }] } } },
                    { label = "U-1", payload = { Uniqueness = { iris_pair = [{ index = 3 }, { index = 4 }] } } },
                ],
                [
                    { label = "D-0", payload = { IdentityDeletion = { parent = "U-0" } } },
                    { label = "R-0", payload = { Reauthorisation = { iris_pair = [{ index = 5 }, { index = 6 }], parent = "U-1" } } },
                    { label = "RC-0", payload = { ResetCheck = { iris_pair = [{ index = 7 }, { index = 8 }] } } },
                    { label = "RU-0", payload = { ResetUpdate = { iris_pair = [{ index = 9 }, { index = 10 }], parent = "U-0" } } },
                ],
            ]
        "#,
        );
        o.validate().expect("should pass validation");
    }
}
