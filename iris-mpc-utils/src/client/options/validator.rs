use std::collections::HashSet;

use super::{
    super::typeset::{BatchKind, ServiceClientError},
    types::{RequestBatchOptions, SharesGeneratorOptions},
    ServiceClientOptions,
};

impl ServiceClientOptions {
    pub(crate) fn validate(&self) -> Result<(), ServiceClientError> {
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
            RequestBatchOptions::Complex { batches } => {
                // Error if used alongside compute shares generation.
                if matches!(
                    self.shares_generator(),
                    SharesGeneratorOptions::FromCompute { .. }
                ) {
                    return Err(ServiceClientError::InvalidOptions("RequestBatchOptions::Complex can only be used with SharesGeneratorOptions::FromFile".to_string()));
                }

                // Error if there are duplicate Iris descriptors.
                let indexes = self.request_batch().iris_code_indexes();
                if !indexes.is_empty() {
                    let mut set = HashSet::with_capacity(indexes.len());
                    if !indexes.iter().all(|i| set.insert(i)) {
                        return Err(ServiceClientError::InvalidOptions(
                            "RequestBatchOptions::Complex contains duplicate Iris descriptors "
                                .to_string(),
                        ));
                    }
                }

                // Error if there are duplicate labels.
                let labels = self.request_batch().labels();
                if !labels.is_empty() {
                    let mut set = HashSet::with_capacity(labels.len());
                    if !labels.iter().all(|l| set.insert(l)) {
                        return Err(ServiceClientError::InvalidOptions(
                            "RequestBatchOptions::Complex contains duplicate labels".to_string(),
                        ));
                    }
                }

                // Error if there are invalid parent labels.
                let labels_of_parents = self.request_batch().labels_of_parents();
                if !labels_of_parents.is_empty() {
                    let labels_set: HashSet<_> = labels.iter().collect();
                    for label_of_parent in &labels_of_parents {
                        if !labels_set.contains(label_of_parent) {
                            return Err(ServiceClientError::InvalidOptions(
                                format!(
                                    "RequestBatchOptions::Complex contains a parent label '{}' that is not found in labels",
                                    label_of_parent
                                ),
                            ));
                        }
                    }
                }

                // Error if a child request references a parent in the same or later batch.
                // can't ResetUpdate, Delete, etc a Uniqueness request that is in the same batch
                // todo: change this if the system ever tests inputs which are purposely invalid
                let mut labels_seen = HashSet::new();
                for batch in batches {
                    for item in batch {
                        if let Some(parent_label) = item.label_of_parent() {
                            if !labels_seen.contains(&parent_label) {
                                return Err(ServiceClientError::InvalidOptions(
                                    format!(
                                        "RequestBatchOptions::Complex: parent '{}' must be in an earlier batch",
                                        parent_label
                                    ),
                                ));
                            }
                        }
                    }
                    let batch_labels: HashSet<_> =
                        batch.iter().filter_map(|item| item.label()).collect();
                    labels_seen.extend(batch_labels);
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
                if BatchKind::from_str(batch_kind).is_none() {
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

    // -- Complex error paths --

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
    fn complex_rejects_duplicate_iris_descriptors() {
        // Index 1 appears in both iris pairs (left of first, left of second).
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
        assert_invalid_options(&o, "duplicate Iris descriptors");
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
        assert_invalid_options(&o, "duplicate labels");
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

    // -- Complex happy path --

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
