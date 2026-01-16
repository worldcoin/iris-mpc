use serde::{Deserialize, Serialize};

use iris_mpc_common::IrisSerialId;

use super::super::typeset::RequestBatch;

/// Service client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceClientConfiguration {
    // Associated request batch generation configuration.
    request_batch: RequestBatchConfiguration,

    // Associated Iris shares generator configuration.
    shares_generator: SharesGeneratorConfiguration,
}

impl ServiceClientConfiguration {
    pub fn request_batch(&self) -> &RequestBatchConfiguration {
        &self.request_batch
    }

    pub fn shares_generator(&self) -> &SharesGeneratorConfiguration {
        &self.shares_generator
    }
}

/// Set of variants over inputs to request batch generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestBatchConfiguration {
    // Batches of single request type
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
    /// A pre-built known set of request batches.
    KnownSet(Vec<RequestBatch>),
}

/// Set of variants over inputs to iris shares generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharesGeneratorConfiguration {
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
        selection_strategy: Option<IrisCodeSelectionStrategy>,
    },
}

/// Enumeration over types of strategy to apply when selecting
/// Iris codes from an NDJSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrisCodeSelectionStrategy {
    // All Iris codes are selected.
    All,
    // Every other Iris code is selected beginning at an even offset.
    Even,
    // Every other Iris code is selected beginning at an odd offset.
    Odd,
}
