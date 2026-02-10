use serde::{Deserialize, Serialize};

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
