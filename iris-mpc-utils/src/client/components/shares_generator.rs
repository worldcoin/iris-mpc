use std::path::PathBuf;

use itertools::Itertools;
use rand::{CryptoRng, Rng, SeedableRng};

use crate::{
    constants::N_PARTIES,
    irises::{generate_iris_shares, reader::read_iris_shares},
};
use iris_mpc_cpu::{
    execution::hawk_main::BothEyes, protocol::shared_iris::GaloisRingSharedIris,
    utils::serialization::iris_ndjson::IrisSelection,
};

// Generates Iris shares either from CPU of file system.
pub(crate) enum SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    FromCompute(SharesGeneratorFromCompute<R>),
    FromFile(SharesGeneratorFromFile<R>),
}

// Implement GenerateShares for the enum so it can be used polymorphically
impl<R> SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    /// Generates pairs of Iris shares for upstream processing.
    pub(crate) fn generate(&mut self) -> BothEyes<[GaloisRingSharedIris; N_PARTIES]> {
        match self {
            Self::FromCompute(generator) => [generator.generate(), generator.generate()],
            Self::FromFile(generator) => [generator.generate(), generator.generate()],
        }
    }
}

/// Set of variants over share generation inputs.
pub(crate) enum SharesGeneratorOptions<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    /// Shares are generated via a random number generator.
    FromCompute {
        // A random number generator acting as an entropy source.
        rng: R,
    },
    /// Shares are generated from a pre-built file.
    FromFile {
        // Path to an NDJSON file.
        path_to_ndjson_file: PathBuf,

        // A random number generator acting as an entropy source.
        rng: R,

        // Strategy to apply in respect of Iris code selection.
        selection_strategy: IrisSelection,
    },
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGeneratorOptions<R> {
    pub fn new_compute(rng_seed: Option<u64>) -> Self {
        Self::FromCompute {
            rng: if let Some(state) = rng_seed {
                R::seed_from_u64(state)
            } else {
                R::from_entropy()
            },
        }
    }

    pub fn new_file(
        path_to_ndjson_file: PathBuf,
        rng_seed: Option<u64>,
        selection_strategy: Option<IrisSelection>,
    ) -> Self {
        Self::FromFile {
            path_to_ndjson_file,
            rng: if let Some(state) = rng_seed {
                R::seed_from_u64(state)
            } else {
                R::from_entropy()
            },
            selection_strategy: match selection_strategy {
                None => IrisSelection::All,
                Some(inner) => inner,
            },
        }
    }
}

impl<R> From<SharesGeneratorOptions<R>> for SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    fn from(opts: SharesGeneratorOptions<R>) -> Self {
        match opts {
            SharesGeneratorOptions::FromCompute { rng } => {
                Self::FromCompute(SharesGeneratorFromCompute::new(rng))
            }
            SharesGeneratorOptions::FromFile {
                path_to_ndjson_file,
                rng,
                selection_strategy,
            } => Self::FromFile(SharesGeneratorFromFile::new(
                path_to_ndjson_file,
                rng,
                selection_strategy,
            )),
        }
    }
}

/// A shares generator that computes Iris shares on the fly.
pub(crate) struct SharesGeneratorFromCompute<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    // A random number generator acting as an entropy source.
    rng: R,
}

impl<R> SharesGeneratorFromCompute<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    fn new(rng: R) -> Self {
        Self { rng }
    }

    fn generate(&mut self) -> [GaloisRingSharedIris; N_PARTIES] {
        // Pass through to sink function.
        generate_iris_shares(&mut self.rng, None)
    }
}

/// A shares generator that reads Iris codes from file and then computes shares on the fly.
pub(crate) struct SharesGeneratorFromFile<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    // Current batch if available Iris shares read from file system.
    batch: Vec<[GaloisRingSharedIris; N_PARTIES]>,

    // Count of cached batches.
    batch_count: usize,

    // Size of each cached batch.
    #[allow(dead_code)]
    batch_size: usize,

    // Number of lines to skip after having read a chunk from NDJSON file.
    n_skip: usize,

    // Number of lines to step over having read a chunk from NDJSON file.
    n_step: usize,

    // Path to an NDJSON file.
    path_to_ndjson_file: PathBuf,

    // A random number generator acting as an entropy source.
    rng: R,
}

impl<R> SharesGeneratorFromFile<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    // Number of lines to read at a time from ndjson file.
    const READ_BUFFER_SIZE: usize = 100;

    fn new(path_to_ndjson_file: PathBuf, rng: R, selection_strategy: IrisSelection) -> Self {
        // Set skip/step after NDJSON chunk is read into working memory.
        let (n_skip, n_step) = selection_strategy.skip_and_step();

        Self {
            batch: Vec::with_capacity(Self::READ_BUFFER_SIZE),
            batch_count: 0,
            batch_size: Self::READ_BUFFER_SIZE,
            n_skip,
            n_step,
            path_to_ndjson_file,
            rng,
        }
    }

    fn generate(&mut self) -> [GaloisRingSharedIris; N_PARTIES] {
        // Read next batch from NDJSON if necessary.
        if self.batch.is_empty() {
            self.batch = self.read_next_batch();
            self.batch_count += 1;
        }

        // Return
        self.batch.pop().expect("Shares generator is exhausted")
    }

    /// Reads next batch of Iris codes from NDJSON file.
    fn read_next_batch(&mut self) -> Vec<[GaloisRingSharedIris; N_PARTIES]> {
        read_iris_shares(
            &self.path_to_ndjson_file,
            self.batch_size * self.batch_count,
            self.batch_size,
            &mut self.rng,
        )
        .unwrap()
        .skip(self.n_skip)
        .step_by(self.n_step)
        .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

    use super::{SharesGenerator, SharesGeneratorOptions};
    use crate::fsys::local::get_path_to_ndjson;

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGeneratorOptions<R> {
        pub(crate) fn new_compute_1() -> Self {
            Self::new_compute(Some(42))
        }

        pub(crate) fn new_file_1() -> Self {
            Self::new_file(get_path_to_ndjson(), Some(42), None)
        }
    }

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
        pub(crate) fn new_compute_1() -> Self {
            Self::from(SharesGeneratorOptions::new_compute_1())
        }

        pub(crate) fn new_file_1() -> Self {
            Self::from(SharesGeneratorOptions::new_file_1())
        }
    }

    #[tokio::test]
    async fn test_new_compute_1() {
        let mut generator = SharesGenerator::<StdRng>::new_compute_1();
        for _ in 0..10 {
            let _ = generator.generate();
        }
    }

    #[tokio::test]
    async fn test_new_file_1() {
        let _ = SharesGenerator::<StdRng>::new_file_1();
    }
}
