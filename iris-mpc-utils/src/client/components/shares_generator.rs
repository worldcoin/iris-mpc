use std::path::PathBuf;

use async_trait::async_trait;
use itertools::Itertools;
use rand::{CryptoRng, Rng, SeedableRng};

use iris_mpc_cpu::{execution::hawk_main::BothEyes, protocol::shared_iris::GaloisRingSharedIris};

use super::super::{
    config::IrisCodeSelectionStrategy,
    typeset::{Initialize, ServiceClientError},
};
use crate::{
    constants::N_PARTIES,
    irises::{generate_iris_shares, reader::read_iris_shares},
};

// Add this enum to wrap both generator types
pub(crate) enum SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    FromCpu(SharesGeneratorFromCpu<R>),
    FromFile(SharesGeneratorFromFile<R>),
}

// Implement GenerateShares for the enum so it can be used polymorphically
impl<R> SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    pub(crate) fn new(opts: SharesGeneratorOptions<R>) -> Self
    where
        R: Rng + CryptoRng + SeedableRng + Send,
    {
        match opts {
            SharesGeneratorOptions::FromFile {
                path_to_ndjson_file,
                rng,
                selection_strategy,
            } => Self::FromFile(SharesGeneratorFromFile::new(
                path_to_ndjson_file,
                rng,
                selection_strategy,
            )),
            SharesGeneratorOptions::FromRng { rng } => {
                Self::FromCpu(SharesGeneratorFromCpu::new(rng))
            }
        }
    }

    /// Generates pairs of Iris shares for upstream processing.
    pub(crate) fn generate(&mut self) -> BothEyes<[GaloisRingSharedIris; N_PARTIES]> {
        match self {
            Self::FromCpu(generator) => [generator.generate(), generator.generate()],
            Self::FromFile(generator) => [generator.generate(), generator.generate()],
        }
    }
}

#[async_trait]
impl<R> Initialize for SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    async fn init(&mut self) -> Result<(), ServiceClientError> {
        match self {
            Self::FromCpu(_) => {}
            Self::FromFile(_generator) => {
                println!("TODO: initialise Iris shares stream reader");
            }
        }

        Ok(())
    }
}

/// A shares generator that computes Iris shares on the fly.
pub(crate) struct SharesGeneratorFromCpu<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    // A random number generator acting as an entropy source.
    rng: R,
}

impl<R> SharesGeneratorFromCpu<R>
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

    // Path to an NDJSON file.
    path_to_ndjson_file: PathBuf,

    // A random number generator acting as an entropy source.
    rng: R,

    // Strategy to apply in respect of Iris code selection.
    #[allow(dead_code)]
    selection_strategy: IrisCodeSelectionStrategy,
}

impl<R> SharesGeneratorFromFile<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    const BATCH_SIZE: usize = 100;

    fn new(
        path_to_ndjson_file: PathBuf,
        rng: R,
        selection_strategy: IrisCodeSelectionStrategy,
    ) -> Self {
        Self {
            batch_count: 0,
            batch_size: Self::BATCH_SIZE,
            path_to_ndjson_file,
            rng,
            selection_strategy,
            batch: Vec::with_capacity(Self::BATCH_SIZE),
        }
    }

    fn generate(&mut self) -> [GaloisRingSharedIris; N_PARTIES] {
        if self.batch.is_empty() {
            self.set_next_batch();
        }
        if self.batch.is_empty() {
            panic!("Iris shares generator exhaustion error");
        }

        self.batch.pop().unwrap()
    }

    fn set_next_batch(&mut self) {
        self.batch = read_iris_shares(
            &self.path_to_ndjson_file,
            Self::BATCH_SIZE * self.batch_count,
            Self::BATCH_SIZE,
            &mut self.rng,
        )
        .unwrap()
        .collect_vec();
        self.batch_count += 1;
    }
}

/// Set of variants over share generation inputs.
pub(crate) enum SharesGeneratorOptions<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    /// Shares are generated from a pre-built file.
    FromFile {
        // Path to an NDJSON file.
        path_to_ndjson_file: PathBuf,

        // A random number generator acting as an entropy source.
        rng: R,

        // Strategy to apply in respect of Iris code selection.
        selection_strategy: IrisCodeSelectionStrategy,
    },
    /// Shares are generated via a random number generator.
    FromRng {
        // A random number generator acting as an entropy source.
        rng: R,
    },
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGeneratorOptions<R> {
    pub fn new_file(path_to_ndjson_file: PathBuf, rng_seed: Option<u64>) -> Self {
        Self::FromFile {
            path_to_ndjson_file,
            rng: if let Some(state) = rng_seed {
                R::seed_from_u64(state)
            } else {
                R::from_entropy()
            },
            selection_strategy: IrisCodeSelectionStrategy::All,
        }
    }

    pub fn new_rng(rng_seed: Option<u64>) -> Self {
        Self::FromRng {
            rng: if let Some(state) = rng_seed {
                R::seed_from_u64(state)
            } else {
                R::from_entropy()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

    use super::{SharesGenerator, SharesGeneratorOptions};
    use crate::fsys::local::get_path_to_ndjson;

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGeneratorOptions<R> {
        pub(crate) fn new_1() -> Self {
            Self::new_rng(Some(42))
        }

        pub(crate) fn new_2() -> Self {
            Self::new_file(get_path_to_ndjson(), Some(42))
        }
    }

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
        pub(crate) fn new_1() -> Self {
            Self::new(SharesGeneratorOptions::new_1())
        }

        pub(crate) fn new_2() -> Self {
            Self::new(SharesGeneratorOptions::new_2())
        }
    }

    #[tokio::test]
    async fn test_new_1() {
        let _ = SharesGenerator::<StdRng>::new_1();
    }

    #[tokio::test]
    async fn test_new_2() {
        let _ = SharesGenerator::<StdRng>::new_2();
    }
}
