use std::path::PathBuf;

use itertools::Itertools;
use rand::{CryptoRng, Rng, SeedableRng};

use crate::{
    constants::N_PARTIES,
    irises::{
        generate_iris_shares_for_upload, reader::read_iris_shares_for_upload,
        GaloisRingSharedIrisForUpload,
    },
};
use iris_mpc_cpu::{
    execution::hawk_main::BothEyes, utils::serialization::iris_ndjson::IrisSelection,
};

/// Generates Iris shares either from computation or file system.
pub(crate) enum SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    FromCompute {
        // A random number generator acting as an entropy source.
        rng: R,
    },
    FromFile {
        // Current batch of Iris shares read from file system.
        batch: Vec<[GaloisRingSharedIrisForUpload; N_PARTIES]>,

        // Count of cached batches.
        batch_count: usize,

        // Number of lines to skip after having read a chunk from NDJSON file.
        n_skip: usize,

        // Number of lines to step over having read a chunk from NDJSON file.
        n_step: usize,

        // Path to an NDJSON file.
        path_to_ndjson_file: PathBuf,

        // A random number generator acting as an entropy source.
        rng: R,
    },
}

impl<R> SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    const READ_BUFFER_SIZE: usize = 100;

    pub fn new_compute(rng_seed: Option<u64>) -> Self {
        Self::FromCompute {
            rng: rng_seed
                .map(R::seed_from_u64)
                .unwrap_or_else(R::from_entropy),
        }
    }

    pub fn new_file(
        path_to_ndjson_file: PathBuf,
        rng_seed: Option<u64>,
        selection_strategy: Option<IrisSelection>,
    ) -> Self {
        let (n_skip, n_step) = selection_strategy
            .unwrap_or(IrisSelection::All)
            .skip_and_step();

        Self::FromFile {
            batch: Vec::with_capacity(Self::READ_BUFFER_SIZE),
            batch_count: 0,
            n_skip,
            n_step,
            path_to_ndjson_file,
            rng: rng_seed
                .map(R::seed_from_u64)
                .unwrap_or_else(R::from_entropy),
        }
    }

    /// Generates pairs of Iris shares for upstream processing.
    pub(crate) fn generate(&mut self) -> BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]> {
        [self.generate_single(), self.generate_single()]
    }

    fn generate_single(&mut self) -> [GaloisRingSharedIrisForUpload; N_PARTIES] {
        match self {
            Self::FromCompute { rng } => generate_iris_shares_for_upload(rng, None),
            Self::FromFile {
                batch,
                batch_count,
                n_skip,
                n_step,
                path_to_ndjson_file,
                rng,
            } => {
                if batch.is_empty() {
                    // TODO: revisit skip/take ... etc.
                    *batch = read_iris_shares_for_upload(path_to_ndjson_file, rng)
                        .unwrap()
                        .skip(Self::READ_BUFFER_SIZE * *batch_count)
                        .take(Self::READ_BUFFER_SIZE)
                        .skip(*n_skip)
                        .step_by(*n_step)
                        .collect_vec();
                    *batch_count += 1;
                }
                batch.pop().expect("Shares generator is exhausted")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

    use super::SharesGenerator;
    use crate::fsys::local::get_path_to_ndjson;

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
        pub(crate) fn new_compute_1() -> Self {
            Self::new_compute(Some(42))
        }

        pub(crate) fn new_file_1() -> Self {
            Self::new_file(get_path_to_ndjson(), Some(42), None)
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
        // TODO: invoke the generate function and fix stack overflow error.
    }
}
