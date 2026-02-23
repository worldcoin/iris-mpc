use std::path::PathBuf;

use itertools::Itertools;
use rand::{CryptoRng, Rng, SeedableRng};

use super::super::typeset::{IrisDescriptor, IrisPairDescriptor};
use crate::{
    client::options::SharesGeneratorOptions,
    constants::N_PARTIES,
    irises::{
        generate_iris_shares_for_upload, generate_iris_shares_mirrored_for_upload,
        reader::read_iris_shares_for_upload, GaloisRingSharedIrisForUpload,
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
        // Iris shares read from an NDJSON file.
        iris_shares: Vec<[GaloisRingSharedIrisForUpload; N_PARTIES]>,
    },
}

impl<R> SharesGenerator<R>
where
    R: Rng + CryptoRng + SeedableRng + Send,
{
    pub fn from_options(opts: SharesGeneratorOptions) -> Self {
        match opts {
            SharesGeneratorOptions::FromCompute { rng_seed } => {
                tracing::info!("Parsing SharesGeneratorOptions::FromCompute");
                SharesGenerator::<R>::new_compute(rng_seed)
            }
            SharesGeneratorOptions::FromFile {
                path_to_ndjson_file,
                rng_seed,
                selection_strategy,
            } => {
                tracing::info!("Parsing SharesGeneratorOptions::FromFile");
                let path = path_to_ndjson_file.as_deref().expect(
                    "FromFile requires path_to_ndjson_file (set via CLI --path-to-iris-shares)",
                );
                SharesGenerator::new_file(PathBuf::from(path), rng_seed, selection_strategy)
            }
        }
    }

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
        let mut rng = rng_seed
            .map(R::seed_from_u64)
            .unwrap_or_else(R::from_entropy);
        let iris_shares = read_iris_shares_for_upload(path_to_ndjson_file.as_path(), &mut rng)
            .unwrap()
            .skip(n_skip)
            .step_by(n_step);

        Self::FromFile {
            iris_shares: iris_shares.collect_vec(),
        }
    }

    /// Generates pairs of Iris shares for upstream processing.
    pub(crate) fn generate(
        &mut self,
        iris_pair: Option<&IrisPairDescriptor>,
    ) -> BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]> {
        let (maybe_left_desc, maybe_right_desc) = match iris_pair {
            Some(descriptor) => (Some(descriptor.left()), Some(descriptor.right())),
            None => (None, None),
        };

        [
            self.generate_single(maybe_left_desc),
            self.generate_single(maybe_right_desc),
        ]
    }

    /// Generates a single set of Iris shares for upstream processing.
    fn generate_single(
        &mut self,
        maybe_iris_descriptor: Option<&IrisDescriptor>,
    ) -> [GaloisRingSharedIrisForUpload; N_PARTIES] {
        match self {
            Self::FromCompute { rng } => generate_iris_shares_for_upload(rng, None),
            Self::FromFile { iris_shares } => match maybe_iris_descriptor {
                Some(iris_descriptor) => iris_shares[iris_descriptor.index()].clone(),
                None => iris_shares.pop().expect("Shares generator is exhausted"),
            },
        }
    }

    /// Generates pairs of mirrored Iris shares for upstream processing.
    /// this function swaps the irises too, so that the user doesn't have to.
    pub(crate) fn generate_mirrored(
        &mut self,
        iris_pair: Option<&IrisPairDescriptor>,
    ) -> BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]> {
        let (maybe_left_desc, maybe_right_desc) = match iris_pair {
            Some(descriptor) => (Some(descriptor.right()), Some(descriptor.left())),
            None => (None, None),
        };

        [
            self.generate_single_mirrored(maybe_left_desc),
            self.generate_single_mirrored(maybe_right_desc),
        ]
    }

    /// Generates a single set of mirrored Iris shares for upstream processing.
    fn generate_single_mirrored(
        &mut self,
        maybe_iris_descriptor: Option<&IrisDescriptor>,
    ) -> [GaloisRingSharedIrisForUpload; N_PARTIES] {
        match self {
            Self::FromCompute { rng } => generate_iris_shares_mirrored_for_upload(rng, None),
            Self::FromFile { iris_shares } => {
                let base = match maybe_iris_descriptor {
                    Some(iris_descriptor) => iris_shares[iris_descriptor.index()].clone(),
                    None => iris_shares.pop().expect("Shares generator is exhausted"),
                };
                base.map(|share| GaloisRingSharedIrisForUpload {
                    code: share.code.mirrored_code(),
                    mask: share.mask.mirrored_mask(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

    use super::SharesGenerator;

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
        pub(crate) fn new_compute_1() -> Self {
            Self::new_compute(Some(42))
        }
    }

    #[test]
    fn test_new_compute_1() {
        let mut generator = SharesGenerator::<StdRng>::new_compute_1();
        for _ in 0..10 {
            let _ = generator.generate(None);
        }
    }
}
