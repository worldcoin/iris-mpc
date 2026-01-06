use std::path::PathBuf;

use async_trait::async_trait;
use rand::{CryptoRng, Rng, SeedableRng};

use iris_mpc_common::galois_engine::degree4::GaloisRingIrisCodeShare;
use iris_mpc_cpu::execution::hawk_main::BothEyes;

use super::super::{
    config::IrisCodeSelectionStrategy,
    typeset::{Initialize, ServiceClientError},
};
use crate::{constants::N_PARTIES, irises::generate_iris_code_and_mask_party_shares_for_both_eyes};

/// Encapsulates logic for generating Iris shares.
#[derive(Debug)]
pub(crate) struct SharesGenerator<R: Rng + CryptoRng + SeedableRng + Send> {
    // Parameters determining how shares are generated.
    params: SharesGeneratorParams<R>,
}

#[async_trait]
impl<R: Rng + CryptoRng + SeedableRng + Send> Initialize for SharesGenerator<R> {
    async fn init(&mut self) -> Result<(), ServiceClientError> {
        match self.params {
            SharesGeneratorParams::FromFile { .. } => {
                unimplemented!()
            }
            _ => Ok(()),
        }
    }
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
    pub(crate) fn new(params: SharesGeneratorParams<R>) -> Self {
        Self { params }
    }

    pub fn generate(&mut self) -> BothEyes<[[GaloisRingIrisCodeShare; N_PARTIES]; 2]> {
        match &mut self.params {
            SharesGeneratorParams::FromFile {
                path_to_ndjson_file: _,
                ..
            } => {
                unimplemented!()
            }
            SharesGeneratorParams::FromRng { rng } => {
                generate_iris_code_and_mask_party_shares_for_both_eyes(rng)
            }
        }
    }
}

/// Set of variants over share generation inputs.
#[derive(Debug)]
pub(crate) enum SharesGeneratorParams<R: Rng + CryptoRng + SeedableRng + Send> {
    /// Shares are generated from a pre-built file.
    FromFile {
        // Path to an NDJSON file.
        #[allow(dead_code)]
        path_to_ndjson_file: PathBuf,

        // Strategy to apply in respect of Iris code selection.
        #[allow(dead_code)]
        selection_strategy: IrisCodeSelectionStrategy,
    },
    /// Shares are generated via a random number generator.
    FromRng {
        // A random number generator acting as an entropy source.
        rng: R,
    },
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGeneratorParams<R> {
    pub fn new_file(path_to_ndjson_file: PathBuf) -> Self {
        Self::FromFile {
            path_to_ndjson_file,
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

    use super::{SharesGenerator, SharesGeneratorParams};
    use crate::fsys::local::get_path_to_ndjson;

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGeneratorParams<R> {
        pub(crate) fn new_1() -> Self {
            Self::new_rng(Some(42))
        }

        pub(crate) fn new_2() -> Self {
            Self::new_file(get_path_to_ndjson())
        }
    }

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
        pub(crate) fn new_1() -> Self {
            Self::new(SharesGeneratorParams::new_1())
        }

        pub(crate) fn new_2() -> Self {
            Self::new(SharesGeneratorParams::new_2())
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
