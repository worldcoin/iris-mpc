use std::path::PathBuf;

use async_trait::async_trait;
use rand::{CryptoRng, Rng};

use iris_mpc_cpu::execution::hawk_main::BothEyes;

use super::super::typeset::GenerateShares;
use crate::irises::generate_iris_code_and_mask_shares_both_eyes as generate_iris_shares;
use crate::types::IrisCodeAndMaskShares;

/// Generates shares from on the fly compute resource.
#[derive(Debug)]
pub(crate) struct SharesGeneratorFromCompute<R: Rng + CryptoRng + Send> {
    /// Entropy source.
    rng: R,
}

impl<R: Rng + CryptoRng + Send> SharesGeneratorFromCompute<R> {
    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    pub(crate) fn new(rng: R) -> Self {
        Self { rng }
    }
}

#[async_trait]
impl<R: Rng + CryptoRng + Send> GenerateShares for SharesGeneratorFromCompute<R> {
    async fn generate(&mut self) -> BothEyes<IrisCodeAndMaskShares> {
        generate_iris_shares(self.rng_mut())
    }
}

/// Generates shares by reading from a static file.
#[derive(Debug)]
pub(crate) struct SharesGeneratorFromFile {
    /// Path to an NDJSON file containing pre computed shares.
    path_to_ndjson_file: PathBuf,
}

impl SharesGeneratorFromFile {
    pub(crate) fn new(path_to_ndjson_file: PathBuf) -> Self {
        Self {
            path_to_ndjson_file,
        }
    }
}

#[async_trait]
impl GenerateShares for SharesGeneratorFromFile {
    async fn generate(&mut self) -> BothEyes<IrisCodeAndMaskShares> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::{SharesGeneratorFromCompute, SharesGeneratorFromFile};
    use crate::fsys::local::get_path_to_ndjson;

    impl SharesGeneratorFromCompute<StdRng> {
        pub(crate) fn new_1() -> Self {
            Self::new(StdRng::from_entropy())
        }
    }

    impl SharesGeneratorFromFile {
        pub(crate) fn new_1() -> Self {
            Self::new(get_path_to_ndjson())
        }
    }

    #[tokio::test]
    async fn test_new_from_compute() {
        let _ = SharesGeneratorFromCompute::<StdRng>::new_1();
    }

    #[tokio::test]
    async fn test_new_from_file() {
        let _ = SharesGeneratorFromFile::new_1();
    }
}
