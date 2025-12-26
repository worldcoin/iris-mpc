use std::path::PathBuf;

use rand::{CryptoRng, Rng, SeedableRng};

use iris_mpc_cpu::execution::hawk_main::BothEyes;

use crate::{
    irises::generate_iris_code_and_mask_shares_both_eyes as generate_iris_shares,
    types::IrisCodeAndMaskShares,
};

#[derive(Debug)]
pub(crate) enum SharesGenerator<R: Rng + CryptoRng + SeedableRng + Send> {
    /// Generates shares from on the fly compute resource.
    FromRng { rng: R },
    /// Generates shares by reading from a static file.
    #[allow(dead_code)]
    FromFile { path_to_ndjson_file: PathBuf },
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
    pub fn new_file(path_to_ndjson_file: PathBuf) -> Self {
        Self::FromFile {
            path_to_ndjson_file,
        }
    }

    pub fn new_rng(rng: R) -> Self {
        Self::FromRng { rng }
    }
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
    pub fn generate(&mut self) -> BothEyes<IrisCodeAndMaskShares> {
        match self {
            Self::FromFile {
                path_to_ndjson_file: _,
            } => {
                unimplemented!()
            }
            Self::FromRng { rng } => generate_iris_shares(rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

    use super::SharesGenerator;
    use crate::fsys::local::get_path_to_ndjson;

    impl SharesGenerator<StdRng> {
        pub(crate) fn new_1() -> Self {
            Self::new_rng(StdRng::from_entropy())
        }
    }

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
        pub(crate) fn new_2() -> Self {
            Self::new_file(get_path_to_ndjson())
        }
    }

    #[tokio::test]
    async fn test_new_from_compute() {
        let _ = SharesGenerator::new_1();
    }

    // #[tokio::test]
    // async fn test_new_from_file<R: Rng + CryptoRng + SeedableRng + Send>() {
    //     let _: SharesGenerator1<R> = SharesGenerator1::new_2();
    // }
}
