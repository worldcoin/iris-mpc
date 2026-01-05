use std::{fs::File, io::BufReader, path::PathBuf};

use rand::{CryptoRng, Rng, SeedableRng};

use iris_mpc_common::galois_engine::degree4::GaloisRingIrisCodeShare;
use iris_mpc_cpu::execution::hawk_main::BothEyes;

use crate::{constants::N_PARTIES, irises::generate_iris_code_and_mask_party_shares_for_both_eyes};

#[derive(Debug)]
pub(crate) enum SharesGenerator<R: Rng + CryptoRng + SeedableRng + Send> {
    /// Generates shares from on the fly compute resource.
    FromRng { rng: R },
    /// Generates shares by reading from a static file.
    #[allow(dead_code)]
    FromFile {
        path_to_ndjson_file: PathBuf,
        current_index: usize,
    },
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
    pub fn new_file(path_to_ndjson_file: PathBuf) -> Self {
        let file = File::open(path_to_ndjson_file.clone()).unwrap();
        let _reader = BufReader::new(file);

        // let numbers: Vec<u32> = reader
        //     .lines()
        //     .filter_map(|line| line.ok())
        //     .filter_map(|line| line.trim().parse::<u32>().ok())
        //     .collect();

        // Ok(Self::File {
        //     numbers,
        //     current_index: 0,
        // })

        Self::FromFile {
            path_to_ndjson_file,
            current_index: 0,
        }
    }

    pub fn new_rng(rng: R) -> Self {
        Self::FromRng { rng }
    }
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesGenerator<R> {
    pub fn generate(&mut self) -> BothEyes<[[GaloisRingIrisCodeShare; N_PARTIES]; 2]> {
        match self {
            Self::FromFile {
                path_to_ndjson_file: _,
                ..
            } => {
                unimplemented!()
            }
            Self::FromRng { rng } => generate_iris_code_and_mask_party_shares_for_both_eyes(rng),
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
