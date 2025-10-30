use crate::shares::{int_ring::IntRing2k, ring_impl::RingElement};
use aes_prng::AesRng;
use eyre::Result;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};

#[cfg(not(feature = "chacha_prf"))]
type PrfRng = aes_prng::AesRng;

#[cfg(feature = "chacha_prf")]
type PrfRng = rand_chacha::ChaCha20Rng;

pub type PrfSeed = [u8; 16];

#[derive(Clone, Debug)]
pub struct Prf {
    pub my_prf: PrfRng,
    pub prev_prf: PrfRng,
}

impl Default for Prf {
    fn default() -> Self {
        Self {
            my_prf: PrfRng::from_entropy(),
            prev_prf: PrfRng::from_entropy(),
        }
    }
}

impl Prf {
    #[cfg(feature = "chacha_prf")]
    pub fn new(my_key: PrfSeed, prev_key: PrfSeed) -> Self {
        Self {
            my_prf: PrfRng::from_seed(Self::expand_seed(my_key)),
            prev_prf: PrfRng::from_seed(Self::expand_seed(prev_key)),
        }
    }

    #[cfg(not(feature = "chacha_prf"))]
    pub fn new(my_key: PrfSeed, prev_key: PrfSeed) -> Self {
        Self {
            my_prf: PrfRng::from_seed(my_key),
            prev_prf: PrfRng::from_seed(prev_key),
        }
    }

    #[cfg(feature = "chacha_prf")]
    fn expand_seed(seed: PrfSeed) -> [u8; 32] {
        use blake3::Hasher;
        let mut h = Hasher::new();
        h.update(&seed);
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(digest.as_bytes());
        out
    }

    pub fn get_my_prf(&mut self) -> &mut PrfRng {
        &mut self.my_prf
    }

    pub fn get_prev_prf(&mut self) -> &mut PrfRng {
        &mut self.prev_prf
    }

    pub fn gen_seed() -> PrfSeed {
        let mut rng = PrfRng::from_entropy();
        rng.gen::<PrfSeed>()
    }

    pub fn gen_rands<T>(&mut self) -> (T, T)
    where
        Standard: Distribution<T>,
    {
        let a = self.my_prf.gen::<T>();
        let b = self.prev_prf.gen::<T>();
        (a, b)
    }

    pub fn gen_zero_share<T: IntRing2k>(&mut self) -> RingElement<T>
    where
        Standard: Distribution<T>,
    {
        let (a, b) = self.gen_rands::<RingElement<T>>();
        a - b
    }

    pub fn gen_binary_zero_share<T: IntRing2k>(&mut self) -> RingElement<T>
    where
        Standard: Distribution<T>,
    {
        let (a, b) = self.gen_rands::<RingElement<T>>();
        a ^ b
    }

    fn gen_u32_mod(&mut self, modulus: u32) -> Result<u32> {
        if modulus == 0 {
            eyre::bail!("modulus must be non-zero");
        }
        let modulus_64 = modulus as u64;
        // Rejection sampling to avoid modulo bias
        // The rejection bound is the largest multiple of modulus that fits in u64 - 1.
        // In this case, the probability of rejection is 2^64 % modulus / 2^64 < 2^(-32).
        let rejection_bound = u64::MAX - (u64::MAX % modulus_64 + 1) % modulus_64;
        loop {
            let v = self.my_prf.gen::<u64>();
            if v <= rejection_bound {
                return Ok((v % modulus_64) as u32);
            }
        }
    }

    pub fn gen_permutation(&mut self, size: u32) -> Result<Vec<u32>> {
        let mut perm: Vec<u32> = (0..size).collect();
        for i in 1..size {
            let j = self.gen_u32_mod(i + 1)?;
            perm.swap(i as usize, j as usize);
        }
        Ok(perm)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use statrs::distribution::{ChiSquared, ContinuousCDF};

    use super::*;

    // Chi-square test for uniformity with the significance level = 10^-6
    fn chi_squared_test(observed: &[u32], expected: u32) -> Result<bool> {
        if observed.len() < 2 {
            eyre::bail!("Need at least two bins for chi-squared test");
        }
        let degrees_of_freedom = observed.len() - 1;
        let expected_f = expected as f64;
        let chi2: f64 = observed
            .iter()
            .map(|o| {
                let diff = *o as f64 - expected_f;
                diff * diff / expected_f
            })
            .sum();

        // Significance level
        let alpha = 1e-6;
        let chi_squared_dist = ChiSquared::new(degrees_of_freedom as f64)?;
        let critical_value = chi_squared_dist.inverse_cdf(1.0 - alpha);

        Ok(chi2 < critical_value)
    }

    #[test]
    fn test_gen_u32_mod() -> Result<()> {
        let mut prf = Prf::default();

        // Expected count for values in each bin
        let expected = 1000;

        let mut helper = |modulus: u32| -> Result<()> {
            let mut counters = vec![0_u32; modulus as usize];
            let num_samples = modulus * expected;
            for _ in 0..num_samples {
                let v = prf.gen_u32_mod(modulus)?;
                counters[v as usize] += 1;
            }

            assert!(chi_squared_test(&counters, expected)?);

            Ok(())
        };
        helper(2)?;
        helper(7)?;
        helper(101)?;

        Ok(())
    }

    #[test]
    fn test_gen_permutation() -> Result<()> {
        let mut prf = Prf::default();
        // Expected count for each permutation
        let expected = 100;

        let mut helper = |size: u32| -> Result<()> {
            let num_bins: u32 = (2..=size).product();
            let num_samples = num_bins * expected;

            let mut perm_stats = HashMap::new();
            for _ in 0..num_samples {
                let perm = prf.gen_permutation(size)?;
                *perm_stats.entry(perm).or_insert(0_u32) += 1;
            }

            // Check that all permutations have been generated.
            assert_eq!(perm_stats.len() as u32, num_bins);

            let counters: Vec<u32> = perm_stats.values().cloned().collect();
            assert!(chi_squared_test(&counters, expected)?);

            Ok(())
        };
        helper(2)?;
        helper(4)?;
        helper(5)?;

        Ok(())
    }
}
