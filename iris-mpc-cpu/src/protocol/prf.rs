use crate::{
    protocol::shuffle::Permutation,
    shares::{int_ring::IntRing2k, ring_impl::RingElement},
};
use aes_prng::AesRng;
use eyre::Result;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};

/// Generate a uniformly random u32 in [0, modulus)
fn gen_u32_mod(rng: &mut AesRng, modulus: u32) -> Result<u32> {
    if modulus == 0 {
        eyre::bail!("modulus must be non-zero");
    }
    let modulus_64 = modulus as u64;
    // Rejection sampling to avoid modulo bias
    // The rejection bound is the largest multiple of modulus that fits in u64 - 1.
    // In this case, the probability of rejection is 2^64 % modulus / 2^64 < 2^(-32).
    let rejection_bound = u64::MAX - (u64::MAX % modulus_64 + 1) % modulus_64;
    loop {
        let v = rng.gen::<u64>();
        if v <= rejection_bound {
            return Ok((v % modulus_64) as u32);
        }
    }
}

pub type PrfSeed = <AesRng as SeedableRng>::Seed;

#[derive(Clone, Debug)]
pub struct Prf {
    pub my_prf: AesRng,
    pub prev_prf: AesRng,
}

impl Default for Prf {
    fn default() -> Self {
        Self {
            my_prf: AesRng::from_entropy(),
            prev_prf: AesRng::from_entropy(),
        }
    }
}

impl Prf {
    pub fn new(my_key: PrfSeed, prev_key: PrfSeed) -> Self {
        Self {
            my_prf: AesRng::from_seed(my_key),
            prev_prf: AesRng::from_seed(prev_key),
        }
    }

    pub fn get_my_prf(&mut self) -> &mut AesRng {
        &mut self.my_prf
    }

    pub fn get_prev_prf(&mut self) -> &mut AesRng {
        &mut self.prev_prf
    }

    pub fn gen_seed() -> PrfSeed {
        let mut rng = AesRng::from_entropy();
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

    // Generates shared random u32 in [0, modulus)
    fn gen_u32_mod(&mut self, modulus: u32) -> Result<(u32, u32)> {
        let a = gen_u32_mod(&mut self.my_prf, modulus)?;
        let b = gen_u32_mod(&mut self.prev_prf, modulus)?;
        Ok((a, b))
    }

    pub fn gen_permutation(&mut self, size: u32) -> Result<Permutation> {
        let mut perm_a: Vec<u32> = (0..size).collect();
        let mut perm_b: Vec<u32> = (0..size).collect();
        for i in 1..size {
            let (j_a, j_b) = self.gen_u32_mod(i + 1)?;
            perm_a.swap(i as usize, j_a as usize);
            perm_b.swap(i as usize, j_b as usize);
        }
        Ok((perm_a, perm_b))
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
            let mut counters_a = vec![0_u32; modulus as usize];
            let mut counters_b = vec![0_u32; modulus as usize];
            let num_samples = modulus * expected;
            for _ in 0..num_samples {
                let (v_a, v_b) = prf.gen_u32_mod(modulus)?;
                counters_a[v_a as usize] += 1;
                counters_b[v_b as usize] += 1;
            }

            assert!(chi_squared_test(&counters_a, expected)?);
            assert!(chi_squared_test(&counters_b, expected)?);

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
            let num_samples = num_bins * expected / 2;

            let mut perm_stats = HashMap::new();
            for _ in 0..num_samples {
                let perm = prf.gen_permutation(size)?;
                *perm_stats.entry(perm.0).or_insert(0_u32) += 1;
                *perm_stats.entry(perm.1).or_insert(0_u32) += 1;
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
