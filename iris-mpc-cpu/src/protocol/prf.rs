use crate::shares::{int_ring::IntRing2k, ring_impl::RingElement};
use aes_prng::AesRng;
use eyre::Result;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};

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

    fn gen_u32_mod(&mut self, modulus: u32) -> Result<u32> {
        if modulus == 0 {
            eyre::bail!("modulus must be non-zero");
        }
        let modulus_64 = modulus as u64;
        // Rejection sampling to avoid modulo bias
        // The rejection bound is the largest multiple of modulus that fits in u64 - 1.
        // In this case, the probability of rejection is 2^64 % modulus / 2^64 < 2^32.
        let rejection_bound = u64::MAX - (u64::MAX % modulus_64 + 1) % modulus_64;
        loop {
            let v = self.my_prf.gen::<u64>();
            if v <= rejection_bound {
                return Ok((v % modulus_64) as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chi_statistics(observed: &[u32], expected: u32) -> f64 {
        let expected_f = expected as f64;
        observed
            .iter()
            .map(|o| {
                let diff = *o as f64 - expected_f;
                diff * diff / expected_f
            })
            .sum()
    }

    #[test]
    fn test_prf_gen_u32_mod() -> Result<()> {
        let mut prf = Prf::default();

        let modulus = 100_u32;
        let mut counters = vec![0_u32; modulus as usize];
        let expected = 1000;
        let num_samples = modulus * expected;
        for _ in 0..num_samples {
            let v = prf.gen_u32_mod(modulus)?;
            counters[v as usize] += 1;
        }

        // Chi-square test for uniformity with the following parameters:
        // - Degrees of freedom = modulus - 1 = 99
        // - Significance level = 0.001
        // Critical value is taken from chi-square distribution table here:
        // https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
        let chi2 = chi_statistics(&counters, expected);
        eprintln!("Chi-square statistic: {}", chi2);
        assert!(chi2 < 149.449);

        Ok(())
    }
}
