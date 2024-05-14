use std::ops::BitXor;

use aes_prng::AesRng;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};

pub type PrfSeed = <AesRng as SeedableRng>::Seed;

pub struct Prf {
    my_prf: AesRng,
    prev_prf: AesRng,
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
    pub fn new(my_key: PrfSeed, next_key: PrfSeed) -> Self {
        Self {
            my_prf: AesRng::from_seed(my_key),
            prev_prf: AesRng::from_seed(next_key),
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

    pub fn gen_binary_zero_share<T: BitXor<Output = T>>(&mut self) -> T
    where
        Standard: Distribution<T>,
    {
        let (a, b) = self.gen_rands::<T>();
        a ^ b
    }
}
