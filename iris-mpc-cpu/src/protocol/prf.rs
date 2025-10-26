use crate::shares::{int_ring::IntRing2k, ring_impl::RingElement};
use rand::rngs::OsRng;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};
use rand_chacha::rand_core;
use rand_core::RngCore;

pub use rand_chacha::ChaCha20Rng as AesRng;

pub type PrfSeed = <AesRng as SeedableRng>::Seed;

#[derive(Clone, Debug)]
pub struct Prf {
    pub my_prf: AesRng,
    pub prev_prf: AesRng,
}

impl Default for Prf {
    fn default() -> Self {
        Self {
            my_prf: AesRng::from_rng(OsRng).expect("OsRng"),
            prev_prf: AesRng::from_rng(OsRng).expect("OsRng"),
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
        let mut s = PrfSeed::default();
        OsRng.fill_bytes(&mut s);
        s
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
}
