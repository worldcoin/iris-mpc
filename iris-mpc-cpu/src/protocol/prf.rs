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
    pub fn new(my_key: [u8; 16], prev_key: [u8; 16]) -> Self {
        let my_seed = Self::expand_seed(my_key);
        let prev_seed = Self::expand_seed(prev_key);
        Self {
            my_prf: AesRng::from_seed(my_seed),
            prev_prf: AesRng::from_seed(prev_seed),
        }
    }

    pub fn gen_seed() -> [u8; 16] {
        let mut s = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut s);
        s
    }

    fn expand_seed(short: [u8; 16]) -> [u8; 32] {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(&short);
        let hash = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(hash.as_bytes());
        out
    }
    pub fn get_my_prf(&mut self) -> &mut AesRng {
        &mut self.my_prf
    }

    pub fn get_prev_prf(&mut self) -> &mut AesRng {
        &mut self.prev_prf
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
